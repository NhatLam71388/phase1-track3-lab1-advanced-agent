from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

load_dotenv()

# Đặt USE_MOCK=true trong .env để dùng mock (nhanh, không tốn API)
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not set in .env")
        _client = OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=api_key,
        )
    return _client

def _chat(system: str, user: str) -> str:
    from openai import RateLimitError
    import time

    for attempt in range(5):
        try:
            response = _get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"[rate limit] chờ {wait}s... (lần {attempt + 1}/5)", flush=True)
            time.sleep(wait)

    raise RuntimeError("Vượt quá số lần retry cho API call")

def _parse_json(text: str) -> dict:
    import re
    # Ưu tiên extract JSON object/array nằm giữa { } trong text (bỏ qua markdown wrapper)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text.strip())


# ── Mock fallback (giữ lại để test nhanh không tốn API) ──────────────────────

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

def _mock_actor(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer

def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])

def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)


# ── Real LLM via OpenRouter ───────────────────────────────────────────────────

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list) -> str:
    if USE_MOCK:
        return _mock_actor(example, attempt_id, agent_type, reflection_memory)

    ctx_text = "\n\n".join(
        f"[{chunk.title}]\n{chunk.text}" for chunk in example.context
    )
    reflection_text = ""
    if reflection_memory:
        entries = "\n".join(
            f"Attempt {r.attempt_id}: {r.next_strategy}" for r in reflection_memory
        )
        reflection_text = f"\n\nPrevious attempt feedback (use this to improve):\n{entries}"

    user_msg = (
        f"Context passages:\n{ctx_text}\n\n"
        f"Question: {example.question}{reflection_text}\n\n"
        "Answer:"
    )
    return _chat(ACTOR_SYSTEM, user_msg)


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if USE_MOCK:
        return _mock_evaluator(example, answer)

    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}"
    )
    raw = _chat(EVALUATOR_SYSTEM, user_msg)
    try:
        data = _parse_json(raw)
        return JudgeResult(
            score=int(data.get("score", 0)),
            reason=str(data.get("reason", "")),
            missing_evidence=data.get("missing_evidence", []),
            spurious_claims=data.get("spurious_claims", []),
        )
    except Exception:
        # Fallback: nếu JSON parse thất bại, so sánh thủ công
        is_correct = normalize_answer(example.gold_answer) == normalize_answer(answer)
        return JudgeResult(
            score=1 if is_correct else 0,
            reason=f"JSON parse failed; falling back to exact match. Raw: {raw[:200]}",
        )


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if USE_MOCK:
        return _mock_reflector(example, attempt_id, judge)

    user_msg = (
        f"Question: {example.question}\n"
        f"Attempt {attempt_id} answer was wrong.\n"
        f"Reason: {judge.reason}\n"
        f"Missing evidence: {judge.missing_evidence}\n"
        f"Spurious claims: {judge.spurious_claims}"
    )
    raw = _chat(REFLECTOR_SYSTEM, user_msg)
    try:
        data = _parse_json(raw)
        return ReflectionEntry(
            attempt_id=int(data.get("attempt_id", attempt_id)),
            failure_reason=str(data.get("failure_reason", judge.reason)),
            lesson=str(data.get("lesson", "")),
            next_strategy=str(data.get("next_strategy", "")),
        )
    except Exception:
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Could not parse reflection; retry with more careful multi-hop reasoning.",
            next_strategy="Re-read all context passages and trace each hop explicitly before answering.",
        )
