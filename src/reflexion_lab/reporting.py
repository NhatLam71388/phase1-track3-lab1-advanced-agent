from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2), "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    by_mode: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        by_mode[record.failure_mode][record.agent_type] += 1
    return {mode: dict(agents) for mode, agents in sorted(by_mode.items())}

def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    examples = [{"qid": r.qid, "agent_type": r.agent_type, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": r.failure_mode, "reflection_count": len(r.reflections)} for r in records]
    return ReportPayload(meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})}, summary=summarize(records), failure_modes=failure_breakdown(records), examples=examples, extensions=["structured_evaluator", "reflection_memory", "benchmark_report_json", "mock_mode_for_autograding"], discussion="Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.")

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- `{item}`" for item in report.extensions)

    def bar(ratio: float, width: int = 20) -> str:
        filled = round(ratio * width)
        return "█" * filled + "░" * (width - filled)

    def arrow(v: float, unit: str = "", higher_better: bool = True) -> str:
        if abs(v) < 1e-6:
            return f"±0{unit}"
        sign = "▲" if v > 0 else "▼"
        good = (v > 0) == higher_better
        return f"{sign} {abs(v):.2f}{unit} {'✅' if good else '⚠️'}"

    r_em   = react.get("em", 0)
    x_em   = reflexion.get("em", 0)
    r_n    = react.get("count", 0)
    x_n    = reflexion.get("count", 0)
    r_corr = round(r_em * r_n)
    x_corr = round(x_em * x_n)

    # Failure mode table rows
    fm_rows = ""
    for mode, counts in sorted(report.failure_modes.items()):
        r_cnt = counts.get("react", 0)
        x_cnt = counts.get("reflexion", 0)
        note = {
            "none": "Trả lời đúng ngay lần đầu",
            "entity_drift": "Nhầm entity ở hop thứ 2",
            "incomplete_multi_hop": "Dừng sau hop 1, chưa hoàn thành",
            "wrong_final_answer": "Suy luận đúng nhưng kết luận sai",
            "looping": "Agent lặp lại câu trả lời",
            "reflection_overfit": "Reflection phụ thuộc quá vào hint",
        }.get(mode, "")
        fm_rows += f"| `{mode}` | {r_cnt} | {x_cnt} | {note} |\n"

    # Top wrong examples
    wrong_examples = [e for e in report.examples if not e["is_correct"]][:5]
    ex_rows = ""
    for e in wrong_examples:
        ex_rows += f"| `{e['qid']}` | `{e['agent_type']}` | {e['predicted_answer'][:30]} | {e['gold_answer'][:30]} | `{e['failure_mode']}` |\n"
    if not ex_rows:
        ex_rows = "| — | — | — | — | — |\n"

    md = f"""# Lab 16 — Reflexion Agent: Benchmark Report

> **Model:** `{report.meta.get('mode', 'mock')}`  |  **Dataset:** `{report.meta['dataset']}`  |  **Records:** {report.meta['num_records']}  |  **Date:** {__import__('datetime').date.today()}

---

## 1. Kết quả tổng quan

| Chỉ số | ReAct | Reflexion | Delta |
|:---|:---:|:---:|:---:|
| **Exact Match (EM)** | **{r_em*100:.1f}%** | **{x_em*100:.1f}%** | {arrow(delta.get('em_abs', 0)*100, '%', True)} |
| Câu đúng / tổng | {r_corr}/{r_n} | {x_corr}/{x_n} | — |
| Số lần thử TB | {react.get('avg_attempts', 0):.2f} | {reflexion.get('avg_attempts', 0):.2f} | {arrow(delta.get('attempts_abs', 0), '', False)} |
| Token TB / câu | {react.get('avg_token_estimate', 0):.0f} | {reflexion.get('avg_token_estimate', 0):.0f} | {arrow(delta.get('tokens_abs', 0), ' tok', False)} |
| Latency TB (ms) | {react.get('avg_latency_ms', 0):.0f} | {reflexion.get('avg_latency_ms', 0):.0f} | {arrow(delta.get('latency_abs', 0), ' ms', False)} |

### Accuracy bar

```
ReAct     {bar(r_em)}  {r_em*100:.1f}%
Reflexion {bar(x_em)}  {x_em*100:.1f}%
```

---

## 2. Phân tích Failure Modes

| Failure mode | ReAct | Reflexion | Giải thích |
|:---|:---:|:---:|:---|
{fm_rows}
---

## 3. Ví dụ sai (top {len(wrong_examples)})

| QID | Agent | Predicted | Gold | Failure mode |
|:---|:---|:---|:---|:---|
{ex_rows}
---

## 4. Extensions đã implement

{ext_lines}

---

## 5. Discussion

{report.discussion}

---

## 6. Kết luận

Reflexion cải thiện EM từ **{r_em*100:.1f}%** lên **{x_em*100:.1f}%** (+{delta.get('em_abs', 0)*100:.1f}%), với chi phí tăng thêm {delta.get('tokens_abs', 0):.0f} token/câu và {delta.get('latency_abs', 0):.0f}ms latency. Kiến trúc phản ánh (reflection) hiệu quả nhất với các câu hỏi multi-hop bị dừng sớm hoặc nhầm entity.
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
