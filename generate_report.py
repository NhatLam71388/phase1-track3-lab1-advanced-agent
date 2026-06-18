"""
generate_report.py — Sinh báo cáo so sánh ReAct vs Reflexion

Cách dùng:
  # Xem báo cáo từ file có sẵn (nhanh nhất)
  python generate_report.py

  # Chạy benchmark với golden test set rồi sinh báo cáo ngay
  python generate_report.py --golden data/golden.json

  # Chỉ định file report khác
  python generate_report.py --report-path outputs/real_run/report.json
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(add_completion=False)
console = Console()

# ── Giá mỗi 1M token (input, output) tính bằng USD ──────────────────────────
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "mock":                          (0.0,   0.0),
    "mistral-small-latest":          (0.10,  0.30),
    "mistral-medium-latest":         (0.40,  1.20),
    "google/gemma-4-31b-it:free":    (0.0,   0.0),
    "openai/gpt-4o-mini":            (0.15,  0.60),
    "openai/gpt-4o":                 (2.50,  10.0),
    "anthropic/claude-haiku-4-5":    (0.80,  4.00),
    "anthropic/claude-sonnet-4-6":   (3.00,  15.0),
}
# Tỉ lệ ước tính input vs output token
INPUT_RATIO = 0.75


def _fmt_pct(v: float) -> str:
    return f"[bold green]{v*100:.1f}%[/bold green]" if v >= 0.9 else f"[yellow]{v*100:.1f}%[/yellow]"


def _fmt_delta(v: float, unit: str = "", higher_is_better: bool = True) -> str:
    if abs(v) < 1e-6:
        return f"[dim]±0{unit}[/dim]"
    arrow = "▲" if v > 0 else "▼"
    color = "green" if (v > 0) == higher_is_better else "red"
    return f"[{color}]{arrow}{abs(v):.2f}{unit}[/{color}]"


def print_comparison_table(summary: dict) -> None:
    react = summary.get("react", {})
    refx  = summary.get("reflexion", {})
    delta = summary.get("delta_reflexion_minus_react", {})

    table = Table(
        title="So sánh ReAct vs Reflexion Agent",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("Chỉ số",              style="bold", width=24)
    table.add_column("ReAct",               justify="right", width=14)
    table.add_column("Reflexion",           justify="right", width=14)
    table.add_column("Delta (Reflex−React)", justify="right", width=22)

    table.add_row(
        "Số ví dụ (count)",
        str(react.get("count", "—")),
        str(refx.get("count", "—")),
        "—",
    )
    table.add_row(
        "Exact Match (EM %)",
        _fmt_pct(react.get("em", 0)),
        _fmt_pct(refx.get("em", 0)),
        _fmt_delta(delta.get("em_abs", 0) * 100, "%", higher_is_better=True),
    )
    table.add_row(
        "Số lần thử TB",
        f"{react.get('avg_attempts', 0):.2f}",
        f"{refx.get('avg_attempts', 0):.2f}",
        _fmt_delta(delta.get("attempts_abs", 0), "", higher_is_better=False),
    )
    table.add_row(
        "Token ước tính TB",
        f"{react.get('avg_token_estimate', 0):.0f}",
        f"{refx.get('avg_token_estimate', 0):.0f}",
        _fmt_delta(delta.get("tokens_abs", 0), " tok", higher_is_better=False),
    )
    table.add_row(
        "Latency TB (ms)",
        f"{react.get('avg_latency_ms', 0):.0f}",
        f"{refx.get('avg_latency_ms', 0):.0f}",
        _fmt_delta(delta.get("latency_abs", 0), " ms", higher_is_better=False),
    )
    console.print(table)


def print_cost_table(meta: dict, summary: dict) -> None:
    mode  = meta.get("mode", "mock")
    n_rec = meta.get("num_records", 0)
    react = summary.get("react", {})
    refx  = summary.get("reflexion", {})

    # Số ví dụ mỗi agent
    n = react.get("count", n_rec // 2)

    # Ước tính tổng token
    react_tokens  = int(react.get("avg_token_estimate", 385) * n)
    refx_tokens   = int(refx.get("avg_token_estimate", 538) * n)
    total_tokens  = react_tokens + refx_tokens

    # Ước tính thời gian chạy (dựa trên latency ms)
    react_ms  = react.get("avg_latency_ms", 200) * n
    refx_ms   = refx.get("avg_latency_ms", 310) * n
    total_sec = (react_ms + refx_ms) / 1000

    # Xác định model đang dùng (ưu tiên khớp trực tiếp với mode trong report)
    if mode == "mock":
        model_used = "mock"
    else:
        model_used = mode if mode in MODEL_PRICING else os.getenv("MISTRAL_MODEL", os.getenv("OPENROUTER_MODEL", "mock"))

    table = Table(
        title="Ước tính Chi phí & Thời gian",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
        title_style="bold white",
    )
    table.add_column("Model",           style="bold", width=36)
    table.add_column("Tổng token",      justify="right", width=14)
    table.add_column("Running time",    justify="right", width=14)
    table.add_column("Chi phí (USD)",   justify="right", width=14)
    table.add_column("Ghi chú",         width=20)

    for model, (price_in, price_out) in MODEL_PRICING.items():
        in_tok  = int(total_tokens * INPUT_RATIO)
        out_tok = total_tokens - in_tok
        cost    = (in_tok / 1_000_000) * price_in + (out_tok / 1_000_000) * price_out

        # Ước tính time: free models chậm hơn do rate limit
        if "free" in model or model == "mock":
            est_sec = total_tokens / 500  # ~500 tok/s cho free/mock
            extra = " (rate-limited)" if "free" in model else ""
        else:
            est_sec = total_sec  # dùng latency thực từ report
            extra = ""

        mins = int(est_sec // 60)
        secs = int(est_sec % 60)
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

        cost_str = "[dim]$0.00[/dim]" if cost < 0.001 else f"[yellow]${cost:.4f}[/yellow]"
        is_current = "◀ atual" if model == model_used else ""
        row_style  = "bold" if model == model_used else ""

        table.add_row(
            f"[{row_style}]{model}[/{row_style}]" if row_style else model,
            f"{total_tokens:,}",
            time_str + extra,
            cost_str,
            is_current,
            style=row_style,
        )

    console.print(table)
    console.print(
        f"  [dim]Tổng token ước tính: {total_tokens:,}  |  "
        f"ReAct: {react_tokens:,}  |  Reflexion: {refx_tokens:,}[/dim]"
    )


def print_failure_table(failure_modes: dict) -> None:
    if not failure_modes:
        return

    table = Table(
        title="Phân tích Failure Modes",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold red",
        title_style="bold white",
    )
    table.add_column("Failure mode",  style="bold", width=28)
    table.add_column("ReAct",         justify="right", width=10)
    table.add_column("Reflexion",     justify="right", width=12)
    table.add_column("Nhận xét",      width=34)

    notes = {
        "none":                  "[green]Trả lời đúng ngay[/green]",
        "entity_drift":          "Nhầm entity ở hop 2",
        "incomplete_multi_hop":  "Dừng sau hop 1, chưa hoàn thành",
        "wrong_final_answer":    "Hop đúng nhưng kết luận sai",
        "looping":               "Lặp lại câu trả lời",
        "reflection_overfit":    "Reflection quá phụ thuộc hint",
    }

    for mode, counts in sorted(failure_modes.items()):
        r = counts.get("react", 0)
        x = counts.get("reflexion", 0)
        r_str = f"[red]{r}[/red]" if r > 0 and mode != "none" else str(r)
        x_str = f"[red]{x}[/red]" if x > 0 and mode != "none" else str(x)
        table.add_row(mode, r_str, x_str, notes.get(mode, ""))

    console.print(table)


def run_golden(golden_path: str, out_dir: str) -> Path:
    """Chạy benchmark trên golden test set và trả về path của report.json."""
    from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
    from src.reflexion_lab.reporting import build_report, save_report
    from src.reflexion_lab.utils import load_dataset, save_jsonl

    examples = load_dataset(golden_path)
    n = len(examples)
    console.print(f"\n[bold cyan]Golden Test Set:[/bold cyan] {n} ví dụ  |  "
                  f"model: [yellow]{os.getenv('OPENROUTER_MODEL', 'mock')}[/yellow]\n")

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=3)

    react_records = []
    t0 = time.time()
    for i, ex in enumerate(examples, 1):
        react_records.append(react.run(ex))
        console.print(f"  [cyan]ReAct[/cyan]    {i:>3}/{n}  {ex.qid}", end="\r", highlight=False)
    console.print()

    reflexion_records = []
    for i, ex in enumerate(examples, 1):
        reflexion_records.append(reflexion.run(ex))
        console.print(f"  [magenta]Reflexion[/magenta] {i:>3}/{n}  {ex.qid}", end="\r", highlight=False)
    console.print()

    elapsed = time.time() - t0
    console.print(f"[dim]  Xong trong {elapsed:.1f}s[/dim]\n")

    all_records = react_records + reflexion_records
    out = Path(out_dir)
    save_jsonl(out / "react_runs.jsonl", react_records)
    save_jsonl(out / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(golden_path).name, mode="mock")
    json_path, _ = save_report(report, out)
    return json_path


@app.command()
def main(
    report_path: str = typer.Option("outputs/sample_run/report.json", help="Path tới report.json"),
    golden: str = typer.Option("", help="Chạy ngay với golden test set (path JSON)"),
    out_dir: str = typer.Option("outputs/golden_run", help="Thư mục lưu kết quả golden run"),
) -> None:
    # Nếu có golden test set → chạy benchmark trước
    if golden:
        if not Path(golden).exists():
            console.print(f"[red]Không tìm thấy file:[/red] {golden}")
            raise typer.Exit(1)
        report_path = str(run_golden(golden, out_dir))

    path = Path(report_path)
    if not path.exists():
        console.print(f"[red]Không tìm thấy report:[/red] {path}")
        console.print("[dim]Gợi ý: chạy python run_benchmark.py trước[/dim]")
        raise typer.Exit(1)

    payload = json.loads(path.read_text(encoding="utf-8"))
    meta    = payload.get("meta", {})
    summary = payload.get("summary", {})

    console.print(Panel(
        f"[bold]Dataset:[/bold] {meta.get('dataset','?')}   "
        f"[bold]Mode:[/bold] {meta.get('mode','?')}   "
        f"[bold]Records:[/bold] {meta.get('num_records','?')}",
        title="Lab 16 — Reflexion Agent Report",
        style="bold blue",
    ))

    print_comparison_table(summary)
    console.print()
    print_cost_table(meta, summary)
    console.print()
    print_failure_table(payload.get("failure_modes", {}))
    console.print()

    # Discussion
    disc = payload.get("discussion", "")
    if disc:
        console.print(Panel(disc, title="Discussion", style="dim"))

    console.print(f"\n[dim]Source: {path.resolve()}[/dim]")


if __name__ == "__main__":
    app()
