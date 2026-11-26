"""Command line interface: seed, check, ask, eval."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .catalog import DEFAULT_CATALOG_PATH, load_catalog
from .evaluate import render_markdown, run_eval, write_reports
from .execute import ExecutionError, run_query
from .guard import Guard
from .model import ModelError, OpenAICompatModel, ScriptedModel, SqlModel
from .pipeline import run_pipeline
from .seed import build_warehouse

app = typer.Typer(
    add_completion=False, help="Guarded text-to-SQL over an accounts-receivable DuckDB."
)
console = Console(soft_wrap=True)

DB_OPTION = typer.Option(Path("data/ar.duckdb"), "--db", help="DuckDB warehouse file.")
CATALOG_OPTION = typer.Option(DEFAULT_CATALOG_PATH, "--catalog", help="Semantic catalog YAML.")
QUESTIONS_OPTION = typer.Option(Path("evals/questions.yaml"), "--questions")
RESPONSES_OPTION = typer.Option(Path("evals/scripted_responses.yaml"), "--responses")


def _model(kind: str, responses: Path, questions: Path) -> SqlModel:
    if kind == "scripted":
        return ScriptedModel.from_files(responses, questions)
    if kind == "openai":
        return OpenAICompatModel.from_env()
    raise typer.BadParameter("model must be 'scripted' or 'openai'")


def _fmt(value: object) -> str:
    return f"{value:,.2f}" if isinstance(value, float) else str(value)


def _scalar(db: Path, sql: str) -> str:
    result = run_query(db, sql)
    return _fmt(result.rows[0][0]) if result.rows else "no rows"


@app.command()
def seed(
    seed: int = typer.Option(7, "--seed", help="Random seed."),
    out: Path = typer.Option(Path("data/ar.duckdb"), "--out", help="Output DuckDB file."),
) -> None:
    """Build the synthetic accounts-receivable warehouse."""
    counts = build_warehouse(seed, out)
    table = Table("table", "rows")
    for name, n in counts.items():
        table.add_row(name, f"{n:,}")
    console.print(f"Wrote {out} (seed {seed})")
    console.print(table)


@app.command()
def check(
    sql: str = typer.Argument(..., help="SQL to check."),
    role: str = typer.Option("analyst", "--role"),
    catalog: Path = CATALOG_OPTION,
    db: Path = DB_OPTION,
    gold: str = typer.Option(
        None, "--gold", help="Gold query; runs both on --db and prints their first values."
    ),
) -> None:
    """Print the guard's violations for SQL. Exit code 1 when there are any."""
    try:
        violations = Guard(load_catalog(catalog)).check(sql, role)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not violations:
        console.print("OK: no violations")
    for v in violations:
        console.print(f"[bold]{v.code}[/bold]  {v.message}", highlight=False)
    if gold:
        try:
            console.print(f"gold total:     {_scalar(db, gold)}")
            console.print(f"query returns:  {_scalar(db, sql)}")
        except ExecutionError as exc:
            console.print(f"could not run the comparison: {exc}")
    raise typer.Exit(1 if violations else 0)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question in plain English."),
    role: str = typer.Option("analyst", "--role"),
    model: str = typer.Option("scripted", "--model", help="scripted (offline) or openai."),
    catalog: Path = CATALOG_OPTION,
    db: Path = DB_OPTION,
    questions: Path = QUESTIONS_OPTION,
    responses: Path = RESPONSES_OPTION,
) -> None:
    """Run the full pipeline: generate, guard, rewrite, execute, repair."""
    try:
        run = run_pipeline(
            question, role, _model(model, responses, questions), load_catalog(catalog), db
        )
    except (ModelError, ValueError, ExecutionError) as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(2) from exc
    for attempt in run.attempts:
        codes = ", ".join(attempt.codes) or "ok"
        console.print(f"attempt {attempt.number} \\[{codes}]", highlight=False)
        console.print(attempt.sql, markup=False, highlight=False)
    if run.status != "ok" or run.result is None:
        console.print(run.message, markup=False)
        raise typer.Exit(1)
    table = Table(*run.result.columns)
    for row in run.result.rows[:50]:
        table.add_row(*(_fmt(v) for v in row))
    console.print(table)
    if run.result.truncated or len(run.result.rows) > 50:
        console.print(f"({len(run.result.rows)} rows fetched, first 50 shown)")


@app.command("eval")
def evaluate_cmd(
    catalog: Path = CATALOG_OPTION,
    db: Path = DB_OPTION,
    questions: Path = QUESTIONS_OPTION,
    responses: Path = RESPONSES_OPTION,
    model: str = typer.Option("scripted", "--model", help="scripted (offline) or openai."),
    out: Path = typer.Option(Path("reports"), "--out", help="Directory for eval.md and eval.json."),
) -> None:
    """Run the question suite in guarded and raw modes and write the reports."""
    from .evaluate import load_questions

    try:
        report = run_eval(
            load_questions(questions),
            lambda: _model(model, responses, questions),
            load_catalog(catalog),
            db,
        )
    except (ModelError, ExecutionError) as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(2) from exc
    md_path, json_path = write_reports(report, out)
    console.print(render_markdown(report), markup=False, highlight=False)
    console.print(f"Wrote {md_path} and {json_path}")


if __name__ == "__main__":
    app()
