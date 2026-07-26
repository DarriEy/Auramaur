"""Operator commands for registered, reproducible market experiments."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.table import Table

from auramaur.cli._base import console, main
from auramaur.db.database import Database
from auramaur.experiments.operations import bind_manifest, load_manifest
from auramaur.experiments.registry import RegisteredExperiment
from auramaur.experiments.runtimes import (
    InMemoryShadowSink,
    LinearExecutionModel,
    ReplayRuntime,
    ShadowRuntime,
)
from auramaur.experiments.specialized import (
    InMemoryPackageSink,
    SpecializedReplayRuntime,
    SpecializedShadowRuntime,
)
from auramaur.research.experiment_reporting import ExperimentReportBuilder
from auramaur.research.experiment_repository import ExperimentRepository
from auramaur.research.experiment_scoreboard import ExperimentScoreboard


@main.group("experiment")
def experiment_group() -> None:
    """Define, replay, shadow, and compare preregistered experiments."""


@experiment_group.command("define")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def define_experiment(manifest: Path) -> None:
    """Validate a manifest and print its immutable lineage ID."""
    loaded = load_manifest(manifest)
    bind_manifest(loaded)
    console.print(loaded.definition.lineage_id)


@experiment_group.command("register")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None)
def register_experiment(manifest: Path, db_path: str | None) -> None:
    """Persist the manifest's prospective registration before collecting evidence."""
    async def run() -> None:
        loaded = load_manifest(manifest)
        registered = bind_manifest(loaded)
        database = Database(db_path)
        await database.connect(ensure_schema=False)
        try:
            record = await ExperimentRepository(database).register(registered)
            console.print(record.lineage_id)
        finally:
            await database.close()
    asyncio.run(run())


@experiment_group.command("replay")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None)
@click.option("--record/--no-record", default=True)
def replay_experiment(manifest: Path, db_path: str | None, record: bool) -> None:
    """Run chronological replay and optionally persist non-graduating evidence."""
    async def run() -> None:
        loaded = load_manifest(manifest)
        registered = bind_manifest(loaded)
        repository = None
        database = None
        if record:
            database = Database(db_path)
            await database.connect(ensure_schema=False)
            repository = ExperimentRepository(database)
        try:
            if isinstance(registered, RegisteredExperiment):
                if loaded.execution_model is None:
                    raise click.ClickException("portable replay requires execution_model")
                model = LinearExecutionModel(**loaded.execution_model)
                report = await ReplayRuntime(model).run(
                    registered, loaded.snapshots, loaded.initial_portfolio
                )
                inserted = await repository.record_replay(registered, report) if repository else 0
                console.print(
                    f"{report.lineage_id} equity={report.final_equity:.2f} "
                    f"events={len(report.results)} recorded={inserted}"
                )
            else:
                report = await SpecializedReplayRuntime().run(
                    registered, loaded.snapshots, loaded.initial_portfolio
                )
                inserted = (
                    await repository.record_specialized_replay(registered, report)
                    if repository else 0
                )
                proposed = sum(item.proposal is not None for item in report.results)
                console.print(
                    f"{report.lineage_id} packages={proposed} "
                    f"events={len(report.results)} recorded={inserted}"
                )
        finally:
            if database is not None:
                await database.close()
    asyncio.run(run())


@experiment_group.command("shadow")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def shadow_experiment(manifest: Path) -> None:
    """Evaluate every manifest event without orders or database writes."""
    async def run() -> None:
        loaded = load_manifest(manifest)
        registered = bind_manifest(loaded)
        if isinstance(registered, RegisteredExperiment):
            sink = InMemoryShadowSink()
            runtime = ShadowRuntime(sink)
        else:
            sink = InMemoryPackageSink()
            runtime = SpecializedShadowRuntime(sink)
        for snapshot in loaded.snapshots:
            result = await runtime.run(registered, snapshot, loaded.initial_portfolio)
            console.print(result.model_dump_json())
    asyncio.run(run())


@experiment_group.command("report")
@click.argument("lineage_id")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None)
@click.option("--execution-model")
@click.option("--data-version")
def report_experiment(
    lineage_id: str,
    db_path: str | None,
    execution_model: str | None,
    data_version: str | None,
) -> None:
    """Print one explicitly selected portable replay cohort report."""
    async def run() -> None:
        database = Database(db_path)
        await database.connect(ensure_schema=False)
        try:
            report = await ExperimentReportBuilder(database).build(
                lineage_id,
                execution_model_version=execution_model,
                data_version=data_version,
            )
            console.print_json(report.model_dump_json())
        finally:
            await database.close()
    asyncio.run(run())


@experiment_group.command("scoreboard")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None)
def experiment_scoreboard(db_path: str | None) -> None:
    """Compare lineages without blending data or execution-model cohorts."""
    async def run() -> None:
        database = Database(db_path)
        await database.connect(ensure_schema=False)
        try:
            rows = await ExperimentScoreboard(database).build()
            table = Table(title="Experiment Scoreboard")
            for column in ("Strategy", "Lineage", "Model/Data", "Replay", "Prospective", "Net P&L", "DD", "Status"):
                table.add_column(column)
            for row in rows:
                table.add_row(
                    row.strategy_source,
                    row.lineage_id[:10],
                    f"{row.execution_model_version or '-'} / {row.data_version or '-'}",
                    f"{row.replay_observations} ({row.independent_markets} mkts)",
                    f"{row.prospective_warmup}/{row.prospective_holdout}",
                    "-" if row.net_pnl is None else f"{row.net_pnl:+.2f}",
                    "-" if row.max_drawdown is None else f"{row.max_drawdown:.1%}",
                    row.status,
                )
            console.print(table)
        finally:
            await database.close()
    asyncio.run(run())
