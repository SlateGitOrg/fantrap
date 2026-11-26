from pathlib import Path

import pytest

from fantrap.catalog import Catalog, load_catalog
from fantrap.guard import Guard
from fantrap.seed import build_warehouse

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return load_catalog(ROOT / "catalog.yaml")


@pytest.fixture(scope="session")
def guard(catalog: Catalog) -> Guard:
    return Guard(catalog)


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("warehouse") / "ar.duckdb"
    build_warehouse(7, path)
    return path


@pytest.fixture(scope="session")
def questions_path() -> Path:
    return ROOT / "evals" / "questions.yaml"


@pytest.fixture(scope="session")
def responses_path() -> Path:
    return ROOT / "evals" / "scripted_responses.yaml"
