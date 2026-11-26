import pytest
from pydantic import ValidationError

from fantrap.catalog import Catalog


def test_loads_tables_keys_and_cardinalities(catalog):
    assert set(catalog.tables) == {
        "regions",
        "customers",
        "invoices",
        "shipments",
        "payments",
        "credit_notes",
        "payroll",
    }
    labels = {rel.label() for rel in catalog.relationships}
    assert "invoices 1:N shipments" in labels
    assert "customers 1:N invoices" in labels
    assert catalog.tables["invoices"].primary_key == ["invoice_id"]


def test_measures_are_declared(catalog):
    assert catalog.measure_columns("invoices") == ["amount"]
    assert "freight_cost" in catalog.measure_columns("shipments")


def test_analyst_allow_list(catalog):
    assert "payroll" not in catalog.allowed_tables("analyst")
    assert ("customers", "tax_id") in catalog.denied_columns("analyst")


def test_hr_admin_reads_only_payroll_and_regions(catalog):
    assert catalog.allowed_tables("hr_admin") == ["regions", "payroll"] or set(
        catalog.allowed_tables("hr_admin")
    ) == {"payroll", "regions"}


def test_unknown_role_is_an_error(catalog):
    with pytest.raises(ValueError, match="unknown role"):
        catalog.role("intern")


def test_uniqueness_follows_primary_keys(catalog):
    assert catalog.is_unique("invoices", {"invoice_id"})
    assert not catalog.is_unique("shipments", {"invoice_id"})
    assert not catalog.is_unique("nope", {"x"})


def test_schema_prompt_hides_what_the_role_cannot_read(catalog):
    analyst = catalog.schema_prompt("analyst")
    assert "payroll" not in analyst and "tax_id" not in analyst
    assert "invoices 1:N shipments" in analyst
    hr = catalog.schema_prompt("hr_admin")
    assert "payroll(" in hr and "invoices(" not in hr


def test_bad_references_are_rejected():
    bad = {
        "tables": {"a": {"primary_key": ["id"], "columns": {"id": "INTEGER"}}},
        "relationships": [{"parent": "a.id", "child": "b.id"}],
        "roles": {"r": {"allow_tables": ["a"]}},
    }
    with pytest.raises(ValidationError):
        Catalog.model_validate(bad)
