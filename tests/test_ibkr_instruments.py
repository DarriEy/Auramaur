from auramaur.exchange.ibkr_instruments import (
    BY_BOOK, BY_KEY, CATALOG, ContractKind, IBKRBook,
)


def test_catalog_has_unique_typed_keys_and_all_six_books():
    assert len(BY_KEY) == len(CATALOG)
    assert set(BY_BOOK) == set(IBKRBook)
    assert all(BY_BOOK[book] for book in IBKRBook)


def test_derivatives_and_bonds_declare_discovery_policy():
    for spec in BY_BOOK[IBKRBook.FUTURES]:
        assert spec.kind is ContractKind.FUTURE
        assert spec.expiry_policy == "front_liquid"
        assert spec.multiplier > 1
    grains = {spec.key: spec for spec in BY_BOOK[IBKRBook.FUTURES]
              if spec.key in {"ZC", "ZW"}}
    assert all(spec.contract_multiplier == 5000 and spec.multiplier == 50
               for spec in grains.values())
    for spec in BY_BOOK[IBKRBook.OPTIONS]:
        assert spec.kind is ContractKind.OPTION
        assert 0 < spec.option_dte_min <= spec.option_dte_max
        assert spec.option_right in {"C", "P"}
        assert spec.multiplier == 100
    for spec in BY_BOOK[IBKRBook.BONDS]:
        assert spec.kind is ContractKind.BOND
        assert spec.bond_query


def test_fx_uses_idealpro_and_native_equities_are_not_usd_only():
    assert all(spec.exchange == "IDEALPRO" for spec in BY_BOOK[IBKRBook.FX])
    currencies = {spec.currency for spec in BY_BOOK[IBKRBook.INTERNATIONAL_EQUITY]}
    assert {"CAD", "GBP", "EUR", "JPY", "HKD", "AUD"} <= currencies
