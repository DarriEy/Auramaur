"""Typed universe for read-only IBKR multi-asset paper books.

Instrument identity is deliberately separate from strategy state.  IBKR cannot
reliably resolve a bare ticker across exchanges and security types, and futures,
options, and bonds require discovery before a quoteable contract exists.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class IBKRBook(str, Enum):
    GLOBAL_ETF = "global_etf"
    FX = "fx"
    FUTURES = "futures"
    INTERNATIONAL_EQUITY = "international_equity"
    OPTIONS = "options"
    BONDS = "bonds"


class ContractKind(str, Enum):
    STOCK = "STK"
    FOREX = "CASH"
    FUTURE = "FUT"
    OPTION = "OPT"
    BOND = "BOND"


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    key: str
    book: IBKRBook
    kind: ContractKind
    symbol: str
    exchange: str
    currency: str
    asset_class: str
    description: str
    primary_exchange: str = ""
    multiplier: float = 1.0
    contract_multiplier: float = 0.0
    paper_capital_per_unit_usd: float = 0.0
    calendar: str = "US_EQUITY"
    # Discovery policies are resolved by the read-only client at runtime.
    expiry_policy: str = ""
    option_right: str = ""
    option_dte_min: int = 0
    option_dte_max: int = 0
    bond_query: str = ""


def _stk(key: str, book: IBKRBook, symbol: str, exchange: str, currency: str,
         asset_class: str, description: str, *, primary: str = "",
         calendar: str = "US_EQUITY") -> InstrumentSpec:
    return InstrumentSpec(key, book, ContractKind.STOCK, symbol, exchange,
                          currency, asset_class, description,
                          primary_exchange=primary, calendar=calendar)


# Liquid US-listed instruments give the global book broad economic coverage
# without cross-currency settlement. Native listings live in their own book.
GLOBAL_ETFS = (
    _stk("SPY", IBKRBook.GLOBAL_ETF, "SPY", "SMART", "USD", "us_equity", "S&P 500", primary="ARCA"),
    _stk("QQQ", IBKRBook.GLOBAL_ETF, "QQQ", "SMART", "USD", "us_equity", "Nasdaq-100", primary="NASDAQ"),
    _stk("IWM", IBKRBook.GLOBAL_ETF, "IWM", "SMART", "USD", "us_equity", "Russell 2000", primary="ARCA"),
    _stk("VEA", IBKRBook.GLOBAL_ETF, "VEA", "SMART", "USD", "international", "Developed markets ex-US", primary="ARCA"),
    _stk("VWO", IBKRBook.GLOBAL_ETF, "VWO", "SMART", "USD", "international", "Emerging markets", primary="ARCA"),
    _stk("INDA", IBKRBook.GLOBAL_ETF, "INDA", "SMART", "USD", "international", "India equities", primary="CBOE"),
    _stk("MCHI", IBKRBook.GLOBAL_ETF, "MCHI", "SMART", "USD", "international", "China equities", primary="NASDAQ"),
    _stk("EWJ", IBKRBook.GLOBAL_ETF, "EWJ", "SMART", "USD", "international", "Japan equities", primary="ARCA"),
    _stk("EWZ", IBKRBook.GLOBAL_ETF, "EWZ", "SMART", "USD", "international", "Brazil equities", primary="ARCA"),
    _stk("TLT", IBKRBook.GLOBAL_ETF, "TLT", "SMART", "USD", "rates", "Long US Treasuries", primary="NASDAQ"),
    _stk("IEF", IBKRBook.GLOBAL_ETF, "IEF", "SMART", "USD", "rates", "Intermediate US Treasuries", primary="NASDAQ"),
    _stk("TIP", IBKRBook.GLOBAL_ETF, "TIP", "SMART", "USD", "rates", "US inflation-linked Treasuries", primary="ARCA"),
    _stk("LQD", IBKRBook.GLOBAL_ETF, "LQD", "SMART", "USD", "credit", "Investment-grade credit", primary="ARCA"),
    _stk("HYG", IBKRBook.GLOBAL_ETF, "HYG", "SMART", "USD", "credit", "High-yield credit", primary="ARCA"),
    _stk("EMB", IBKRBook.GLOBAL_ETF, "EMB", "SMART", "USD", "credit", "Emerging-market sovereign debt", primary="NASDAQ"),
    _stk("GLD", IBKRBook.GLOBAL_ETF, "GLD", "SMART", "USD", "commodity", "Gold", primary="ARCA"),
    _stk("SLV", IBKRBook.GLOBAL_ETF, "SLV", "SMART", "USD", "commodity", "Silver", primary="ARCA"),
    _stk("DBC", IBKRBook.GLOBAL_ETF, "DBC", "SMART", "USD", "commodity", "Broad commodities", primary="ARCA"),
    _stk("VNQ", IBKRBook.GLOBAL_ETF, "VNQ", "SMART", "USD", "real_estate", "US real estate", primary="ARCA"),
    _stk("UUP", IBKRBook.GLOBAL_ETF, "UUP", "SMART", "USD", "currency", "US dollar basket", primary="ARCA"),
)

FX = tuple(
    InstrumentSpec(pair, IBKRBook.FX, ContractKind.FOREX, pair[:3], "IDEALPRO",
                   pair[3:], "fx", f"{pair[:3]}/{pair[3:]}", multiplier=1000,
                   paper_capital_per_unit_usd=100, calendar="FX_24X5")
    for pair in ("EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF",
                 "NZDUSD", "EURGBP", "EURJPY", "GBPJPY")
)

FUTURES = tuple(
    InstrumentSpec(key, IBKRBook.FUTURES, ContractKind.FUTURE, symbol, exchange,
                   currency, asset, description, multiplier=multiplier,
                   paper_capital_per_unit_usd=capital, calendar="FUTURES",
                   expiry_policy="front_liquid")
    for key, symbol, exchange, currency, asset, description, multiplier, capital in (
        ("MES", "MES", "CME", "USD", "equity_index", "Micro E-mini S&P 500", 5, 2500),
        ("MNQ", "MNQ", "CME", "USD", "equity_index", "Micro E-mini Nasdaq-100", 2, 3500),
        ("M2K", "M2K", "CME", "USD", "equity_index", "Micro E-mini Russell 2000", 5, 1500),
        ("ZN", "ZN", "CBOT", "USD", "rates", "10-year US Treasury note", 1000, 3000),
        ("MCL", "MCL", "NYMEX", "USD", "energy", "Micro WTI crude oil", 100, 1500),
        ("MGC", "MGC", "COMEX", "USD", "metals", "Micro Gold", 10, 2000),
        ("SIC", "SIC", "COMEX", "USD", "metals", "100-ounce Silver", 100, 1000),
        ("ZC", "ZC", "CBOT", "USD", "agriculture", "Corn", 50, 2500),
        ("ZW", "ZW", "CBOT", "USD", "agriculture", "Wheat", 50, 3000),
    )
)

# IBKR qualifies grains with the physical 5,000-bushel contract multiplier and
# priceMagnifier=100. P&L per displayed price point is therefore $50.
FUTURES = tuple(
    replace(spec, contract_multiplier=5000.0)
    if spec.key in {"ZC", "ZW"} else spec
    for spec in FUTURES
)

INTERNATIONAL_EQUITIES = (
    _stk("RY.TO", IBKRBook.INTERNATIONAL_EQUITY, "RY", "SMART", "CAD", "canada", "Royal Bank of Canada", primary="TSE", calendar="TORONTO"),
    _stk("SHOP.TO", IBKRBook.INTERNATIONAL_EQUITY, "SHOP", "SMART", "CAD", "canada", "Shopify", primary="TSE", calendar="TORONTO"),
    _stk("SHEL.L", IBKRBook.INTERNATIONAL_EQUITY, "SHEL", "SMART", "GBP", "uk", "Shell", primary="LSE", calendar="LONDON"),
    _stk("AZN.L", IBKRBook.INTERNATIONAL_EQUITY, "AZN", "SMART", "GBP", "uk", "AstraZeneca", primary="LSE", calendar="LONDON"),
    _stk("SAP.DE", IBKRBook.INTERNATIONAL_EQUITY, "SAP", "SMART", "EUR", "europe", "SAP", primary="IBIS", calendar="XETRA"),
    _stk("ASML.AS", IBKRBook.INTERNATIONAL_EQUITY, "ASML", "SMART", "EUR", "europe", "ASML", primary="AEB", calendar="AMSTERDAM"),
    _stk("7203.T", IBKRBook.INTERNATIONAL_EQUITY, "7203", "TSEJ", "JPY", "japan", "Toyota Motor", calendar="TOKYO"),
    _stk("0700.HK", IBKRBook.INTERNATIONAL_EQUITY, "700", "SEHK", "HKD", "hong_kong", "Tencent", calendar="HONG_KONG"),
    _stk("BHP.AX", IBKRBook.INTERNATIONAL_EQUITY, "BHP", "ASX", "AUD", "australia", "BHP Group", calendar="SYDNEY"),
)

OPTIONS = tuple(
    InstrumentSpec(f"{symbol}_ATM_{right}", IBKRBook.OPTIONS,
                   ContractKind.OPTION, symbol, "SMART", "USD", "options",
                   f"{symbol} 30-60 DTE ATM {right}", multiplier=100,
                   paper_capital_per_unit_usd=0,
                   option_right=right, option_dte_min=30, option_dte_max=60)
    for symbol in ("SPY", "QQQ", "IWM", "TLT", "GLD")
    for right in ("C", "P")
)

# Bond queries are discovery descriptors, not assumed conIds. The client must
# resolve an exact issue and retain its conId before requesting quotes.
BONDS = tuple(
    InstrumentSpec(key, IBKRBook.BONDS, ContractKind.BOND, "", "SMART", "USD",
                   asset, description, multiplier=10,
                   paper_capital_per_unit_usd=1000, calendar="US_BOND",
                   bond_query=query)
    for key, asset, description, query in (
        ("UST_2Y", "government", "US Treasury near 2-year maturity", "UST:2Y"),
        ("UST_5Y", "government", "US Treasury near 5-year maturity", "UST:5Y"),
        ("UST_10Y", "government", "US Treasury near 10-year maturity", "UST:10Y"),
        ("UST_30Y", "government", "US Treasury near 30-year maturity", "UST:30Y"),
        ("CORP_5Y", "corporate", "US corporate bond near 5-year maturity", "CORP:5Y"),
    )
)

CATALOG = GLOBAL_ETFS + FX + FUTURES + INTERNATIONAL_EQUITIES + OPTIONS + BONDS
BY_KEY = {instrument.key: instrument for instrument in CATALOG}
BY_BOOK = {book: tuple(i for i in CATALOG if i.book is book) for book in IBKRBook}

if len(BY_KEY) != len(CATALOG):  # pragma: no cover - import-time invariant
    raise RuntimeError("duplicate IBKR instrument key")
