"""Multi-market metadata resolution utility."""

from data.schemas.market_data import MarketMetadata


def get_market_metadata(symbol: str, provider_name: str = "yfinance") -> MarketMetadata:
    """Resolve exchange, country, currency, timezone, and session rules for a given symbol."""
    sym_upper = symbol.strip().upper()

    # India Markets (.NS = NSE, .BO = BSE)
    if sym_upper.endswith(".NS") or sym_upper.endswith(".BO"):
        exchange = "NSE" if sym_upper.endswith(".NS") else "BSE"
        return MarketMetadata(
            symbol=sym_upper,
            exchange=exchange,
            country="India",
            currency="INR",
            timezone="Asia/Kolkata",
            trading_session="09:15-15:30",
            data_source=provider_name,
        )

    # UAE Markets (.AE, .DU = DFM, .AD = ADX)
    if sym_upper.endswith(".AE") or sym_upper.endswith(".DU") or sym_upper.endswith(".AD"):
        exchange = "ADX" if sym_upper.endswith(".AD") else "DFM"
        return MarketMetadata(
            symbol=sym_upper,
            exchange=exchange,
            country="United Arab Emirates",
            currency="AED",
            timezone="Asia/Dubai",
            trading_session="10:00-15:00",
            data_source=provider_name,
        )

    # UK Markets (.L = LSE)
    if sym_upper.endswith(".L"):
        return MarketMetadata(
            symbol=sym_upper,
            exchange="LSE",
            country="United Kingdom",
            currency="GBP",
            timezone="Europe/London",
            trading_session="08:00-16:30",
            data_source=provider_name,
        )

    # Crypto Pairs (-USD, -EUR, -BTC)
    if "-USD" in sym_upper or "-EUR" in sym_upper or "-BTC" in sym_upper:
        curr = sym_upper.split("-")[-1]
        return MarketMetadata(
            symbol=sym_upper,
            exchange="CCCAGG",
            country="Global",
            currency=curr,
            timezone="UTC",
            trading_session="24/7",
            data_source=provider_name,
        )

    # Default: US Equity / Index (NASDAQ / NYSE / AMEX)
    return MarketMetadata(
        symbol=sym_upper,
        exchange="NASDAQ/NYSE",
        country="United States",
        currency="USD",
        timezone="America/New_York",
        trading_session="09:30-16:00",
        data_source=provider_name,
    )
