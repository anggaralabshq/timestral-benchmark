import pandas as pd


def to_daily(df: pd.DataFrame, commodity: str, region: str = "Semua Provinsi") -> pd.Series:
    s = df[(df.commodity == commodity) & (df.region == region)].set_index("date").price.sort_index()
    return s.asfreq("D").rename(commodity)
