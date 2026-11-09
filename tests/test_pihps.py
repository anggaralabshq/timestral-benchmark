import json
from datetime import date
from pathlib import Path

import pytest

from tsbench import pihps
from tsbench.pihps import COMMODITIES, download, fetch_month, parse_grid

FIX = Path(__file__).parent / "fixtures" / "pihps_sample.json"


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        if isinstance(self.payload, str):
            raise ValueError("not json")
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return FakeResp(self.payload)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(pihps.time, "sleep", lambda s: None)


def test_parse_grid_values():
    df = parse_grid(json.load(open(FIX)), "cabai_merah")
    assert set(df.columns) == {"date", "region", "level", "commodity", "price"}
    nat = df[df.region == "Semua Provinsi"]
    assert (nat.level == 0).all() and len(nat) == 5
    assert df.price.dtype == float
    assert df.price.isna().sum() == 1  # sel "-"
    assert df.price.dropna().min() > 1000  # "67,900" -> 67900.0, bukan 67.9
    assert nat.set_index("date").price[str(date(2026, 11, 2))] == 84400.0


def test_download_skips_cached_but_refetches_current_month(tmp_path):
    today = date.today()
    prev = date(today.year - (today.month == 1), (today.month - 2) % 12 + 1, 1)
    for cid in COMMODITIES.values():
        (tmp_path / f"{cid}_{prev:%Y-%m}.json").write_text('{"data": []}')
    s = FakeSession({"data": []})
    download(tmp_path, prev, today, session=s)
    months = {p["start_date"][:7] for p in s.calls}
    assert months == {f"{today:%Y-%m}"}
    assert len(s.calls) == len(COMMODITIES)


def test_fetch_month_retries_then_raises(tmp_path):
    s = FakeSession("<html>maintenance</html>")
    with pytest.raises(RuntimeError):
        fetch_month("com_14", 2026, 10, s)
    assert len(s.calls) == 3
    with pytest.raises(RuntimeError):
        download(tmp_path, date(2026, 10, 1), date(2026, 10, 31), session=s)
    assert list(tmp_path.iterdir()) == []


def test_url_matches_spec():
    # Regression: the pangan→tsbench rename rewrote the BI path segment "hargapangan"
    assert pihps.URL == "https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataKomoditas"


def test_download_refetches_month_cached_before_it_ended(tmp_path):
    # A file written mid-month holds a partial month; it must be completed on a later run
    import os
    from datetime import datetime
    partial = {cid: tmp_path / f"{cid}_2026-10.json" for cid in COMMODITIES.values()}
    for f in partial.values():
        f.write_text('{"data": []}')
        mid = datetime(2026, 10, 15).timestamp()
        os.utime(f, (mid, mid))
    s = FakeSession({"data": []})
    download(tmp_path, date(2026, 10, 1), date(2026, 10, 31), session=s)
    assert len(s.calls) == len(COMMODITIES)
    assert not list(tmp_path.glob("*.tmp"))
