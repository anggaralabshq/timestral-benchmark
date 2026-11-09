"""Scraper PIHPS Nasional (Bank Indonesia). Endpoint tidak terdokumentasi; data tidak dipublikasikan ulang."""

import json
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataKomoditas"

# Dari GetRefCommodityAndCategory (dicek 2026-11-09)
COMMODITIES = {
    "cabai_merah": "com_14",  # Cabai Merah Keriting
    "cabai_rawit": "com_16",  # Cabai Rawit Merah
    "bawang_merah": "com_11",  # Bawang Merah Ukuran Sedang
    "beras": "com_3",  # Beras Kualitas Medium I
}
NAMES = {v: k for k, v in COMMODITIES.items()}


def parse_grid(payload: dict, commodity: str) -> pd.DataFrame:
    recs = []
    for row in payload["data"]:
        for k, v in row.items():
            if "/" not in k:
                continue
            price = float(v.replace(",", "")) if v and v.strip() != "-" else float("nan")
            recs.append((datetime.strptime(k, "%d/%m/%Y"), row["name"].strip(), int(row["level"]), commodity, price))
    return pd.DataFrame(recs, columns=["date", "region", "level", "commodity", "price"])


def fetch_month(comcat_id: str, year: int, month: int, session) -> dict:
    start = date(year, month, 1)
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).date()
    params = {
        "price_type_id": 1, "comcat_id": comcat_id, "province_id": "", "regency_id": "",
        "showKota": "false", "showPasar": "false", "tipe_laporan": 1,
        "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}",
    }
    err = None
    for attempt in range(3):
        try:
            r = session.get(URL, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and "data" in payload:
                return payload
            err = f"respons tanpa 'data': {str(payload)[:100]}"
        except (requests.RequestException, ValueError) as e:
            err = repr(e)
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"PIHPS gagal untuk {comcat_id} {start:%Y-%m}: {err}")


def download(raw_dir: Path, start: date, end: date, session=None) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0 (timestral-benchmark)"
    for month in pd.period_range(start, end, freq="M"):
        month_end = month.end_time.date()
        for cid in COMMODITIES.values():
            f = raw_dir / f"{cid}_{month}.json"
            # cache only counts if written after the month ended; otherwise it is partial (or the current month)
            if f.exists() and date.fromtimestamp(f.stat().st_mtime) > month_end:
                continue
            payload = fetch_month(cid, month.year, month.month, session)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False))
            tmp.replace(f)  # atomic: an interrupted run never leaves truncated JSON
            time.sleep(1.0)


def load(raw_dir: Path) -> pd.DataFrame:
    frames = [parse_grid(json.loads(f.read_text()), NAMES[f.stem.rsplit("_", 1)[0]])
              for f in sorted(Path(raw_dir).glob("com_*.json"))]
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "region", "commodity"], keep="last")
