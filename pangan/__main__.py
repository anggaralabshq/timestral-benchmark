import argparse
from datetime import date
from pathlib import Path

RAW = Path("data/raw")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m pangan")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="ambil data PIHPS ke data/raw")
    d.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    d.add_argument("--end", type=date.fromisoformat, default=date.today())
    a = p.parse_args()

    if a.cmd == "download":
        from pangan.pihps import download
        download(RAW, a.start, a.end)


if __name__ == "__main__":
    main()
