import argparse
from datetime import date
from pathlib import Path

RAW = Path("data/raw")
RESULTS = Path("results")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m pangan")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="ambil data PIHPS ke data/raw")
    d.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    d.add_argument("--end", type=date.fromisoformat, default=date.today())
    b = sub.add_parser("backtest", help="rolling-origin backtest untuk satu model")
    b.add_argument("--model", required=True)
    b.add_argument("--cov", action="store_true")
    b.add_argument("--jobs", type=int, default=1)
    sub.add_parser("evaluate", help="hitung metrik dan ringkasan dari results/forecasts")
    a = p.parse_args()

    if a.cmd == "download":
        from pangan.pihps import download
        download(RAW, a.start, a.end)
    elif a.cmd == "backtest":
        from pangan.backtest import load_series, run
        print(run(a.model, a.cov, load_series(RAW), RESULTS, jobs=a.jobs))
    elif a.cmd == "evaluate":
        from pangan.backtest import evaluate, load_series, summarize
        m = evaluate(RESULTS / "forecasts", load_series(RAW))
        m.to_csv(RESULTS / "metrics.csv", index=False)
        summarize(m).to_csv(RESULTS / "summary.csv", index=False)
        print(f"{len(m)} baris metrik -> {RESULTS}")


if __name__ == "__main__":
    main()
