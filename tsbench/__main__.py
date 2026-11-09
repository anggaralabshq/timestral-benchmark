import argparse
from datetime import date
from pathlib import Path

RAW = Path("data/raw")
RESULTS = Path("results")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m tsbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="ambil data PIHPS ke data/raw")
    d.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    d.add_argument("--end", type=date.fromisoformat, default=date(2026, 11, 9), help="default: published data cut-off")
    b = sub.add_parser("backtest", help="rolling-origin backtest untuk satu model")
    b.add_argument("--model", required=True)
    b.add_argument("--cov", action="store_true")
    b.add_argument("--jobs", type=int, default=1)
    sub.add_parser("evaluate", help="hitung metrik dan ringkasan dari results/forecasts")
    sub.add_parser("speed", help="benchmark CPU foundation model (K3)")
    sub.add_parser("claims", help="status klaim K1-K4 dari summary.csv + speed.csv")
    sub.add_parser("plot", help="grafik utama Lebaran 2026")
    sub.add_parser("sensitivity", help="Chronos-2 and Timestral-209M with forward-filled input (not used for verdicts)")
    sub.add_parser("all", help="download -> backtest semua -> evaluate -> speed -> claims -> plot")
    a = p.parse_args()
    if a.cmd == "all":
        import subprocess
        import sys
        from tsbench.models import FOUNDATION, MODELS
        steps = [["download"]]
        for name, (_, cov) in MODELS.items():
            jobs = [] if name in FOUNDATION else ["--jobs", "6"]
            steps += [["backtest", "--model", name, *jobs]] + ([["backtest", "--model", name, "--cov", *jobs]] if cov else [])
        steps += [["evaluate"], ["speed"], ["claims"], ["plot"], ["sensitivity"]]
        for st in steps:  # proses terpisah per langkah: satu foundation model per proses (RAM 8 GB)
            print(">>", " ".join(st), flush=True)
            subprocess.run([sys.executable, "-m", "tsbench", *st], check=True)
        return

    if a.cmd == "download":
        from tsbench.pihps import download
        download(RAW, a.start, a.end)
    elif a.cmd == "backtest":
        from tsbench.backtest import load_series, run
        print(run(a.model, a.cov, load_series(RAW), RESULTS, jobs=a.jobs))
    elif a.cmd == "evaluate":
        from tsbench.backtest import evaluate, load_series, summarize
        m = evaluate(RESULTS / "forecasts", load_series(RAW))
        m.to_csv(RESULTS / "metrics.csv", index=False)
        summarize(m).to_csv(RESULTS / "summary.csv", index=False)
        print(f"{len(m)} baris metrik -> {RESULTS}")
    elif a.cmd == "speed":
        import pandas as pd
        from tsbench.backtest import load_series
        from tsbench.speed import REPOS, speed
        series = load_series(RAW)
        rows = [speed(m, series) for m in REPOS]  # satu proses: model kecil, RAM cukup (±2 GB total)
        pd.DataFrame(rows).to_csv(RESULTS / "speed.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    elif a.cmd == "claims":
        import pandas as pd
        from tsbench.claims import judge
        j = judge(pd.read_csv(RESULTS / "summary.csv"), pd.read_csv(RESULTS / "speed.csv"))
        j.to_csv(RESULTS / "claims.csv", index=False)
        md = "| Claim | Verdict | Evidence |\n|---|---|---|\n" + "".join(f"| {r.claim} | {r.status} | {r.evidence} |\n" for r in j.itertuples())
        (RESULTS / "claims.md").write_text(md)
        print(md)
    elif a.cmd == "sensitivity":
        import shutil
        from tsbench.backtest import evaluate, load_series, run, summarize
        from tsbench.models import MODELS, fill_for_model
        out, series = RESULTS / "sensitivity", load_series(RAW)
        (out / "forecasts").mkdir(parents=True, exist_ok=True)
        shutil.copy(RESULTS / "forecasts" / "snaive__nocov.parquet", out / "forecasts")
        for name in ("chronos2", "timestral_209m"):
            fn = MODELS[name][0]
            run(f"{name}_ffill", False, series, out,
                models={f"{name}_ffill": (lambda h, H, c, fn=fn: fn(fill_for_model(h), H, c), False)})
        summarize(evaluate(out / "forecasts", series)).to_csv(out / "summary.csv", index=False)
        print(out / "summary.csv")
    elif a.cmd == "plot":
        import pandas as pd
        from tsbench.backtest import load_series
        from tsbench.plot import plot
        series, summary = load_series(RAW), pd.read_csv(RESULTS / "summary.csv")
        for lang, name in (("en", "chart_lebaran_2026.png"), ("id", "chart_lebaran_2026_id.png")):
            print(plot(series, summary, RESULTS / "forecasts", RESULTS / name, lang=lang))


if __name__ == "__main__":
    main()
