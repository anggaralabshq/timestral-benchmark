<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.png">
  <img src="docs/assets/banner-light.png" width="100%" alt="timestral-benchmark: does Timestral hold up on Indonesian food prices? Timestral compared with Chronos-2, TimesFM 2.5, AutoARIMA, Prophet, ETS and seasonal naive">
</picture>

### An honest, reproducible benchmark of Timestral against other forecasting approaches on Indonesian daily food prices

**[Quick start ↓](#quick-start)** &nbsp;·&nbsp; **[Results ↓](#results)** &nbsp;·&nbsp; **[Methodology ↓](#methodology)** &nbsp;·&nbsp; **[Design spec →](docs/superpowers/specs/2026-11-09-timestral-pangan-design.md)**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](pyproject.toml)
[![Runs on CPU](https://img.shields.io/badge/hardware-CPU%20only%2C%20no%20GPU-orange)](#methodology)
[![Data: PIHPS](https://img.shields.io/badge/data-PIHPS%20%C2%B7%20Bank%20Indonesia-lightgrey)](#data)

</div>

Timestral, Mistral's time-series foundation model released in October 2026, is reported to rank first on public forecasting leaderboards. It is also advertised as small enough to run on a CPU and as able to use calendar information that is known in advance. Leaderboards are built from public benchmark datasets, though, and teams forecasting Indonesian prices want to know something else: does the model still win on their own data, around Eid al-Fitr, when chili prices jump?

This repository answers that for four staple commodities, without taking sides:

- **Four claims, tested one by one.** Accuracy, the benefit of holiday covariates, small-model speed, and interval calibration. Each one ends as *proven*, *partly proven* or *not proven*.
- **Rules locked before the first run.** Pass and fail thresholds are written in the [design spec](docs/superpowers/specs/2026-11-09-timestral-pangan-design.md) and in `pangan/claims.py`, with tests, before any result existed.
- **Fair to the rivals.** Every model gets the same history. Every model that supports covariates runs with and without them, and each rival is scored in its best configuration.
- **Runs on a laptop.** All models run zero-shot on a MacBook M1 with 8 GB RAM, with no GPU and no cost.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.png">
  <img src="docs/assets/architecture-light.png" width="100%" alt="Pipeline: PIHPS daily prices and an Indonesian holiday calendar feed eight zero-shot models, each run with and without calendar covariates, into a rolling-origin backtest of 93 weekly origins by 4 commodities with 28-day forecasts, scored by MASE, CRPS, coverage and CPU latency, ending in four claims judged proven, partly proven or not proven">
</picture>

## Results

<!-- RESULTS -->

## Quick start

**1. Install**

```bash
git clone https://github.com/anggaralabshq/timestral-benchmark.git && cd timestral-benchmark
uv sync
```

**2. Run the whole benchmark**

```bash
uv run python -m pangan all
```

This downloads the PIHPS data, backtests every model, scores them, runs the CPU benchmark, judges the claims and draws the chart. Expect 1–2 hours on an M1 laptop, mostly ARIMA and Prophet refits.

**3. Read the verdict**

```bash
cat results/claims.md
open results/chart_lebaran_2026.png
```

## Install

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/), and about 3 GB of disk for model weights. Python 3.12 is installed by uv.

```bash
uv sync                          # all dependencies, CPU builds
uv run pytest -m "not slow"      # fast tests
uv run pytest -m slow            # downloads the four foundation models and checks each adapter
```

## Usage

Each step can run on its own:

| Command | What it does |
|---|---|
| `python -m pangan download` | Fetch PIHPS monthly files into `data/raw/` (cached; the current month is always refetched) |
| `python -m pangan backtest --model NAME [--cov] [--jobs N]` | Rolling-origin backtest for one model, written to `results/forecasts/` |
| `python -m pangan evaluate` | Per-origin metrics (`results/metrics.csv`) and the summary (`results/summary.csv`) |
| `python -m pangan speed` | CPU load time and latency of the foundation models (`results/speed.csv`) |
| `python -m pangan claims` | Claim verdicts (`results/claims.csv`, `results/claims.md`) |
| `python -m pangan plot` | The headline chart (`results/chart_lebaran_2026.png`) |

Model names: `snaive`, `ets`, `arima`, `prophet`, `timestral_5m`, `timestral_209m`, `chronos2`, `timesfm25`. Run each foundation model in its own process to stay within 8 GB of RAM; `all` does this for you.

## Data

- **Source: PIHPS Nasional, Bank Indonesia.** Traditional-market prices, national average, working days, 2018-01-01 to 2026-11-09.
- Commodities: curly red chili (cabai merah keriting), red bird's eye chili (cabai rawit merah), medium shallot (bawang merah), medium-quality rice I (beras medium I).
- **This repository contains no data.** `python -m pangan download` fetches it from PIHPS's public endpoint, which is undocumented and may change, with a one-second pause between requests. The data is copyright PIHPS / Bank Indonesia.

## Participants

| Model | Size | License | Covariates |
|---|---|---|---|
| Timestral-209M, Timestral-5M | 209M / 5M | Modified MIT ("2026 - Mistral AI"; companies above a revenue threshold need a commercial license) | yes, known-future |
| Chronos-2 | 120M | Apache 2.0 | yes |
| TimesFM 2.5 | 200M | Apache 2.0 (not 3.0, whose weights are non-commercial) | yes, via XReg |
| AutoARIMA / ARIMAX | – | statsforecast, Apache 2.0 | yes, exogenous |
| Prophet | – | MIT | yes, regressors + Indonesian holidays |
| AutoETS, seasonal naive | – | statsforecast, Apache 2.0 | no |

## Methodology

- **Rolling origin.** Every Monday from 2025-01-06 to 2026-10-12: 93 origins × 4 commodities. Each forecast covers 28 days. Results are reported at h=14 (mean over days 1–14) and h=28 (days 1–28).
- **Same history for every model.** The 1,024 calendar days before the origin. Days without a price are NaN; models that cannot take NaN get forward-filled input. Scoring uses only days with an actual price.
- **Calendar covariates, known in advance.** Days to the next Eid al-Fitr (capped at 60), a Ramadan flag, and ±7-day windows around Eid al-Fitr, Eid al-Adha, Christmas and New Year.
- **Metrics.** MASE (median against in-sample naive-1), CRPS (weighted quantile loss over the 0.1–0.9 quantiles), and 80% and 90% interval coverage. Headline scores are geometric means of the ratio to seasonal naive, so below 1 is better.
- **Eid window.** Origins from 45 days before Eid al-Fitr up to the day itself, for 2025 (31 March) and 2026 (20 March).
- **Fairness.** Rivals are compared in their best configuration, with or without covariates, chosen per score. Default settings from each model card, no tuning.

### Claim criteria

| Claim | Proven when |
|---|---|
| K1 Accuracy | Timestral-209M with covariates ranks first on both MASE and CRPS across all origins (h=28) |
| K2 Covariates | The calendar lowers Timestral's MASE by ≥5% in the Eid window and costs ≤2% elsewhere |
| K3 Small and fast | Timestral-5M is within 10% of the 209M's MASE and is the fastest foundation model per series on CPU |
| K4 Calibration | 80% intervals cover 75–85% of actual prices and 90% intervals cover 85–95% |

*Partly proven* and *not proven* are defined in [`pangan/claims.py`](pangan/claims.py) and the [spec](docs/superpowers/specs/2026-11-09-timestral-pangan-design.md).

## Limitations

- **Possible training-data overlap.** Timestral was pretrained on synthetic data, but its post-training used real data that is not named, with no stated cutoff (tech report, arXiv 2610.37585). Chronos-2 and TimesFM 2.5 were also trained on real corpora. None of the models can be ruled out as having seen PIHPS data.
- Islamic holiday dates come from Wikipedia because the Ministry of Religious Affairs site could not be fetched automatically. Eid al-Fitr 2026 may differ by one day from the official decision.
- National level and four commodities only. This is not a general claim about all time series.
- Speed is measured on one M1 laptop (median of five runs), one series at a time.

<!-- NOTES -->

## Development

```text
pangan/
  pihps.py      PIHPS scraper and parser
  series.py     daily series with NaN for missing days
  calendar.py   Indonesian holiday calendar and covariates
  models.py     one adapter per model, same contract
  metrics.py    MASE, CRPS (weighted quantile loss), coverage
  backtest.py   rolling-origin backtest, evaluation, summary
  claims.py     claim verdicts with the pre-registered rules
  speed.py      CPU benchmark
  plot.py       headline chart
docs/assets/remotion/   source for the README images (npm install && npm run render)
```

## License

The code license will be added before this repository goes public. Model weights and data keep their own licenses, listed above.
