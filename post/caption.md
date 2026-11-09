# Compact LinkedIn caption

Image: `results/chart_lebaran_2026.png`

---

Mistral is known for LLMs. Now it has a time-series forecasting model, Timestral. Is it actually good, or just a new name on the leaderboard?

I benchmarked it against Chronos-2 (Amazon), TimesFM 2.5 (Google) and the classics (ARIMA, ETS, Prophet). Zero-shot, CPU only, 4 claims with pass/fail rules locked before the first run.

✅ Accuracy: most accurate on average, error 7% below the best rival
✅ Size: the 5M version (21 MB) is only 6% behind the 209M, and the fastest foundation model tested at 93 ms per series
✅ Uncertainty: its 80% interval really covers 80%. Chronos-2 and TimesFM are overconfident (65–70%)
❌ Calendar input: adding known events made it worse in the period they should help most

Verdict: Mistral's first time-series model is legit. It beats Google's and Amazon's models here, and the tiny version is a solid default. But the covariate feature isn't there yet.

Code and every number, including the ones that don't favour Timestral:
https://github.com/anggaralabshq/timestral-benchmark

Tested on daily food prices (PIHPS, Bank Indonesia).

#TimeSeries #Forecasting #MachineLearning #Mistral
