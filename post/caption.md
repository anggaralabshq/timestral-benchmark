# Compact LinkedIn caption

Image: `results/chart_lebaran_2026.png`

---

Every year before Eid, chili prices in Indonesia jump. Could the newest time-series AI see it coming?

Mistral just released Timestral. I put it against Chronos-2, TimesFM 2.5, ARIMA, ETS and Prophet on daily PIHPS food prices (chili, shallot, rice). Zero-shot, on a laptop, no GPU. Rules locked before the first run.

✅ Most accurate overall: error 7% below the best rival (ETS)
✅ The 5M version (21 MB) is only 6% behind the 209M, at 93 ms per series on CPU
✅ Its 80% interval really covers 80% of actual prices
❌ The Eid spike: no model saw it coming, and adding the holiday calendar made Timestral ~10% worse

So the new model wins on average and misses the one week that matters most.

Takeaway: small foundation models are now a solid default baseline. Just don't trust one average number; test the period your business actually cares about.

Code and every number, including the ones that don't favour Timestral:
https://github.com/anggaralabshq/timestral-benchmark

Data: PIHPS Nasional, Bank Indonesia

#TimeSeries #Forecasting #MachineLearning #DataScience
