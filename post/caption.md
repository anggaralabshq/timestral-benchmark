# Compact LinkedIn caption

Image: `results/chart_lebaran_2026.png`

---

Mistral's Timestral vs Chronos-2, TimesFM 2.5, ARIMA, ETS and Prophet. I tested them on Indonesian daily food prices from PIHPS (chili, shallot, rice), zero-shot, on a laptop with no GPU.

4 claims, rules locked before the experiment ran:

✅ Most accurate: error 7% below the best rival (ETS)
✅ The 5M version (21 MB) is only 6% behind the 209M, at 93 ms per series on CPU
✅ Its 80% interval really covers 80% of actual prices
❌ Adding the Eid holiday calendar made error ~10% worse. No model caught the chili price spike before Eid

Takeaway: small foundation models are now a solid default baseline. But don't trust one average number; check the period that matters most for your business.

Code and every number, including the ones that don't favour Timestral:
https://github.com/anggaralabshq/timestral-benchmark

Data: PIHPS Nasional, Bank Indonesia

#TimeSeries #Forecasting #MachineLearning #DataScience
