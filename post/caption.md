# Compact LinkedIn caption (Bahasa Indonesia)

Image: `results/chart_lebaran_2026_id.png`

---

Timestral dari Mistral vs Chronos-2, TimesFM 2.5, ARIMA, ETS, Prophet. Saya uji di harga pangan harian PIHPS (cabai, bawang merah, beras), zero-shot, di laptop tanpa GPU.

4 klaim, aturan dikunci sebelum eksperimen:

✅ Paling akurat: error 7% di bawah pesaing terbaik (ETS)
✅ Versi 5M (21 MB) cuma 6% di belakang versi 209M, 93 ms per seri di CPU
✅ Interval 80% benar-benar mencakup 80% harga aktual
❌ Kalender Lebaran malah bikin error naik ~10%. Lonjakan cabai menjelang Lebaran tidak tertangkap model mana pun

Pelajaran: foundation model kecil sudah layak jadi baseline. Tapi jangan percaya satu angka rata-rata, cek periode yang paling penting buat bisnismu.

Kode dan semua angka, termasuk yang tidak menguntungkan Timestral:
https://github.com/anggaralabshq/timestral-benchmark

Data: PIHPS Nasional, Bank Indonesia

#TimeSeries #Forecasting #MachineLearning #DataScience
