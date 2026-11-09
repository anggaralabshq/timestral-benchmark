# Draft post LinkedIn (Bahasa Indonesia)

Gambar: `results/chart_lebaran_2026_id.png`

---

Mistral merilis Timestral, model forecasting time-series yang diklaim peringkat 1 di leaderboard publik. Saya penasaran: apakah klaimnya masih berlaku di data Indonesia?

Saya uji di harga pangan harian PIHPS (Bank Indonesia) 2018–2026: cabai merah, cabai rawit, bawang merah, beras. Lawannya Chronos-2, TimesFM 2.5, ARIMA, ETS, dan Prophet. Semua zero-shot, jalan di laptop M1 tanpa GPU, biaya Rp0.

Aturannya saya kunci sebelum eksperimen jalan: 4 klaim, masing-masing berakhir "terbukti", "sebagian", atau "tidak terbukti". Pesaing selalu dinilai dalam konfigurasi terbaiknya.

Hasilnya (372 forecast 28 hari per model):

✅ Akurasi: Timestral-209M paling akurat secara rata-rata. Error 22% di bawah baseline naive, 7% lebih baik dari pesaing terbaik (ETS).

❌ Kalender hari besar: justru tidak membantu saat paling dibutuhkan. Di sekitar Lebaran, memasukkan kalender membuat error Timestral naik ~10%. Di periode itu, error titik ETS klasik malah yang terendah (walau rentang ketidakpastian Timestral tetap paling baik). Grafik di bawah: semua model gagal menangkap lonjakan cabai rawit menjelang Lebaran 2026.

✅ Model kecil: versi 5M (21 MB) hanya 6% di belakang versi 209M dan tetap mengalahkan semua model non-Timestral. Cukup 93 ms per seri di CPU.

✅ Interval prediksi jujur: interval 80% benar-benar mencakup 80% harga aktual. Chronos-2 dan TimesFM terlalu percaya diri (65–70%).

Pelajaran buat saya:
1. Foundation model kecil sudah layak jadi baseline default untuk forecasting harga, bahkan di laptop.
2. Lonjakan musiman seperti Lebaran masih jadi titik lemah. Jangan percaya satu angka rata-rata; cek periode yang paling penting untuk bisnismu.
3. Uji di datamu sendiri. Saya sempat menemukan bug di adapter salah satu model yang membuat skornya 5× lebih buruk. Tanpa cek kewajaran hasil, kesimpulannya bisa salah total.

Catatan jujur: data training model-model ini tidak sepenuhnya dibuka, jadi kemungkinan pernah "melihat" data PIHPS tidak bisa dinolkan untuk model mana pun.

Kode, metodologi, dan semua angka (termasuk yang tidak menguntungkan Timestral): [link repo]

Sumber data: PIHPS Nasional, Bank Indonesia.

#TimeSeries #Forecasting #MachineLearning #DataScience #Indonesia
