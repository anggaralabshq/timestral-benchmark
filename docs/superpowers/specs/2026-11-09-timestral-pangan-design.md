# Desain: Benchmark Timestral pada Harga Pangan Indonesia

Tanggal: 2026-11-09 · Status: draft untuk review

## 1. Intent

Eksperimen ini menjadi bahan **satu post LinkedIn** (post #1 dari 2; post #2 = WeVisOne-VL, dirancang terpisah). Topik post: **keunggulan Timestral** (model forecasting time-series dari Mistral, rilis Okt 2026), diuji secara jujur terhadap pendekatan lain pada data harga pangan harian Indonesia.

Pembaca: **rekruter** (butuh headline + satu visual yang langsung dipahami) dan **engineer ML/data Indonesia** (akan mengecek metodologi; perbandingan yang terlihat cherry-pick akan diserang di komentar).

Sikap: **netral, tidak memihak stack manapun.** Klaim keunggulan Timestral diperlakukan sebagai hipotesis. Hasil apa pun dilaporkan, termasuk jika Timestral kalah.

## 2. Goals

1. Menguji 4 klaim Timestral; masing-masing berakhir dengan status **terbukti / sebagian / tidak terbukti**:
   - **K1 Akurasi**: Timestral lebih akurat dari foundation model lain dan model statistik.
   - **K2 Known-future covariates**: kalender hari besar (Ramadan/Lebaran dll.) menurunkan error, terutama di sekitar lonjakan harga.
   - **K3 Kecil & ringan**: varian 5M (21 MB) tetap kompetitif; inferensi cepat di CPU laptop.
   - **K4 Interval terkalibrasi**: interval prediksi 80%/90% benar-benar mencakup ~80%/90% harga aktual.
2. Menghasilkan aset post: 1 grafik utama, 1 tabel klaim, angka headline.
3. Repo yang bisa direproduksi engineer lain di laptop tanpa GPU.

## 3. Non-goals

- Bukan sistem production, bukan dashboard, bukan API.
- Tidak melatih / fine-tune model apa pun (semua foundation model dipakai zero-shot; model statistik di-fit per origin seperti biasa).
- Tidak mencari model terbaik secara absolut; hanya membandingkan peserta di §6.
- Level provinsi bukan target utama (lihat §5, stretch).

## 4. Anti-goals (gagal walau "jalan")

- Peserta lain diberi konfigurasi lebih buruk dari Timestral (context lebih pendek, tanpa covariates padahal mendukung, dll.).
- Hanya melaporkan komoditas/periode yang menguntungkan Timestral.
- Data mentah PIHPS dipublikasikan ulang di repo.
- Biaya > $0 (semua jalan di laptop).

## 5. Data

**Sumber:** PIHPS Nasional (Bank Indonesia), endpoint JSON tak terdokumentasi, tanpa login:

```
GET https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataKomoditas
  ?price_type_id=1            # pasar tradisional
  &comcat_id=com_N            # satu komoditas per call
  &province_id=               # kosong = nasional + semua provinsi dalam satu respons
  &regency_id=&showKota=false&showPasar=false
  &tipe_laporan=1             # harian
  &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
```

Respons: `{"data":[...]}`; baris `level:0` "Semua Provinsi" = nasional, `level:1` = 34 provinsi. Kolom tanggal berformat `DD/MM/YYYY` (hanya hari kerja), nilai string `"14,900"`, `"-"` = kosong. Sudah diverifikasi 2026-11-09 untuk satu komoditas.

**Komoditas (keputusan Claude):** cabai merah (keriting), cabai rawit merah, bawang merah, beras (kualitas medium). Tiga pertama volatil, beras sebagai kontras stabil. Mapping `comcat_id` → nama harus diverifikasi saat scraping (ada laporan bahwa sebagian ID tidak tersedia; jika bawang merah tidak tersedia, ganti dengan komoditas volatil lain dan catat di README).

**Rentang:** 2018-01-01 s/d 2026-11-09 (atau tanggal paling awal yang tersedia; dicatat).

**Level:** nasional = analisis utama. Provinsi ikut ter-scrape (gratis, satu respons) tapi dipakai hanya jika waktu cukup (stretch).

**Scraper:** chunk per bulan, jeda ≥1 detik antar request, retry dengan backoff, cache respons mentah ke `data/raw/` (gitignored). Perkiraan ~430 request untuk 4 komoditas.

**Lisensi:** footer PIHPS "Semua Hak Dilindungi". Repo hanya memuat **script pengambil data**, bukan data. Atribusi "Sumber: PIHPS Nasional, Bank Indonesia" di README dan post.

**Pra-proses (keputusan Claude):** index kalender harian; hari tanpa observasi (akhir pekan, libur, cuti bersama Lebaran) = NaN. Model yang mendukung NaN menerima NaN; model lain menerima forward-fill. **Evaluasi hanya pada tanggal dengan observasi aktual.** Fallback jika endpoint mati: benchmark publik GIFT-Eval (cerita jadi kurang lokal, catat di post).

## 6. Peserta

| Kelompok | Model | Catatan |
|---|---|---|
| Subjek | Timestral-5M, Timestral-209M | `mistralai/Timestral-*`, Modified MIT (aman untuk penggunaan pribadi; atribusi "2026 - Mistral AI") |
| Foundation model | Chronos-2 (120M) | `amazon/chronos-2`, Apache 2.0, mendukung covariates |
| Foundation model | TimesFM 2.5 (200M) | `google/timesfm-2.5-200m-pytorch`, Apache 2.0, install dari GitHub, covariates via XReg (update Okt 2025). Jangan pakai TimesFM 3.0: weights non-komersial |
| Statistik | Seasonal naive, AutoETS, AutoARIMA (juga sebagai ARIMAX dengan covariates) | via `statsforecast` |
| Klasik populer | Prophet | dengan holidays bawaan Prophet untuk Indonesia + event §7 |

Varian Timestral-28M/123M opsional untuk kurva ukuran-vs-akurasi (K3) jika waktu cukup.

**Aturan keadilan:**
- Semua model mendapat **jendela histori sama**: 1.024 hari kalender terakhir sebelum origin (keputusan Claude; model statistik juga).
- Semua model yang mendukung covariates (Timestral, Chronos-2, TimesFM 2.5 via XReg, AutoARIMA sebagai ARIMAX, Prophet) diuji **dengan dan tanpa** covariates. Jika integrasi covariates satu model gagal secara teknis, catat di README; jangan diam-diam dihilangkan.
- Pengaturan default dari model card; tanpa tuning hyperparameter khusus untuk model mana pun.
- **Kuantil bersama: 0.1, 0.2, …, 0.9** (9 kuantil, yang bisa dikeluarkan semua peserta). CRPS dan coverage 80% (0.1–0.9) dihitung dari set ini untuk semua model. Coverage 90% (0.05–0.95) hanya dilaporkan untuk model yang mengeluarkannya (kemungkinan TimesFM 2.5 tidak; dicatat). Prophet: kuantil dari sampel posterior (`uncertainty_samples`), bukan dari satu `interval_width`.

## 7. Covariates (known-future)

Event: awal Ramadan, Idul Fitri (Lebaran), Idul Adha, Natal, Tahun Baru. Tanggal dari kalender resmi (hard-code tabel 2018–2027 di repo, sumber dicatat).

Fitur per hari (keputusan Claude): `days_to_lebaran` (dipotong ke 0..60), flag `ramadan`, flag `h-7..h+7` untuk tiap event. Builder boleh menyederhanakan selama setiap model yang mendukung covariates menerima fitur yang sama.

## 8. Protokol evaluasi

- **Rolling origin**: origin setiap Senin dari 2025-01-06 s/d (tanggal data terakhir − 28 hari). ±95 origin × 4 komoditas.
- **Horizon**: forecast 28 hari kalender; dilaporkan untuk h=14 dan h=28.
- **Slice**: (a) semua origin, (b) **jendela Lebaran** = origin 45 hari sebelum s/d hari Lebaran (2025-03-31, 2026-03-20), (c) per komoditas.
- **Metrik**:
  - Akurasi titik: **MASE** (median kuantil 0.5; skala = naive-1 in-sample pada jendela histori).
  - Probabilistik: **CRPS** (pendekatan weighted quantile loss atas 9 kuantil bersama §6), **coverage** interval 80% (semua model) dan 90% (jika tersedia).
  - Efisiensi (K3): ukuran file model, waktu load, waktu forecast per seri di CPU M1 (median 5 run, batch 1 dan batch semua seri). Lingkungan dicatat.
- **Horizon dalam angka**: "h=14" = rata-rata error hari 1–14; "h=28" = rata-rata hari 1–28 (bukan error di satu hari).
- **Agregasi**: skor utama = rata-rata geometrik (atas komoditas × origin) MASE dan CRPS **relatif terhadap seasonal naive**. Rank rata-rata dilaporkan sebagai pendukung, tidak dipakai untuk status klaim. Semua tabel lengkap di `results/`.

**Kriteria status klaim (ditetapkan sebelum eksperimen, tidak diubah setelah hasil keluar):**

Definisi bersama:
- *Timestral utama* = Timestral-209M **dengan** covariates.
- *Rival* = setiap peserta non-Timestral, masing-masing dalam **konfigurasi terbaiknya** (dengan atau tanpa covariates, mana yang skornya lebih baik). Ini menguntungkan rival, disengaja. Varian Timestral lain bukan rival.
- *Skor utama* = MASE relatif dan CRPS relatif di slice (a), horizon h=28.

- **K1 Akurasi.** Terbukti: Timestral utama peringkat 1 pada **kedua** skor utama. Sebagian: peringkat 1 pada tepat satu skor utama, **atau** peringkat 1 pada kedua skor hanya di slice (b). Selain itu: tidak terbukti. (Hasil per komoditas dan h=14 dilaporkan, tapi tidak menentukan status.)
- **K2 Covariates.** Diukur pada Timestral-209M (dengan vs tanpa). Terbukti: MASE relatif turun ≥5% di slice (b) **dan** tidak naik >2% di slice (a). Sebagian: turun tapi <5% di slice (b), atau slice (a) memburuk >2%. Tidak terbukti: tidak turun di slice (b). Kenaikan dari covariates pada Chronos-2/TimesFM/ARIMAX/Prophet dilaporkan sebagai konteks pembanding, bukan syarat status.
- **K3 Ringan.** "Tercepat" = median waktu forecast per seri, batch 1, CPU M1, **tidak termasuk waktu load**. Terbukti: Timestral-5M (dengan covariates) MASE relatifnya dalam 10% dari Timestral utama **dan** tercepat di antara foundation model. Sebagian: hanya salah satu. Waktu load dan ukuran file dilaporkan sebagai pendukung.
- **K4 Kalibrasi.** Diukur pada Timestral utama, slice (a), semua hari 1–28 digabung. Terbukti: coverage 80% di [75%, 85%] **dan** coverage 90% di [85%, 95%]. Sebagian: hanya salah satu. Tidak terbukti: keduanya di luar rentang.

**Risiko kebocoran data (diungkap, bukan dihindari):** Timestral menurut tech report-nya (arXiv 2610.37585) di-pretrain sepenuhnya dengan data sintetis, tapi post-training memakai data real yang tidak disebutkan, dan tidak ada tanggal cutoff. Chronos-2 dan TimesFM 2.5 juga dilatih dengan korpus real. Jadi kemungkinan model pernah melihat data PIHPS tidak bisa dinolkan untuk peserta mana pun. Keputusan: periode uji tetap 2025–2026; README dan post menyebut risiko ini secara eksplisit beserta fakta training data tiap model.

## 9. Output

1. `results/*.csv`: semua angka per model × komoditas × slice × horizon.
2. **Grafik utama**: harga cabai (komoditas terpilih) menjelang Lebaran 2026: aktual, Timestral dengan pita 80%, dan pesaing dengan MASE terbaik. PNG 1200×1200 (rasio LinkedIn), label Bahasa Indonesia, terbaca di ponsel.
3. **Tabel klaim**: K1–K4, status, satu angka pendukung masing-masing.
4. **README**: tujuan, cara reproduksi (satu perintah), metodologi, aturan keadilan, keterbatasan, atribusi data dan lisensi model.
5. **Draft teks post** (Bahasa Indonesia), ditulis setelah hasil keluar; headline mengikuti hasil, bukan sebaliknya.

## 10. Constraints

- Hardware: MacBook **Apple M1, RAM 8 GB**, 8 core. Satu foundation model dimuat dalam satu waktu. Tanpa GPU.
- Biaya: $0 (RunPod tidak dipakai untuk post ini; saldo $10 disimpan untuk post #2).
- Python: dikelola `uv` (sudah terpasang, 0.9.15) dengan Python 3.12 (Python sistem 3.14 terlalu baru untuk sebagian dependensi).
- Repo: GitHub **private** `anggaralabshq/timestral-benchmark` (sudah dibuat; awalnya `timestral-pangan`, di-rename 2026-11-09), dijadikan publik saat post terbit.

## 11. Keputusan

**Dari user:** dua post terpisah; Timestral duluan; netral dan jujur meski Timestral kalah; audiens rekruter + engineer ML/data Indonesia; data harga pangan Indonesia (atau data publik siap pakai jika sulit); fokus post = keunggulan Timestral; eksperimen tidak memakai biaya; repo private dulu.

**Keputusan Claude (bisa diubah user):** 4 komoditas di §5; level nasional; horizon 14/28; jendela histori 1.024 hari; skema NaN/forward-fill; fitur covariates §7; definisi rival & kriteria status klaim §8; kuantil bersama 0.1–0.9; TimesFM dan ARIMAX ikut diberi covariates; periode uji tetap + pengungkapan risiko kebocoran; repo memuat script, bukan data; identitas commit = noreply GitHub `anggaralabshq` (git config lokal, permintaan user).

## 12. Diserahkan ke builder

Struktur modul, nama file, library plotting, format CLI, detail retry scraper, cara memanggil API tiap model sesuai model card, desain visual grafik dalam batas §9.
