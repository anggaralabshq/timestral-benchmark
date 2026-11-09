# Timestral Pangan Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark jujur Timestral vs Chronos-2, TimesFM 2.5, model statistik, dan Prophet pada harga pangan harian PIHPS, menghasilkan status klaim K1–K4, grafik, README, dan draft post.

**Architecture:** Paket Python kecil `pangan/` dengan CLI `python -m pangan <cmd>`. Pipeline: download PIHPS → seri harian + covariates kalender → backtest rolling-origin per model (satu proses per model, hemat RAM 8 GB) → forecast parquet → metrik → status klaim → grafik. Semua jalan di CPU M1.

**Tech Stack:** Python 3.12 (uv), pandas, numpy, requests, pyarrow, statsforecast, prophet, chronos-forecasting>=2.0, timestral (git), timesfm 2.5 (git, dengan dependensi XReg), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-timestral-pangan-design.md`

## Global Constraints

- Semua jalan di laptop: Apple M1, RAM 8 GB, CPU saja. Satu foundation model dimuat per proses.
- Python 3.12 via `uv` (bukan Python sistem 3.14).
- `data/` gitignored; repo tidak pernah memuat data PIHPS. Atribusi: "Sumber: PIHPS Nasional, Bank Indonesia".
- Jeda ≥1 detik antar request PIHPS.
- Komoditas: cabai merah keriting, cabai rawit merah, bawang merah, beras kualitas medium. Level nasional ("Semua Provinsi").
- Jendela histori: 1.024 hari kalender terakhir sebelum origin, sama untuk semua model.
- Origin: setiap Senin dari 2025-01-06 s/d (tanggal data terakhir − 28 hari). Horizon forecast 28 hari kalender.
- Kuantil bersama: 0.1, 0.2, …, 0.9. Kuantil 0.05 dan 0.95 opsional per model (NaN bila tidak ada).
- Jendela Lebaran: origin dalam [L − 45 hari, L] untuk L ∈ {2025-03-31, 2026-03-20}.
- TimesFM yang dipakai 2.5 (`google/timesfm-2.5-200m-pytorch`), bukan 3.0.
- Model dengan covariates diuji dengan dan tanpa covariates: Timestral (5M, 209M), Chronos-2, TimesFM 2.5 (XReg), AutoARIMA (ARIMAX), Prophet.
- Kriteria klaim K1–K4 persis seperti spec §8; tidak diubah setelah hasil keluar.

## Review Focus

1. Sel PIHPS `"-"`, tanggal hilang, atau angka berformat `"14,900"` → NaN / 14900.0, tidak crash. (Task 1)
2. Respons PIHPS bukan JSON / HTTP error → retry dengan backoff, lalu error jelas; respons rusak tidak pernah di-cache; bulan berjalan selalu di-fetch ulang. (Task 1)
3. Libur panjang Lebaran (hari tanpa observasi) → target NaN dilewati metrik, bukan dihitung sebagai 0 atau ffill. (Task 3)
4. Skala MASE nol (harga datar di jendela histori, mungkin pada beras) → MASE NaN + tercatat, bukan inf. (Task 3)
5. Kuantil model bersilang (q0.1 > q0.2) → kuantil diurutkan per baris sebelum metrik. (Task 3)

---

### Task 1: Setup proyek + scraper PIHPS

**Files:**
- Create: `pyproject.toml`, `pangan/__init__.py`, `pangan/__main__.py`, `pangan/pihps.py`
- Test: `tests/test_pihps.py`, `tests/fixtures/pihps_sample.json`

**Interfaces:**
- Produces:
  - `COMMODITIES: dict[str, str]` (key pendek → `comcat_id`): `cabai_merah`, `cabai_rawit`, `bawang_merah`, `beras`
  - `parse_grid(payload: dict, commodity: str) -> pd.DataFrame` kolom `date (datetime64), region (str), level (int), commodity (str), price (float)`
  - `fetch_month(comcat_id: str, year: int, month: int, session) -> dict`
  - `download(raw_dir: Path, start: date, end: date, session=None) -> None` — cache `raw_dir/{comcat_id}_{YYYY-MM}.json`
  - `load(raw_dir: Path) -> pd.DataFrame` (gabungan `parse_grid` semua file cache)
  - CLI: `python -m pangan download`

- [ ] **Step 1: Setup env.** `uv init --package` tidak dipakai; tulis `pyproject.toml` manual (`requires-python = ">=3.12,<3.13"`, paket `pangan`, dependensi Tech Stack, dev: pytest). `uv sync`. Expected: `uv run python -c "import pandas, statsforecast"` sukses. Jika `timestral`/`timesfm`/`prophet` gagal di arm64, catat error dan tunda ke Task 4/5 (jangan blokir task ini).
- [ ] **Step 2: Cari mapping komoditas.** Satu request ke endpoint (spec §5) per `comcat_id` com_1..com_21 untuk rentang 1 minggu; catat nama tiap ID. Isi `COMMODITIES`. Jika bawang merah tidak ada, pakai komoditas volatil lain yang tersedia dan catat untuk README. Simpan satu respons nyata (dipangkas 2 baris × 5 tanggal, sisipkan satu `"-"`) sebagai `tests/fixtures/pihps_sample.json`.
- [ ] **Step 3: Tulis test gagal**

```python
def test_parse_grid_values():
    df = parse_grid(json.load(open(FIX)), "cabai_merah")
    assert set(df.columns) == {"date", "region", "level", "commodity", "price"}
    nat = df[df.region == "Semua Provinsi"]
    assert (nat.level == 0).all()
    assert df.price.dtype == float
    assert df.price.isna().sum() == 1          # sel "-"
    assert df.price.dropna().min() > 1000      # "14,900" -> 14900.0, bukan 14.9

def test_download_skips_cached_but_refetches_current_month(tmp_path):
    # FakeSession mencatat URL; file bulan lalu sudah ada di cache
    ...  # bulan lalu tidak di-request, bulan berjalan di-request

def test_fetch_month_retries_then_raises():
    # FakeSession: 3x respons HTML (bukan JSON) -> RuntimeError; tidak ada file cache tertulis
```

- [ ] **Step 4:** `uv run pytest tests/test_pihps.py -v` → FAIL (import error).
- [ ] **Step 5: Implement** `pangan/pihps.py`. Tanggal kolom `DD/MM/YYYY`; buang koma ribuan; `"-"` → NaN. `fetch_month`: 3 percobaan, backoff 2/4/8 detik, validasi `"data" in resp.json()`. `download`: loop komoditas × bulan, `time.sleep(1.0)` antar request, tulis file hanya setelah validasi. CLI `download` default start 2018-01-01, end hari ini, `raw_dir=data/raw`.
- [ ] **Step 6:** `uv run pytest tests/test_pihps.py -v` → PASS.
- [ ] **Step 7: Download data nyata.** `uv run python -m pangan download` (±430 request, ±10 menit). Expected: `load()` mengembalikan data nasional untuk 4 komoditas; catat tanggal paling awal yang tersedia.
- [ ] **Step 8: Commit** `feat: PIHPS scraper and parser`.

### Task 2: Seri harian + covariates kalender

**Files:**
- Create: `pangan/series.py`, `pangan/calendar.py`
- Test: `tests/test_series.py`, `tests/test_calendar.py`

**Interfaces:**
- Consumes: `load(raw_dir)` (Task 1)
- Produces:
  - `to_daily(df: pd.DataFrame, commodity: str, region: str = "Semua Provinsi") -> pd.Series` — index kalender harian kontinu, NaN untuk hari tanpa observasi, name = commodity
  - `EVENTS: dict[str, list[date]]` key: `ramadan_start`, `lebaran`, `idul_adha`, `natal`, `tahun_baru`, 2018–2027
  - `covariates(index: pd.DatetimeIndex) -> pd.DataFrame` kolom: `days_to_lebaran` (int 0..60, dipotong), `ramadan` (0/1), `win_lebaran`, `win_idul_adha`, `win_natal`, `win_tahun_baru` (0/1, H-7..H+7)

- [ ] **Step 1: Tulis test gagal**

```python
def test_to_daily_fills_calendar_with_nan():
    s = to_daily(df_with_friday_and_monday_only, "beras")
    assert s.index.freq == "D" and len(s) == 4      # Jum, Sab, Min, Sen
    assert s.isna().sum() == 2

def test_lebaran_2025_features():
    cov = covariates(pd.date_range("2025-03-01", "2025-04-10"))
    assert cov.loc["2025-03-31", "days_to_lebaran"] == 0
    assert cov.loc["2025-03-21", "days_to_lebaran"] == 10
    assert cov.loc["2025-03-24", "win_lebaran"] == 1 and cov.loc["2025-04-08", "win_lebaran"] == 0
    assert cov.loc["2025-03-15", "ramadan"] == 1 and cov.loc["2025-04-05", "ramadan"] == 0

def test_events_cover_range():
    for k, ds in EVENTS.items():
        assert {d.year for d in ds} >= set(range(2018, 2028)), k
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** `EVENTS` hard-coded; **verifikasi tiap tanggal ke daftar hari libur nasional/SKB resmi pemerintah** dan tulis URL sumber di komentar. Ramadan = [ramadan_start, lebaran − 1]. Lebaran 2025 harus 2025-03-31; Lebaran 2026 pakai tanggal resmi pemerintah (spec memakai 2026-03-20; jika resmi berbeda, pakai resmi dan perbarui jendela Lebaran di `backtest.py`).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `feat: daily series and holiday covariates`.

### Task 3: Metrik

**Files:**
- Create: `pangan/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)`, `EXTRA_Q = (0.05, 0.95)`
  - `sort_quantiles(q: np.ndarray) -> np.ndarray` (shape `(h, n_q)`, sort per baris)
  - `mase(y: np.ndarray, median: np.ndarray, history: np.ndarray) -> float` — skala = mean |diff| naive-1 pada history (abaikan NaN); skala 0 → NaN
  - `wql(y: np.ndarray, q: np.ndarray, levels=QUANTILES) -> float` — Σ pinball / Σ|y| (pendekatan CRPS)
  - `coverage(y, lo, hi) -> float`
  - Semua fungsi mengabaikan posisi dengan `y` NaN.

- [ ] **Step 1: Tulis test gagal**

```python
def test_mase_ignores_nan_targets():
    assert mase(np.array([10, np.nan, 12.]), np.array([11, 999, 11.]), np.array([8, 9, 10, 11.])) == 1.0

def test_mase_zero_scale_is_nan():
    assert np.isnan(mase(np.array([5.]), np.array([5.]), np.array([5, 5, 5.])))

def test_wql_perfect_is_zero():
    y = np.array([1., 2.]); q = np.tile(y[:, None], (1, 9))
    assert wql(y, q) == 0

def test_wql_known_value():
    # y=10, semua kuantil=8: pinball_tau = tau*2; Σ_tau = 2*4.5 = 9; / (n_tau=9 * |y|=10) = 0.1
    assert wql(np.array([10.]), np.full((1, 9), 8.)) == pytest.approx(0.1)

def test_sort_quantiles_fixes_crossing():
    assert (np.diff(sort_quantiles(np.array([[3, 1, 2.]])), axis=1) >= 0).all()

def test_coverage_skips_nan():
    assert coverage(np.array([1, np.nan, 5.]), np.array([0, 0, 0.]), np.array([2, 2, 2.])) == 0.5
```

Definisi `wql`: `Σ_t Σ_tau pinball_tau(y_t, q_t) / (n_tau · Σ_t |y_t|)`, dengan `pinball_tau(y, q) = max(tau·(y−q), (tau−1)·(y−q))`.

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `feat: forecast metrics`.

### Task 4: Adapter model statistik + Prophet

**Files:**
- Create: `pangan/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `QUANTILES`, `EXTRA_Q` (Task 3)
- Produces (dipakai Task 5, 6, 8):
  - Signature semua adapter: `fn(history: pd.Series, horizon: int, cov: pd.DataFrame | None) -> pd.DataFrame`
    - `history`: seri harian (bisa ada NaN), panjang ≤ 1024
    - `cov`: covariates untuk `history.index ∪ horizon hari ke depan` (None = tanpa covariates)
    - Return: index = `horizon` tanggal setelah `history.index[-1]`; kolom float `0.1 … 0.9` wajib, `0.05`/`0.95` jika model punya
  - `MODELS: dict[str, tuple[Callable, bool]]` → nama → (fn, supports_cov). Nama: `snaive`, `ets`, `arima`, `prophet` (Task 5 menambah `timestral_5m`, `timestral_209m`, `chronos2`, `timesfm25`)
  - `fill_for_model(history: pd.Series) -> pd.Series` — forward-fill (lalu back-fill awal) untuk model yang tidak menerima NaN

- [ ] **Step 1: Tulis test gagal** (parametrize semua nama di `MODELS`, data sintetis sinus + tren 400 hari dengan 10% NaN; covariates dari `covariates()`)

```python
@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("use_cov", [False, True])
def test_adapter_contract(name, use_cov):
    fn, supports = MODELS[name]
    if use_cov and not supports: pytest.skip()
    out = fn(hist, 28, cov_full if use_cov else None)
    assert len(out) == 28 and out.index[0] == hist.index[-1] + pd.Timedelta(days=1)
    assert all(q in out.columns for q in QUANTILES)
    assert out[list(QUANTILES)].notna().all().all()
```

- [ ] **Step 2:** `uv run pytest tests/test_models.py -v` → FAIL.
- [ ] **Step 3: Implement** di `pangan/models.py`: `snaive` (statsforecast `SeasonalNaive(season_length=7)`), `ets` (`AutoETS`), `arima` (`AutoARIMA`; dengan cov → covariates sebagai exogenous `X_df`). Interval statsforecast `level=[80, 90]` + kuantil lain dari `level` yang sesuai (10/20/…/80 → kuantil simetris). Prophet: `uncertainty_samples=1000`, kuantil dari `predictive_samples`; dengan cov → setiap kolom covariates via `add_regressor`, plus `add_country_holidays("ID")`. Default lain dari dokumentasi.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `feat: statistical and prophet adapters`.

### Task 5: Adapter foundation model

**Files:**
- Modify: `pangan/models.py`
- Test: `tests/test_models.py` (parametrize otomatis mencakup model baru; tandai `@pytest.mark.slow` karena download weights)

**Interfaces:**
- Consumes: kontrak adapter (Task 4)
- Produces: entri `MODELS` baru `timestral_5m`, `timestral_209m`, `chronos2`, `timesfm25` (semua `supports_cov=True`)

- [ ] **Step 1:** Tambah nama baru ke `MODELS` sebagai stub `NotImplementedError` → `uv run pytest tests/test_models.py -v -m slow` FAIL.
- [ ] **Step 2: Implement.** Baca README/model card terbaru untuk call yang tepat; model dimuat lazy sekali per proses (`functools.cache`), device CPU.
  - Timestral: `TimestralForecaster.from_pretrained("mistralai/Timestral-209M" | "-5M")`; input NumPy dengan NaN (didukung, spec §5); covariates sebagai known-future covariates sesuai API README.
  - Chronos-2: `chronos-forecasting>=2.0`, `Chronos2Pipeline`, covariates future via API `predict_df`/setara; kuantil yang diminta = QUANTILES (+EXTRA_Q jika didukung).
  - TimesFM 2.5: dari `github.com/google-research/timesfm`, covariates via XReg (update Okt 2025). Jika XReg tidak bisa dipasang/jalan di M1, set `supports_cov=False` untuk `timesfm25` dan catat alasannya di `results/notes.md` (spec §6: jangan dihilangkan diam-diam).
  - Model yang tidak menerima NaN memakai `fill_for_model`.
- [ ] **Step 3:** `uv run pytest tests/test_models.py -v -m slow` → PASS (catat juga RAM puncak kasar; harus muat di 8 GB satu model per proses).
- [ ] **Step 4: Commit** `feat: foundation model adapters`.

### Task 6: Backtest + evaluasi

**Files:**
- Create: `pangan/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `load`, `to_daily`, `covariates`, `MODELS`, metrik
- Produces:
  - `origins(last_date: date) -> list[date]` (Senin, 2025-01-06 … last_date − 28)
  - `LEBARAN_WINDOWS: list[tuple[date, date]]`; `in_lebaran_window(origin: date) -> bool`
  - `run(model: str, use_cov: bool, series: dict[str, pd.Series], out_dir: Path) -> Path` → `out_dir/forecasts/{model}__{cov|nocov}.parquet` kolom: `commodity, origin, date, h, y, q0.05, q0.1 … q0.9, q0.95`
  - `evaluate(forecast_dir: Path, series: dict[str, pd.Series]) -> pd.DataFrame` → disimpan `results/metrics.csv`: per `model, cov, commodity, origin, horizon∈{14,28}`: `mase, wql, cov80, cov90, lebaran`
  - `summarize(metrics: pd.DataFrame) -> pd.DataFrame` → `results/summary.csv`: per `model, cov, slice∈{all,lebaran}, horizon`: `rel_mase, rel_wql` (geo-mean rasio terhadap `snaive__nocov` pada commodity×origin yang sama, NaN dibuang), `cov80, cov90` (rata-rata), `n`
  - CLI: `python -m pangan backtest --model NAME [--cov]` (satu model per proses), `python -m pangan evaluate`

- [ ] **Step 1: Tulis test gagal**

```python
def test_origins_are_mondays_and_leave_room():
    os_ = origins(date(2026, 11, 9))
    assert os_[0] == date(2025, 1, 6) and all(o.weekday() == 0 for o in os_)
    assert os_[-1] <= date(2026, 10, 12)

def test_lebaran_window():
    assert in_lebaran_window(date(2025, 3, 3)) and not in_lebaran_window(date(2025, 4, 7))

def test_history_window_has_no_future_leak():
    # run() dengan adapter palsu yang merekam history: history.index[-1] < origin, len <= 1024

def test_summarize_relative_to_snaive():
    # metrics sintetis: model mase = 0.5 × snaive di semua baris -> rel_mase == 0.5
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** History = 1024 hari sebelum origin (eksklusif). `h` = 1..28; metrik horizon 14 = rata-rata hari 1–14 (error dihitung dari hari 1..14 saja), horizon 28 = hari 1–28. Kuantil diurutkan (`sort_quantiles`) sebelum metrik. `lebaran` flag dari origin.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Jalankan backtest penuh** untuk semua model × {nocov, cov bila didukung}, satu proses per kombinasi:
  `for m in snaive ets arima prophet timestral_5m timestral_209m chronos2 timesfm25; do uv run python -m pangan backtest --model $m; done` dan ulangi dengan `--cov` untuk model ber-covariates. Lalu `uv run python -m pangan evaluate`. Expected: `results/summary.csv` berisi semua kombinasi, `n` > 0 di slice `lebaran`.
- [ ] **Step 6: Commit** kode + `results/*.csv` (hasil boleh di-commit; data mentah tidak) `feat: rolling-origin backtest and evaluation`.

### Task 7: Status klaim K1–K4 + benchmark kecepatan

**Files:**
- Create: `pangan/claims.py`, `pangan/speed.py`
- Test: `tests/test_claims.py`

**Interfaces:**
- Consumes: `results/summary.csv` (Task 6), `MODELS` (Task 4/5)
- Produces:
  - `speed(model: str, series: pd.Series, runs: int = 5) -> dict` → `{model, file_mb, load_s, per_series_s (median batch 1, tanpa load), batch_all_s (median, 4 seri sekaligus, tanpa load)}`; CLI `python -m pangan speed` → `results/speed.csv` untuk semua foundation model (dengan cov)
  - `judge(summary: pd.DataFrame, speed: pd.DataFrame) -> pd.DataFrame` → kolom `claim, status∈{terbukti,sebagian,tidak terbukti}, evidence (str berisi angka)`; CLI `python -m pangan claims` → `results/claims.csv` + `results/claims.md`

- [ ] **Step 1: Tulis test gagal** — satu test per cabang status dari spec §8 dengan `summary` sintetis:

```python
def test_k1_terbukti_when_rank1_both_scores(): ...
def test_k1_sebagian_when_rank1_one_score(): ...
def test_k1_sebagian_when_rank1_only_lebaran(): ...
def test_k1_uses_rival_best_config():   # rival nocov lebih baik dari cov -> nocov yang dibandingkan
def test_k2_terbukti_at_5pct_and_no_harm(): ...
def test_k2_tidak_when_no_gain_in_lebaran(): ...
def test_k3_requires_within_10pct_and_fastest(): ...
def test_k4_ranges_80_and_90(): ...
```

Konstanta dari spec: Timestral utama = `timestral_209m` + cov; skor utama = `slice=all, horizon=28`; K2 ambang 5% (slice lebaran) dan 2% (slice all); K3 10% dan "tercepat" = `per_series_s`; K4 [0.75,0.85] dan [0.85,0.95].

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** `judge` dan `speed`.
- [ ] **Step 4:** Run → PASS. Lalu `uv run python -m pangan speed && uv run python -m pangan claims`.
- [ ] **Step 5: Commit** `feat: claim judging and CPU speed benchmark`.

### Task 8: Grafik, README, draft post

**Files:**
- Create: `pangan/plot.py`, `README.md`, `post/draft.md`
- Modify: `pangan/__main__.py` (subcommand `all` = download → backtest semua → evaluate → speed → claims → plot)

**Interfaces:**
- Consumes: forecast parquet, `summary.csv`, `claims.csv`
- Produces: `results/chart_lebaran_2026.png` (1200×1200), `README.md`, `post/draft.md`

- [ ] **Step 1: Implement** `plot(commodity: str = "cabai_merah", out: Path) -> Path`: aktual Jan–Apr 2026, forecast `timestral_209m__cov` dari origin terakhir sebelum Lebaran 2026 dengan pita 0.1–0.9, plus rival dengan `rel_mase` terbaik (konfigurasi terbaiknya) dari origin yang sama; garis vertikal Lebaran; label Bahasa Indonesia; teks terbaca di layar ponsel. Komoditas grafik = komoditas cabai dengan lonjakan Lebaran terbesar (tentukan dari data, catat alasannya; jangan pilih berdasarkan siapa yang menang).
- [ ] **Step 2: Verifikasi** `uv run python -m pangan plot` → buka PNG, cek visual (label, legend, pita).
- [ ] **Step 3: README.md**: tujuan, satu perintah reproduksi (`uv sync && uv run python -m pangan all`), data & atribusi PIHPS, peserta + lisensi (Timestral Modified MIT, atribusi "2026 - Mistral AI"), aturan keadilan, kriteria klaim, tabel `claims.md`, risiko kebocoran data (spec §8), keterbatasan (endpoint tak terdokumentasi, catatan `results/notes.md`), lingkungan (M1 8 GB).
- [ ] **Step 4: `post/draft.md`** (Bahasa Indonesia): headline mengikuti hasil `claims.csv` (bukan sebaliknya), 3–4 temuan, keterbatasan + risiko kebocoran 1 kalimat, link repo, atribusi data.
- [ ] **Step 5: Verifikasi reproduksi** dari clone bersih: `git clone` ke folder temp, `uv sync`, `uv run pytest -m "not slow"` PASS.
- [ ] **Step 6: Commit** `docs: chart, README, and post draft` + push.
