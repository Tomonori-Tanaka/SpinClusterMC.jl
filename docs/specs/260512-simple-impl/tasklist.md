# Tasklist — `src/simple/` Reference Implementation

開始: 2026-05-12。

> 関連 spec: [requirements.md](requirements.md) / [design.md](design.md)

この tasklist は**マイルストーン単位**で粒度を粗く保つ。日々の細かい作業は
Claude Code 内蔵の TaskCreate で管理し、ここには反映しない。完了したマイルストーンは
`- [x]` でチェックし、完了日を併記する。

## テストファイルの配置規約

- `test/simple/` — Simple 版固有のテスト (XML parser, CG, energy 内部一致 など)
- `test/parity/` — simple vs optimized の数値一致テスト
- `test/runtests.jl` から両者を include。slow gate (`"slow" in ARGS`) は `ferh_4x4x4` のみ。

## マイルストーン

### M1. プロジェクト骨格 (完了: 2026-05-12)
- [x] `src/simple/Simple.jl` (サブモジュールのスケルトン: `module Simple` + 後続 include は空)
- [x] `src/SpinClusterMC.jl` を更新:
      ```julia
      include("simple/Simple.jl")
      using .Simple
      export JPhiMagestyCarlo, Simple
      ```
- [x] 空 module でも `make test` が通る (precompile が壊れていないことの確認)
- [x] `julia --project=. -e 'using SpinClusterMC.Simple'` がエラーなく走る

### M2. XML parser (完了: 2026-05-12)
- [x] `src/simple/xml_io.jl` (独立 parser: SALC list, basis, JPhi)
- [x] `bcc_2x2x2 / fege_2x2x2 / ferh_4x4x4` の XML を読んで `parse_jphi_xml` が
      ClusterInstance 候補のデータを返せる (型は M3 まで暫定)
- [x] 既存 `src/xml_io.jl` の `parse_system_xml` 相当を simple 側で書き直し
      (system.xml の lattice/positions/translations)
- [x] テスト: `test/simple/test_simple_xml.jl` (3 fixture 全部ロード成功)

### M3. 型 + CGTable (完了: 2026-05-12)
- [x] `src/simple/types.jl` (`SpinClusterHamiltonian`, `ClusterInstance`, `CGTable`)
- [x] `src/simple/cg.jl` (CGTable 構築: `Magesty.AngularMomentumCoupling.build_all_real_bases`
      を unique `ls` ごとに 1 回呼ぶ)
- [x] CGTable のキー長 invariant `length(Lseq) == max(0, N-2)` をコンストラクタで assert
- [x] **G9 判断**: `Magesty.AngularMomentumCoupling` API (`build_all_real_bases`,
      `enumerate_paths_left_all`) のシグネチャを確認 (`Magesty = "0.1.0"` で互換 OK)
- [x] テスト: `test/simple/test_simple_cg.jl` (CGTable shape 確認、Magesty 直叩きとの一致)

### M4. Energy API (完了: 2026-05-12)
- [x] `src/simple/energy.jl` の以下 4 関数:
  - [x] `total_energy(h, spins)`
  - [x] `local_energy(h, spins, i)`
  - [x] `delta_local_energy(h, spins, i, S_new)`
  - [x] `gradient(h, spins, i)`
- [x] 数式を実装する関数の docstring に LaTeX 数式 + Magesty docs 参照を入れる
- [x] テスト: `test/simple/test_simple_energy.jl` (total = sum(local) / N の内部一致、
      gradient の数値微分一致)
- [x] テスト: `test/parity/test_parity_bcc.jl` (`rtol = 1e-8` for total, `1e-7` for delta)

### M5. 外場 (完了: 2026-05-12)
- [x] `src/simple/external.jl` (`ExternalTerm` 抽象 + `Zeeman` 実装)
- [x] `ExternalTerm` も `local_energy / delta_local_energy / gradient` を実装
- [x] テスト: `test/simple/test_simple_external.jl` (Zeeman の `gradient = -m_i·B` 等)

設計判断:
- 磁気モーメントの大きさ `m_i` は `MomentModel` 抽象で表現
  (`UniformMoment` / `PerSiteMoment` で副格子依存に対応、将来
  `ClusterExpansionMoment` で局所環境依存に拡張可能 — `moment_at(model, i, spins)`
  query API が `spins` を受けるため signature を変えずに拡張できる)
- Zeeman: `E = -Σ_i m_i (B · S_i)`
- 単位は積 `m_i · B` が eV になるよう呼び出し側で揃える

### M6. Spin proposal + initial spins (完了: 2026-05-12)
- [x] `src/simple/spin_proposal.jl`:
  - [x] `_rand_unit_spin(rng)` — `SVector{3, Float64}` 返し (optimized 側 Tuple とは別物)
  - [x] `_propose_spin_geodesic(rng, u, theta_max)` — SVector 入出力、θmax=0 で early return
  - [x] `init_spins(spec, n_atoms, base_n_atoms; rng)`: Symbol / Tuple /
        AbstractVector / AbstractMatrix (base or supercell) / AbstractDict
        の multi-dispatch、`n_atoms == base_n_atoms` のとき as-is 優先
- [x] テスト: `test/simple/test_simple_spin_proposal.jl` (各モード × normalize 確認 +
      geodesic stays-on-sphere + θmax 上限尊重 + validation)

### M7. MC 型 + Carlo glue (完了: 2026-05-12)
- [x] `src/simple/mc.jl`:
  - [x] `mutable struct SCEMC <: Carlo.AbstractMC`
  - [x] `Carlo.init!`, `Carlo.measure!`, `Carlo.register_evaluables`
        (Magnetization 4 観測量 + Energy/Energy2 + `:SpecificHeat` / `:BinderRatio` /
        `:Susceptibility`)
  - [x] `extra_measure` / `extra_evaluables` callback
  - [x] PT 後付け可能な field 配置 (T mutable, energy field, xml_path, repeat 保持)
- [x] `src/simple/updates/metropolis.jl` (`metropolis_sweep!`)
- [x] `Carlo.sweep!` 内の dispatch (v1 は `:metropolis` のみ; constructor で
      validate して unsupported scheme は ArgumentError)
- [x] 周期的 renormalization (`renorm_every`, default 1000) + energy drift check
      (rtol=1e-10 + atol=1e-12; assert で error)
- [x] **G4 判断**: `enabled_bodies` は v1 不採用 (Simple は教材性優先、optimize 側機能を
      全部再現しない)。requirements.md / design.md に明記。
- [x] テスト: `test/simple/test_simple_mc.jl` (default + user-supplied params、init、sweep、
      drift、measure、Zeeman 統合、validation)
- [x] テスト: `test/parity/test_parity_fege.jl` (Lf=0..4 異方性含む、`rtol = 1e-8` for
      total、`1e-7` for delta、`sum_local / 2 = total` identity)

設計判断:
- 温度は Kelvin 入力 (constructor で eV 変換) — Simple API のみ。optimize 側は
  eV 入力のまま (CLAUDE.md「物理規約」明記)。
- ExternalTerm は `external::Union{Nothing, ExternalTerm}`、`_external_*` dispatch
  helper で `Nothing` 経路を type-stable に no-op 化 (JET union split 対応)。

### M8. Examples (完了: 2026-05-12)
- [x] `examples/01_quickstart.jl` (bcc_2x2x2、SCEMC + Carlo manual loop)
- [x] `examples/02_cooling_run.jl` (T 高→低 simulated annealing、`mc.T` mutation で
      spin 状態を引き継ぎ、CSV 出力)
- [x] `examples/03_anisotropy_demo.jl` (fege_2x2x2、Lf>0 で +x̂/+ŷ/+ẑ/diagonal の
      energy 差を比較)
- [x] `examples/04_initial_spin_presets.jl` (`init_spins` 全 mode: Symbol / Tuple /
      AbstractVector / SVector / Matrix(base) / Matrix(supercell) / AbstractDict)
- [x] `examples/05_custom_observable.jl` (ferh_4x4x4、`params[:extra_measure]`
      callback で Fe/Rh 副格子磁化を記録。ferh の現状性能制約から sweep=1 は暫定値)
- [x] `examples/README.md` (30 秒 quickstart + 30 分 reading order + 規約)
- [x] **G7**: CI smoke test を組み込み。`Makefile` に `make examples-smoke` を追加し
      短時間 example (01, 03, 04) のみ実行。`.github/workflows/CI.yml` に step 追加。

設計判断:
- 全 example が `SCEMC + Carlo.MCContext` の manual loop で構成 (Carlo の job runner
  `Carlo.start(...)` は使わない、教材性優先)。
- 各 example 冒頭に「Pedagogical vs production notes」を入れ、本番運用との差分
  (raw sampling vs `Carlo.measure!` + binning、CSV vs HDF5 結果ファイル、等) を明示。
- `:repeat` / `:external` / `:update_scheme` の **デフォルト値を全 example に明記**
  (ユーザーが SCEMC の全 params を 1 つの dict から把握できる)。
- 温度単位: `params[:T]` は Kelvin (constructor で eV 変換) を全 example で踏襲。
- output ファイル (`cooling_results.csv` 等) は `.gitignore` で除外。

### M9. Benchmark (完了: 2026-05-13)
- [x] `benchmark/simple/fixtures.jl` (3 fixture ロード共通化 + `simple_avg_time` warm-up)
- [x] `benchmark/simple/bench_construction.jl` (XML / CG / Hamiltonian)
- [x] `benchmark/simple/bench_energy.jl` (total / local / delta_local / gradient)
- [x] `benchmark/simple/bench_sweep.jl` (metropolis_sweep! + per-flip)
- [x] `benchmark/simple/bench_compare.jl` (simple vs optimized: `sce_energy` reference
      と `_energy_from_instances_cached` production fast path の両方と比較、rel-err も出力)
- [x] `benchmark/simple/runbench.jl` (各 bench を個別 Julia process で連続実行、`--fast` で軽量)
- [x] `benchmark/simple/README.md`
- [x] **G8**: `benchmark/README.md` (parent) を新規作成、`optimized/` と `simple/` の
      使い分けを表形式で説明

設計判断:
- BenchmarkTools 非依存 (optimized 側のスタイルに合わせて `@elapsed` + checksum)
- 共通 helper を `fixtures.jl` に集約: `SIMPLE_FIXTURES`, `simple_parse_args`,
  `simple_parse_repeat`, `simple_random_spins`, `simple_fmt_time`, `simple_avg_time`
- `simple_avg_time` は最初に 1 回 untimed warm-up を実行 (closure-specialization 等の
  初期コストを timed loop から除外)
- 各スクリプトの fixture loop ヘルパーは `bench_fixture(xml, repeat, ...)` で統一
- `bench_compare.jl` の cached path は `_build_zlm_cache` の re-build を毎回含めて測る
  (`Carlo.init!` 内の実コストと同じ)
- ferh は `bench_sweep` / `bench_compare` のデフォルトから除外 (`--fixtures=ferh` で
  個別指定可)。理由: 839 936 cluster instances + Simple 側 SH cache 未実装で時間がかかる。

### M10. 完了確認
- [ ] requirements.md の「完了基準」全項目クリア
- [ ] テスト: `test/parity/test_parity_ferh.jl` を追加し `make test-slow` で通過
- [ ] `make test` 通過 (slow テスト含む)
- [ ] `parity-checker` サブエージェント (保留中だが、ここまで来たら導入を再検討)
- [ ] design_notes.md の simple-impl pointer を「完了」マーク付きに更新
- [ ] **G6**: `CLAUDE.md` の `make test ~10s` を実態 (推定 50〜90s) に更新。slow test 規約は維持。
- [ ] **G10**: `docs/src/api.md` に `## Simple Module` セクションを追加し、
      `using Documenter; @docs` で Simple 版の公開 API を出力。

## メモ

- M1〜M3 はモジュール骨格。順序固定。
- M4 と M5 は並列着手可。
- M6 は M4 と独立だが、M7 が両方に依存。
- M8〜M9 は M7 後にまとめて。
- 各マイルストーン完了時に **テストが通ること**を確認してから次へ進む。
- 中規模以上の判断 (e.g., `enabled_bodies` の要否) はその場で design.md に追記する。
