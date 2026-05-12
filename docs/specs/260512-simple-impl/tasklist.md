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

### M6. Spin proposal + initial spins
- [ ] `src/simple/spin_proposal.jl`:
  - [ ] `_rand_unit_spin(rng)`
  - [ ] `_propose_spin_geodesic(rng, ux, uy, uz, theta_max)`
  - [ ] `init_spins(params, n_atoms, base_n_atoms)`: `:initial_spins` を型 dispatch
        (Symbol `:random` / `:ferromagnetic`, Tuple, SVector, Matrix (base or supercell))
- [ ] テスト: `test/simple/test_simple_spin_proposal.jl` (各モード × normalize 確認)

### M7. MC 型 + Carlo glue
- [ ] `src/simple/mc.jl`:
  - [ ] `mutable struct SCEMC <: Carlo.AbstractMC`
  - [ ] `Carlo.init!`, `Carlo.measure!`, `Carlo.register_evaluables`
        (Magnetization 4 観測量 + Energy/Energy2 + 派生は requirements.md 参照)
  - [ ] `extra_measure` / `extra_evaluables` callback
  - [ ] PT 後付け可能な field 配置 (T mutable, energy field, xml_path, repeat 保持)
- [ ] `src/simple/updates/metropolis.jl` (`metropolis_sweep!`)
- [ ] `Carlo.sweep!` から `params[:update_scheme]` で dispatch
- [ ] 周期的 renormalization (`renorm_every`, default 1000)
- [ ] **G4 判断**: `enabled_bodies` field を入れるかを user に確認。
  - 入れる場合: 既存 JPhiSpinMC と同じ `params[:enabled_bodies]` 規約。
  - 入れない場合: 初版非サポートと design.md / requirements.md に明記、parity テストは
    `params[:enabled_bodies] => nothing` でのみ実施。
- [ ] テスト: `test/simple/test_simple_mc.jl` (init + 1 sweep が走る)
- [ ] テスト: `test/parity/test_parity_fege.jl` (異方性込み、`rtol = 1e-8`)

### M8. Examples
- [ ] `examples/01_quickstart.jl` (bcc_2x2x2)
- [ ] `examples/02_cooling_run.jl` (T 高→低スキャン、CSV 出力)
- [ ] `examples/03_anisotropy_demo.jl` (fege_2x2x2, Lf>0 の方向選好)
- [ ] `examples/04_initial_spin_presets.jl` (`:random` / `:ferromagnetic` / SVector / Matrix)
- [ ] `examples/05_custom_observable.jl` (ferh_4x4x4, Fe/Rh 副格子磁化を callback で追加)
- [ ] `examples/README.md` (30 秒 quickstart + 30 分読む順序)
- [ ] **G7**: CI smoke test を組み込み。`Makefile` に `make examples-smoke` を追加し
      短時間 example (01, 04 等) のみ実行。`.github/workflows/CI.yml` に step 追加。

### M9. Benchmark
- [ ] `benchmark/simple/fixtures.jl` (3 fixture ロード共通化)
- [ ] `benchmark/simple/bench_construction.jl` (XML / CG / Hamiltonian)
- [ ] `benchmark/simple/bench_energy.jl` (total / local / delta_local / gradient)
- [ ] `benchmark/simple/bench_sweep.jl` (metropolis_sweep! + per-flip)
- [ ] `benchmark/simple/bench_compare.jl` (simple vs optimized 比率)
- [ ] `benchmark/simple/runbench.jl` (集約サマリ)
- [ ] `benchmark/simple/README.md`
- [ ] **G8**: `benchmark/README.md` (parent) を新規作成、`optimized/` と `simple/` の関係を 1 段上から説明

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
- ギャップ監査 (2026-05-12) の出処は `/Users/tomorin/.claude/plans/distributed-fluttering-robin.md`。
