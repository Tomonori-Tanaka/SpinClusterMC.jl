# Tasklist — `src/simple/` Reference Implementation

開始: 2026-05-12。

> 関連 spec: [requirements.md](requirements.md) / [design.md](design.md)

この tasklist は**マイルストーン単位**で粒度を粗く保つ。日々の細かい作業は
Claude Code 内蔵の TaskCreate で管理し、ここには反映しない。完了したマイルストーンは
`- [x]` でチェックし、完了日を併記する。

## マイルストーン

### M1. プロジェクト骨格
- [ ] `src/simple/Simple.jl` (サブモジュールのスケルトン: `module Simple` + `include`s + `export`)
- [ ] `src/SpinClusterMC.jl` から `include("simple/Simple.jl")` を追加し、`using .Simple` だけ済ませる (export はしない、`SpinClusterMC.Simple.xxx` 経由でアクセス)
- [ ] 空 module でも `make test` が通る (precompile が壊れていないことの確認)

### M2. XML parser
- [ ] `src/simple/xml_io.jl` (独立 parser: SALC list, basis, JPhi)
- [ ] `bcc_2x2x2 / fege_2x2x2 / ferh_4x4x4` の XML を読んで `parse_jphi_xml` が
      ClusterInstance 候補のデータを返せる (型は M3 まで暫定)
- [ ] 既存 `src/xml_io.jl` の `parse_system_xml` 相当を simple 側で書き直し
      (system.xml の lattice/positions/translations)

### M3. 型 + CGTable
- [ ] `src/simple/types.jl` (`SpinClusterHamiltonian`, `ClusterInstance`, `CGTable`)
- [ ] `src/simple/cg.jl` (CGTable 構築: `Magesty.AngularMomentumCoupling.build_all_real_bases`
      を unique `ls` ごとに 1 回呼ぶ)
- [ ] CGTable のキー長 invariant `length(Lseq) == N - 2` をコンストラクタで assert

### M4. Energy API
- [ ] `src/simple/energy.jl` の以下 4 関数:
  - [ ] `total_energy(h, spins)`
  - [ ] `local_energy(h, spins, i)`
  - [ ] `delta_local_energy(h, spins, i, S_new)`
  - [ ] `gradient(h, spins, i)`
- [ ] 数式を実装する関数の docstring に LaTeX 数式 + Magesty docs 参照を入れる
- [ ] テスト: 同 XML + 同 spins で `total_energy(simple) ≈ sce_energy(optimized) rtol=1e-8`

### M5. 外場
- [ ] `src/simple/external.jl` (`ExternalTerm` 抽象 + `Zeeman` 実装)
- [ ] `ExternalTerm` も `local_energy / delta_local_energy / gradient` を実装

### M6. Spin proposal + initial spins
- [ ] `src/simple/spin_proposal.jl`:
  - [ ] `_rand_unit_spin(rng)`
  - [ ] `_propose_spin_geodesic(rng, ux, uy, uz, theta_max)`
  - [ ] `init_spins(params, n_atoms, base_n_atoms)`: `:initial_spins` を型 dispatch
        (Symbol `:random` / `:ferromagnetic`, Tuple, SVector, Matrix (base or supercell))
- [ ] テスト: 各 initial_spins モードで `Carlo.init!` が正常に動く

### M7. MC 型 + Carlo glue
- [ ] `src/simple/mc.jl`:
  - [ ] `mutable struct SCEMC <: Carlo.AbstractMC`
  - [ ] `Carlo.init!`, `Carlo.measure!`, `Carlo.register_evaluables`
  - [ ] `extra_measure` / `extra_evaluables` callback
  - [ ] PT 後付け可能な field 配置 (T mutable, energy field, xml_path, repeat 保持)
- [ ] `src/simple/updates/metropolis.jl` (`metropolis_sweep!`)
- [ ] `Carlo.sweep!` から `params[:update_scheme]` で dispatch
- [ ] 周期的 renormalization (`renorm_every`, default 1000)
- [ ] テスト: 短い MC run (seed 固定) で `:Energy` 系列が optimized 版と
      `rtol=1e-8` で一致

### M8. Examples
- [ ] `examples/01_quickstart.jl` (bcc_2x2x2)
- [ ] `examples/02_cooling_run.jl` (T 高→低スキャン、CSV 出力)
- [ ] `examples/03_anisotropy_demo.jl` (fege_2x2x2, Lf>0 の方向選好)
- [ ] `examples/04_initial_spin_presets.jl` (`:random` / `:ferromagnetic` / SVector / Matrix)
- [ ] `examples/05_custom_observable.jl` (ferh_4x4x4, Fe/Rh 副格子磁化を callback で追加)
- [ ] `examples/README.md` (30 秒 quickstart + 30 分読む順序)
- [ ] CI smoke test (短時間 example のみ)

### M9. Benchmark
- [ ] `benchmark/simple/fixtures.jl` (3 fixture ロード共通化)
- [ ] `benchmark/simple/bench_construction.jl` (XML / CG / Hamiltonian)
- [ ] `benchmark/simple/bench_energy.jl` (total / local / delta_local / gradient)
- [ ] `benchmark/simple/bench_sweep.jl` (metropolis_sweep! + per-flip)
- [ ] `benchmark/simple/bench_compare.jl` (simple vs optimized 比率)
- [ ] `benchmark/simple/runbench.jl` (集約サマリ)
- [ ] `benchmark/simple/README.md`

### M10. 完了確認
- [ ] requirements.md の「完了基準」全項目クリア
- [ ] `make test` 通過 (slow テスト含む)
- [ ] `parity-checker` サブエージェント (保留中だが、ここまで来たら導入を再検討)
- [ ] design_notes.md の simple-impl pointer を「完了」マーク付きに更新

## メモ

- M1〜M3 はモジュール骨格。順序固定。
- M4 と M5 は並列着手可。
- M6 は M4 と独立だが、M7 が両方に依存。
- M8〜M9 は M7 後にまとめて。
- 各マイルストーン完了時に **テストが通ること**を確認してから次へ進む。
- 中規模以上の判断 (e.g., `enabled_bodies` の要否) はその場で design.md に追記する。
