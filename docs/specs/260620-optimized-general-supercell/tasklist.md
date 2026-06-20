# Tasklist — Optimized 版の一般スーパーセル対応 (Phase 1)

> 関連 spec: [requirements.md](requirements.md) / [design.md](design.md)

各マイルストーンの完了時に `- [x]` + 完了日を付ける (途中状態は触らない、
`CLAUDE.md` の運用ルール準拠)。日々の細かい作業は `TaskCreate` で管理。

## N1: 共有モジュール `SupercellCommon` 抽出

- [ ] `src/supercell_common.jl` (新規, SciML Style): 整数線形代数・
      `PrimitiveCell`・`extract_primitive`(生配列引数)・`_cluster_base_stabilizer`・
      `_enumerate_cells`・`_supercell_from_repeat`・`_cluster_offsets` を移管。
- [ ] `src/SpinClusterMC.jl`: 両サブモジュールより前に `include`。
- [ ] `src/simple/supercell.jl` / `types.jl` / `mc.jl`: 共有関数を
      `using ..SupercellCommon` 経由に置換。`build_templates` /
      `_generate_instances_matrix` は Simple 固有として残し、共有ヘルパを呼ぶ。
- [ ] 既存 Simple テスト (M1-M6 含む) が **無改変 pass** = 挙動不変の担保。

## N2: optimized 一般 M instance 生成 (`:tensor`)

> 依存: N1 完了後。

- [ ] `_build_cluster_instances_matrix(salc_list, jphi, map_sym, n_trans, prim, M)`:
      `eff_mult = multiplicity ÷ s_base` + accumulate-dedup + primitive
      cell-major 付番で optimized `ClusterInstance` を生成。低レベルカーネルは
      不変。
- [ ] `load_sce_hamiltonian` に `supercell_matrix` kwarg、`SCEHamiltonian` に
      `supercell_matrix` フィールド (legacy=`nothing`、M パスで `repeat=(0,0,0)`)。
      二重指定・`det=0`・3×3 でない のエラー。
- [ ] `sce_energy` に M パス分岐を追加 (`h.supercell_matrix !== nothing` で
      instance 総和 `_energy_from_instances` を返す。legacy は不変)。

## N3: カーネル選択 + キャッシュ + MPI

> 依存: N2 完了後。

- [ ] `JPhiSpinMC`: `params[:supercell_matrix]`、`supercell_matrix` フィールド、
      M パスで kernel を `:tensor` 強制、`params[:energy_kernel]==:tensor_template`
      +M は `ArgumentError`。`init!` で M パス + base-cell サイズ
      `initial_spins` を明示エラー (Simple のガードと同様)。
- [ ] キャッシュキーに `M` (flatten or nothing) を第4要素として追加
      (`_HAM_CACHE`/`_ECACHE_CACHE`/`_DERIVED_CACHE`、`_mpi_build_ham_and_cache`、
      `_get_or_build_derived`)。
- [ ] `Serialization.serialize/deserialize` に `M` を追加し round-trip 再構築。
- [ ] `register_evaluables` の `n_atoms` 取得を M 対応 (ローカル key にも `M`)。

## N4: parity + テスト

> 依存: N2-N3 完了後。

- [ ] `test/parity/test_parity_supercell_matrix.jl`: optimized `sce_energy` ≈
      Simple matrix path total を複数 `M` (base 倍数/非倍数/非対角) で
      ferro + random 配置検証。
- [ ] `test/optimized/test_supercell_matrix.jl`: load/エラー/MPI serialize
      round-trip/sweep!。
- [ ] 重い fege/ferh は slow 節へ。`runtests.jl` に include 追加。
- [ ] `make test` / `make test-slow` 全 pass、JET pass。

## N5: ドキュメント + 例 + レビュー

- [ ] `load_sce_hamiltonian` / `JPhiSpinMC` docstring に `supercell_matrix`、
      `docs/terminology.md` に optimized も対応した旨、`docs/design_notes.md` に
      Phase 2 (template 高速パス + matrix ベンチ) future-work を記録。
- [ ] `examples/06_general_supercell.jl` (新規, Simple `SCEMC` の
      `supercell_matrix` デモ: 非対角・非整数倍) + `examples/README.md` 表に追記。
- [ ] `README.md` の supercell 節を一般 3×3 整数行列対応に更新。
- [ ] `JuliaFormatter` は `src/simple` と新規共有ファイルにのみ適用
      (既存 `JPhiMagestyCarlo.jl` / `template_energy.jl` は別スタイルなので
      一括整形しない — CLAUDE.md)。
- [ ] `code-reviewer` / `numerical-reviewer` で差分レビュー。
