# Tasklist — Phase 2: template un-fold + `repeat` 統一

> 関連 spec: [requirements.md](requirements.md) / [design.md](design.md)

完了時に `- [x]` + 完了日 (途中状態は触らない)。日々の作業は `TaskCreate`。
各段階で `make test` を維持しながら進める。

## P2-M1: template un-fold 構築 (sweep はまだ legacy)

- [ ] `BaseClusterInstance{,2,3}` を primitive (pivot_subl/site_subl/site_delta)
      ベースに作り直す (`src/template_energy.jl`)。
- [ ] `build_local_energy_template` を `prim`+`M` 入力で構築
      (`_cluster_offsets`/`_cluster_base_stabilizer`/`effective_mult`)。
      `related{,2,3}_by_subl`、`LocalEnergyTemplate` に prim/M/cell_index 保持。
- [ ] `_build_sai_table_n` を cell-major precompute に作り直す
      (`_enumerate_cells`/`_wrap_offset_into_supercell`)。`_tile_coords`/
      `supercell_atom_index` 依存を除去。
- [ ] 単体: 構築した SAI が `_build_cluster_instances_matrix` の instance 集合と
      整合 (同じ supercell 原子ペア)。

## P2-M2: sweep + 2 パス整合

> 依存: P2-M1。

- [ ] `_template_local_energy!` を `i→(cell_id,subl)` 分解 + cell-major SAI で
      書き換え (N=2/3 fast, N≥4 on-the-fly wrap)。
- [ ] `:tensor_template` と `:tensor` が一般 M・random 配置で **総エネルギー一致**
      (init! 初期化 + sweep! ΔE 両方)。
- [ ] `:tensor_template`+`supercell_matrix` のエラー (Phase 1) を解除。

## P2-M3: `repeat` un-fold 統一 + folded 撤去

> 依存: P2-M2。

- [ ] `load_sce_hamiltonian`/`Simple.SpinClusterHamiltonian`: `repeat` を
      `M = reshape_base*diag(repeat)` に変換し un-fold パスへ合流。
- [ ] folded コード撤去 (open decision 1 で確定した範囲):
      `supercell_atom_index`/`_foreach_translated_instance`/
      `coupled_cluster_energy`/folded geometry/Simple `_generate_instances`/
      `_tile_coords`/`_foreach_base_instance`/`_tile_base_matrix`。
- [ ] `sce_energy` の `repeat`/`coupled_cluster_energy` 分岐を撤去し、M 変換後は
      常に `_build_cluster_instances(h)` (instance 総和) パスのみにする。
- [ ] 原子番号付けが tile-major → cell-major に変わる (破壊的)。`mc.spins` の
      列順に依存するテスト・docstring・serialize を要確認 (design §3)。
- [ ] `repeat` == `supercell_matrix` 等価を確認。`repeat=(1,1,1)` 不変。

## P2-M4: init_spins / cache / MPI

> 依存: P2-M3。

- [ ] `init_spins`/`_tile_base_spins!` を open decision 2 に従い対応
      (推奨: base-cell タイリング廃止 → random or full config)。
- [ ] キャッシュキーを M ベースに一本化、`register_evaluables`/serialize 整理。
- [ ] MPI serialize round-trip (repeat 由来 M / 直接 M 両方)。

## P2-M5: テスト全更新

> 依存: P2-M2〜M4。

- [ ] 既存 repeat>1 テスト (bcc (2,2,2) 等) を un-fold 期待値に更新。
- [ ] tile-major 前提テストを cell-major に書き換えるか撤去
      (`test/bcc_2x2x2`: "initial_spins tiling" / "cross-tile",
      `test/simple/test_simple_spin_proposal.jl`: Matrix tiling)。
- [ ] `supercell_atom_index`/folded 関数の直接テストを撤去。
- [ ] parity に `:tensor_template`×M を追加。"intended divergence" テスト撤去。
- [ ] `make test` / `make test-slow` 全 pass、JET pass。

## P2-M6: docs + レビュー

- [ ] `repeat` が un-fold 糖衣・folded 廃止を docstring/terminology/README に反映。
      `design_notes.md` の Phase 2 future-work を「完了」に。
- [ ] `JuliaFormatter` (src/simple, 共有, test/simple, test/parity)。
      `template_energy.jl`/`JPhiMagestyCarlo.jl` は既存スタイル維持で整形対象外
      (CLAUDE.md)。
- [ ] `code-reviewer` / `numerical-reviewer`。
