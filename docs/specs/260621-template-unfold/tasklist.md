# Tasklist — Phase 2: template un-fold + `repeat` 統一

> 関連 spec: [requirements.md](requirements.md) / [design.md](design.md)

完了時に `- [x]` + 完了日 (途中状態は触らない)。日々の作業は `TaskCreate`。
各段階で `make test` を維持しながら進める。

> **✅ Phase 2 完了・クローズ (2026-06-21)**。`main` にマージ済み
> (`f4373b3..1b4af0f`)。`:tensor_template` 高速カーネルを一般 M に un-fold 対応し、
> `repeat` を `M = reshape_base·diag(repeat)` の糖衣として folded コードを撤去、
> N=2/3 特殊化カーネルで sweep を高速化 (fege 1.90× / ferh 2.45×、bit-identical)。
> **唯一の descoped**: キャッシュキー M 一本化 (P2-M4、churn のため見送り、下記)。

---

## P2-M1: template un-fold 構築 (sweep はまだ legacy)

> 実装で型/関数名が変わった: `BaseClusterInstance{,2,3}`→`PrimClusterTemplate`、
> `_build_sai_table_n`→`_build_sai_table_cellmajor`+`UnfoldSAITable`、
> `_build_cluster_instances_matrix`→`_build_cluster_instances`。

- [x] cluster template を primitive (pivot_subl/site_subl/site_delta) ベースで構築
      (`PrimClusterTemplate`, `src/template_energy.jl`)。 (2026-06-21)
- [x] `build_local_energy_template` を `prim`+`M` 入力で構築
      (`_cluster_offsets`/`_cluster_base_stabilizer`/`eff_mult`)。 (2026-06-21)
- [x] per-atom cell-major precompute (`_build_sai_table_cellmajor`→`UnfoldSAITable`)。
      `_tile_coords`/`supercell_atom_index` 依存を除去。 (2026-06-21)
- [x] 単体: 構築した SAI が `_build_cluster_instances` の instance 集合と整合
      (test_supercell_matrix.jl "primitive cell-major template SAI ...")。 (2026-06-21)

## P2-M2: sweep + 2 パス整合

> 依存: P2-M1。

- [x] `_template_local_energy!` を `i→(cell_id,subl)` 分解 + cell-major SAI で
      書き換え (N=2/3 fast `_contract_n{2,3}_unfold_changed`, N≥4 generic
      fallback)。同一プロセス A/B で fege 1.90× / ferh 2.45×、bit-identical。
      (2026-06-21)
- [x] `:tensor_template` と `:tensor` が一般 M・random 配置で **総エネルギー一致**
      (init! + sweep! ΔE、bit-for-bit トラジェクトリ一致まで検証)。 (2026-06-21)
- [x] `:tensor_template`+`supercell_matrix` のエラー (Phase 1) を解除。 (2026-06-21)

## P2-M3: `repeat` un-fold 統一 + folded 撤去

> 依存: P2-M2。

- [x] `load_sce_hamiltonian`/`Simple.SpinClusterHamiltonian`: `repeat` を
      `M = reshape_base*diag(repeat)` に変換し un-fold パスへ合流。 (2026-06-21)
- [x] folded コード撤去 (open decision 1 で確定した範囲):
      `supercell_atom_index`/`_foreach_translated_instance`/
      `coupled_cluster_energy`/folded geometry/Simple `_generate_instances`/
      `_tile_coords`/`_foreach_base_instance`/`_tile_base_matrix`。 (2026-06-21)
- [x] `sce_energy` の `repeat`/`coupled_cluster_energy` 分岐を撤去し、M 変換後は
      常に `_build_cluster_instances(h)` (instance 総和) パスのみにする。 (2026-06-21)
- [x] 原子番号付けが tile-major → cell-major に変わる (破壊的)。`mc.spins` の
      列順に依存するテスト・docstring・serialize を確認・更新。base cell 自体が
      primitive supercell (n_trans>1) のため `repeat=(1,1,1)` でも再番号付け
      されるが物理は不変 (requirements 訂正済み)。 (2026-06-21)
- [x] `repeat` == `supercell_matrix` 等価を確認。`repeat=(1,1,1)` 物理不変。
      (2026-06-21)

## P2-M4: init_spins / cache / MPI

> 依存: P2-M3。

- [x] `init_spins`/`_tile_base_spins!`: base-cell タイリング廃止 → random or
      full `3×n_atoms` config のみ受理 (`_tile_base_spins!` 撤去)。 (2026-06-21)
- [~] キャッシュキーを M ベースに一本化 → **見送り (descoped)**。repeat→M 変換に
      `extract_primitive` (XML パース) が要り、キャッシュ参照の前に走らせる必要が
      ある churn。現 dual-key `(xml, repeat, scm_key, thr)` は正しく安価で、同一
      スーパーセルを repeat と supercell_matrix の両方で指定する実運用ケースが無い
      ため利得が無い。`register_evaluables`/serialize は dual-key のまま整合。
- [x] MPI serialize round-trip (repeat 由来 M / 直接 M 両方、両カーネル)。 (2026-06-21)

## P2-M5: テスト全更新

> 依存: P2-M2〜M4。

- [x] 既存 repeat>1 テスト (bcc (2,2,2) 等) を un-fold 期待値に更新。 (2026-06-21)
- [x] tile-major 前提テストを cell-major / full-config に書き換え・撤去。 (2026-06-21)
- [x] `supercell_atom_index`/folded 関数の直接テストを撤去。 (2026-06-21)
- [x] parity に `:tensor_template`×M を追加。"intended divergence" テスト撤去。
      N=3 担保に ferh 単一 primitive cell ケースを通常テストへ追加。 (2026-06-21)
- [x] `make test` + JET pass。 (2026-06-21)
- [x] `make test-slow` 全 pass (N=2/3 カーネル変更後の再確認: ferh_4x4x4
      delta energy consistency + Simple↔optimized parity)。 (2026-06-21)

## P2-M6: docs + レビュー

- [x] `repeat` が un-fold 糖衣・folded 廃止を docstring/terminology/README/
      design_notes に反映 (N=2/3 高速パス + ベンチ含む)。 (2026-06-21)
- [x] `JuliaFormatter` (src/simple, 共有, test/simple, test/parity) 整形済み確認。
      `template_energy.jl`/`JPhiMagestyCarlo.jl` は既存スタイル維持で整形対象外
      (CLAUDE.md)。 (2026-06-21)
- [x] `code-reviewer` / `numerical-reviewer`: P2-M3 (un-fold 統一) と N=2/3 高速パス
      (commit 1b4af0f) の両方をレビュー。いずれも重大 0、N=2/3 は bit-for-bit
      同一を独立確認。 (2026-06-21)
