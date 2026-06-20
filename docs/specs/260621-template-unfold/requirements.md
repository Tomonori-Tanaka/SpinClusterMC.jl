# Requirements — Phase 2: template kernel un-fold + `repeat` 統一

開始: 2026-06-21。

> 関連 spec: [design.md](design.md) / [tasklist.md](tasklist.md)
> 前提: [`260620-general-supercell`](../260620-general-supercell/) (Simple un-fold),
> [`260620-optimized-general-supercell`](../260620-optimized-general-supercell/) (optimized :tensor un-fold)

## 目的

Phase 1 で `supercell_matrix`(一般 M, un-fold) を **`:tensor` カーネル限定**で
実装した。Phase 2 では:

1. **高速 `:tensor_template` カーネルを un-fold + 一般 M 対応**にする
   (O(n_templates) メモリの利点を保つ)。
2. **`repeat=(n1,n2,n3)` を `supercell_matrix = reshape_base*diag(n)` の糖衣に
   統一**し、`repeat` も un-fold 経由(幾何的に正しい + 高速)にする。
   **folded(索引ベース複製)の挙動を廃止**する。

物理規約: クラスタは相対ベクトルで定義される幾何的対象。un-fold が正
(face ペアを ±Δ の別原子に展開)。Magesty 規約の folded は基本セルの索引ペアを
畳んだまま複製する有限サイズ近似であり、幾何的な正当性を後方互換より優先して
**廃止する** (2026-06-21 user 確定、open decision 1)。

## スコープ

### 含む

- **`:tensor_template` の un-fold 化** (`src/template_energy.jl`):
  - `BaseClusterInstance{,2,3}` を base 原子ベースから primitive
    (pivot_subl, site_subl, site_delta) ベースに作り直す。
  - `build_local_energy_template` を `prim`(PrimitiveCell) + `M` 入力で構築
    (共有 `_cluster_offsets`/`_enumerate_cells`/`_wrap_offset_into_supercell`)。
    `effective_mult = multiplicity ÷ s_base` を prefactor に反映。
  - SAI テーブルを **primitive cell-major (cell_id, subl)** で precompute
    (`_tile_coords`/`supercell_atom_index` を廃止)。
  - `_template_local_energy!` を `i → (cell_id, subl)` 分解 + cell ベース SAI で
    書き換え。N=2/3 高速パス + N≥4 on-the-fly wrap。
- **`repeat` の un-fold 統一**:
  - `load_sce_hamiltonian` / `Simple.SpinClusterHamiltonian` で
    `repeat=(n1,n2,n3)` を `M = reshape_base*diag(n)` に変換し un-fold パスへ。
  - `repeat` と `supercell_matrix` が同一結果になる (両 un-fold)。
  - **folded コードの撤去**: `supercell_atom_index`, `_tile_coords`,
    `_foreach_base_instance`(folded), `_foreach_translated_instance`,
    `coupled_cluster_energy`, folded `_build_supercell_geometry`, Simple の
    folded `_generate_instances`/`_supercell_atom_index`。
- **`init_spins` / `_tile_base_spins!`**: primitive cell-major では base-cell
  タイリングが非整合 → 設計判断 (下記 open decision)。
- **テスト更新**:
  - `:tensor_template` ↔ `:tensor` ↔ Simple が一般 M で一致 (2 パス整合性)。
  - 既存 repeat>1 テスト (bcc (2,2,2) 等) の期待値を un-fold 値に更新。
  - `supercell_atom_index`/folded 関数の直接テストを撤去。
  - parity (Simple↔optimized) は両 un-fold で一致 ((1,1,1) は不変)。
- ドキュメント更新 (`repeat` が un-fold になった旨、folded 廃止)。

### 含まない

- ベンチマーク追加 (Phase 2 完了後、別途)。
- 新しい観測量・物理量の追加。
- `jphi.xml` / Magesty 側の変更。

## 不変条件 (絶対に守る)

- 物理規約は不変: スピン `3×n_atoms`、温度 (optimized は eV)、per-atom 観測量、
  実テッサー `Zlm`、`Φᵥ`、`E = Σ contract·multiplicity·(4π)^(N/2)`。
- **`:tensor_template` と `:tensor` は同一エネルギーを出す** (2 パス整合性、
  CLAUDE.md 連動箇所)。一般 M でも一致。
- **`repeat=(1,1,1)` の結果は不変** (folded≡un-fold at base cell)。既存
  (1,1,1) テスト・parity は無改変 pass。
- **`repeat` と `supercell_matrix` が同一結果** (両 un-fold)。
- 低レベル縮約カーネル `_tensor_contract_*` の数式は不変 (atoms を渡す側のみ変更)。
- メモリ/性能は folded template と同等 (O(n_templates) を保つ)。

## 完了基準

- `make test` / `make test-slow` 全 pass。一般 M で `:tensor_template` ↔
  `:tensor` ↔ Simple が一致。`repeat` = `supercell_matrix` 等価。
- folded コードが撤去され、`repeat` が un-fold 糖衣として動く。
- MPI serialize round-trip・キャッシュが M/repeat 両方で動く。
- `code-reviewer` / `numerical-reviewer` レビュー pass。

## Open decisions (確定済み, 2026-06-21 user)

1. **folded コードの撤去範囲** → **全面撤去**。`coupled_cluster_energy` 含む folded
   関数を全削除し un-fold に一本化。reference は un-fold instance-list (`sce_energy`
   = `_build_cluster_instances(_matrix)` の総和)。
2. **`init_spins` base-cell タイリング** → **廃止**。`:random` か full `3×n_atoms`
   配列のみ許可 (matrix パスの既存ガードを `repeat` 経路にも適用)。
3. **`repeat` API** → **残す (un-fold 糖衣)**。`repeat=(n)` を `M=reshape_base*
   diag(n)` に変換。`:supercell` エイリアスも維持。API 表面・既存スクリプトは
   そのまま (値は n>1 で un-fold に変わる)。`register_evaluables`/constructor の
   `_parse_repeat_param` 経路も M ベースに連動。
4. **serialize 互換性** → **非互換を許容** (「同一バージョン内のみ」前提)。
   `mc.supercell_matrix`(M) を保持。`mc.repeat` は記録用に残してよい (M に変換済)。
