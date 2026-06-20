# Requirements — Optimized 版の一般スーパーセル対応 (Phase 1)

開始: 2026-06-20。

> 関連 spec: [design.md](design.md) / [tasklist.md](tasklist.md)
> 前提 spec: [`260620-general-supercell`](../260620-general-supercell/) (Simple 版で完了)

> **実装中の規約確定 (2026-06-21)**: Magesty 規約は folded だが、user 判断で
> **un-fold(`supercell_matrix`)を物理的に正しいものとして採用**(クラスタは相対
> ベクトルで定義される幾何的対象。`repeat` の folded は基本セル索引ペアを畳んだ
> まま複製する有限サイズ近似で、n>1・非共線で un-fold と異なる)。本 spec は
> un-fold の `supercell_matrix` を提供(:tensor)。`repeat` は当面 folded・高速の
> まま維持し、**Phase 2 で高速 template カーネルを un-fold 化 → `repeat` を統一**
> (この順序でないと大規模性能が壊れる)。詳細: メモリ
> `project_sce_self_overlap_convention` / `feedback_correctness_over_compat`。

## 目的

Simple 実装で完成した「primitive セルから任意の 3×3 整数 supercell 行列 `M`
でタイリングする」機能を、**optimized 実装 (`src/JPhiMagestyCarlo.jl` /
`src/template_energy.jl`)** にも広げる。これにより MPI/PT・Zlm キャッシュ付きの
高速 MC を、対角 `repeat` だけでなく非対角・非整数倍・斜めセル
(spiral/AFM 秩序ベクトル等) でも実行できるようにする。

物理規約は Simple と同一: SCE エネルギー = `Σ contract · multiplicity ·
(4π)^(n/2)`、面上 (半周期) クラスタは `effective_mult = multiplicity ÷ s_base`
で un-fold する。

## スコープ (Phase 1)

### 含む

- **共有モジュール `src/supercell_common.jl`** (`SupercellCommon`) を新設し、
  Simple と optimized の双方から使う:
  - 整数線形代数 `_int_det3` / `_adjugate3` / `_col_hermite` /
    `_wrap_offset_into_supercell`。
  - `PrimitiveCell` 型と `extract_primitive`(生配列引数版) /
    `_cluster_base_stabilizer` / `_enumerate_cells` /
    `_supercell_from_repeat` / クラスタ offset 幾何ヘルパ。
  - Simple の `src/simple/supercell.jl` をこの共有モジュール利用に refactor
    (Simple の挙動は不変、既存テストで担保)。
- **optimized の `:tensor` リファレンス/キャッシュパスを一般 `M` 対応**:
  - `load_sce_hamiltonian(xml; repeat, supercell_matrix, jphi_threshold)`。
  - 一般 `M` のとき instance を明示列挙する新タイリング
    (`_build_cluster_instances` の M 版)。`effective_mult = multiplicity ÷
    s_base` + accumulate-dedup。primitive cell-major 付番。
  - `JPhiSpinMC` の `params[:supercell_matrix]`。一般 `M` のときカーネルは
    **`:tensor` を強制** (sweep! は `_related_instances_by_atom` ベースで動く)。
  - `sce_energy` (total) が一般 `M` Hamiltonian で動くこと (instance 総和)。
  - キャッシュ (`_HAM_CACHE` / `_ECACHE_CACHE` / `_DERIVED_CACHE`) のキーに `M`
    を含める。MPI serialize/deserialize に `M` を含め round-trip で再構築。
- **後方互換**: 対角 `repeat` パスは legacy のまま完全不変。
- **テスト**:
  - optimized 一般 `M` の `sce_energy` が **Simple 一般 `M` と parity 一致**
    (複数 `M`: base 倍数・非倍数・非対角)。
  - 既存 optimized/parity/JET テストが無改変で pass。
  - MPI serialize round-trip (`M` 込み)。
  - エラー: `:tensor_template` + `supercell_matrix` は明示エラー (Phase 2 案内)。
- **ユーザー向けドキュメント / 例**:
  - `examples/06_general_supercell.jl` (新規, Simple `SCEMC` で `supercell_matrix`
    を使い非対角・非整数倍セルを構築するデモ。optimized `JPhiSpinMC` も同じ
    `:supercell_matrix` を受ける旨をコメント) + `examples/README.md` の表に追記。
  - `README.md` の supercell 節を一般 3×3 整数行列 `supercell_matrix` 対応
    (Simple + optimized Phase 1、対角は従来通り) に更新。

### 含まない (Phase 2 以降 / 別 spec)

- **`:tensor_template` 高速パスの一般 `M` 対応** (SAI テーブル /
  `_tile_coords` / `_foreach_base_instance` の primitive cell-major 作り直し)。
  一般 `M` では当面 `:tensor` カーネルを使う。
- `coupled_cluster_energy` リファレンス path の一般 `M` 対応 (必要なら instance
  ベースの total で代替。`coupled_cluster_energy` 自体は対角のまま)。
- Simple 側の機能変更 (refactor のみ、挙動不変)。
- **ベンチマーク** (`benchmark/`)。matrix パスは Phase 1 では `:tensor` のみで、
  `:tensor_template` との非対称比較になるため、ベンチ追加は Phase 2
  (template 高速パス対応後) に回す。

## 不変条件 (絶対に守る)

- 物理規約は不変: スピン `3×n_atoms`、温度 (optimized は eV 受け取り)、
  per-atom 観測量、実テッサー `Zlm`、`Φᵥ`、`E = Σ contract·multiplicity`。
- **対角 `repeat` パスは bit-exact に不変** (既存 optimized/parity テストが
  無改変 pass)。`:tensor_template` デフォルトカーネルも対角では一切変えない。
- 一般 `M` の optimized エネルギーは **Simple の一般 `M` と parity 一致**
  (同じ物理、付番は primitive cell-major で両者一致)。
- 共有モジュール化で Simple の挙動を変えない (既存 Simple テストが担保)。
- 低レベルカーネル (`_tensor_contract_instance*`) は触らない (instance を
  渡す側だけ M 対応)。

## 完了基準

- `make test` / `make test-slow` 全 pass (既存 + 新規 parity + JET)。
- optimized `params[:supercell_matrix]` で MC が走り、`sce_energy` が Simple と
  一致 (parity 範囲内)。
- MPI serialize round-trip が `M` を保持。
- `:tensor_template` + `supercell_matrix` が明示エラー。
- `code-reviewer` / `numerical-reviewer` レビュー pass。
