# 用語定義

このリポジトリで使う格子・セル関連の用語を定義する。
コードを読むときや設計メモを書くときの参照用。

---

## セルの階層

### プリミティブセル（primitive cell）

対称性から判定される最小の繰り返し単位。
jphi.xml が定義する基本セルより小さい場合がある。

### 基本セル（base cell）

jphi.xml で定義されるセル。コード中では `base_n_atoms` 個の原子を持つ。
`map_sym`（`base_n_atoms × n_trans` 行列）はこのセル内の原子間の並進対称性を記述する。
プリミティブセルと一致するとは限らない。

### スーパーセル（supercell）

基本セルを `repeat = (n1, n2, n3)` でタイリングしたセル。
原子数は `n_atoms = base_n_atoms × n1 × n2 × n3`。
格子ベクトルは `[n1·a1, n2·a2, n3·a3]`。

#### 一般スーパーセル行列

`Simple.SpinClusterHamiltonian` / `Simple.SCEMC` と、optimized 側の
`JPhiMagestyCarlo.load_sce_hamiltonian` / `JPhiSpinMC` は、対角 `repeat` に
加えて **プリミティブセル単位の一般 3×3 整数行列** `supercell_matrix = M`
（`det(M) ≠ 0`）を受け付ける。これにより非対角・非整数倍・斜めセル
（spiral/AFM 秩序ベクトル等）や、基本セルより小さい単一プリミティブセルまで
扱える。

- optimized 側は Phase 2 で **両カーネル**（`:tensor` リファレンス/キャッシュと
  高速 `:tensor_template`）が一般 M に対応した（いずれも un-fold パス）。
  `:tensor_template` がデフォルトで、`supercell_matrix` 指定でも使える
  （Phase 1 の `:tensor_template` + `supercell_matrix` エラーは撤去済み）。

- プリミティブセルは jphi.xml の並進対称性（`map_sym` / `n_trans`）から復元する
  （`base_lattice = primitive_lattice × reshape_base`、`|det(reshape_base)| =
  n_trans`、`n_prim = base_n_atoms / n_trans` 副格子）。
- 原子数は `n_atoms = n_prim × |det(M)|`。原子番号は **プリミティブ cell-major**
  （`subl + n_prim·(cell_id − 1)`）。`repeat` パスもこの番号付けを共有する
  （旧 tile-major は撤去）。
- **クラスターは相対ベクトルで定義された幾何的対象**として配置される。基本セルで
  自己重なりする面上クラスター（半周期、`multiplicity ≥ 2`）は `multiplicity ÷
  s_base`（`s_base` = クラスターを自分に写す基本セル並進数）で**真の ±Δ 隣接に
  un-fold** される。
  - **`repeat` は `M = reshape_base · diag(n1,n2,n3)` の糖衣**で、`supercell_matrix`
    と同じ un-fold パスを通る。等価な `repeat` / `supercell_matrix` は
    **element-identical な Hamiltonian** を与える（基本セル自体が n_trans>1 の
    primitive supercell なので、`repeat=(1,1,1)` でも primitive cell-major に
    再番号付けされるが物理は不変）。面上ペアは常に幾何的に正しく un-fold される
    （旧 `repeat` の「畳んだまま持ち越す」有限サイズ近似は撤去）。
- `repeat` と `supercell_matrix` は排他（同時指定はエラー）。

詳細は [`docs/specs/260620-general-supercell/`](specs/260620-general-supercell/)、
[`260620-optimized-general-supercell/`](specs/260620-optimized-general-supercell/)、
[`260621-template-unfold/`](specs/260621-template-unfold/)。

---

## クラスター

### クラスターの定義（jphi.xml）

jphi.xml の `<basis atoms="i j ...">` で指定される原子インデックスの組み合わせ。
インデックスは基本セル内の原子番号（1-based）。

**プリミティブセル中の原子が必ず1つ含まれる**。
例：`atoms="1 11"` では原子 1 がプリミティブセルに属する原子。

この性質により、同じクラスター型はスーパーセル内でちょうど
`n1 × n2 × n3`（プリミティブセルと基本セルが一致する場合）以上の回数現れる。

### ClusterInstance（クラスターインスタンス）

クラスターをスーパーセル内の特定の並進位置に配置した具体例。
`build_local_energy_cache` で全並進・全クラスターについて列挙される。

各インスタンスは以下を持つ：
- `atoms`：そのインスタンスに含まれるスーパーセル原子インデックスのリスト
- `coeff_flat`, `dims`, `strides`, `prefactor`：テンソル収縮に使うデータ
  （同じクラスター型の全インスタンスで共通）

### body / N-body cluster

クラスターが触れる**サイト数** `N = length(atoms)`。XML では `<SALC body="N">` 属性として直接記録される。

- `body = 1`：単一サイト項（例：単一イオン異方性）
- `body = 2`：2 体項（pair / bond、Heisenberg 結合など）
- `body = 3`：3 体項（triplet）
- `body ≥ 4`：高次

SCE は `body` および `l_max` に上限を置かない。

### uniform body / mixed body Hamiltonian

- **uniform body**: ハミルトニアン内の全 `ClusterInstance` が同じ `N` を持つ
  状態（例：`bcc_2x2x2`, `fege_2x2x2` はいずれも全 SALC が `body = 2`）。
  単純な恒等式

  ```
  Σ_i local_energy(i) = N · total_energy
  ```

  が成立する。
- **mixed body**: 異なる `N` の cluster が混在する状態（例：`ferh_4x4x4` は
  `body = 2` と `body = 3` を含む）。一般化された恒等式

  ```
  Σ_i local_energy(i) = Σ_inst length(inst.atoms) · E_inst
  ```

  が成立するが、単一の `N` で割って `total_energy` を復元することはできない。

この区別は内部整合テスト（`sum_local / N = total`）の前提として
`test/simple/test_simple_energy.jl`, `test/parity/test_parity_{bcc,fege}.jl`
で参照される。

---

## コード中の対応関係

| 概念 | コード中の変数・フィールド |
|---|---|
| 基本セルの原子数 | `h.base_n_atoms` |
| スーパーセルの原子数 | `h.n_atoms` |
| スーパーセル行列（正準） | `h.supercell_matrix`（`repeat` も常にこの M に変換して保持） |
| `repeat` 由来か（記録用） | `h.repeat`（直接 M 指定時は `(0,0,0)` センチネル） |
| 復元したプリミティブセル | `h.prim::PrimitiveCell`（`reshape_base` / `n_prim` / `n_trans`） |
| スーパーセル原子番号（cell-major） | `subl + n_prim·(cell_id − 1)` |
| クラスターインスタンスの列挙 | `_build_cluster_instances(h)`（un-fold, 幾何的） |
| 高速カーネルの per-atom テーブル | `_build_sai_table_cellmajor(...)` → `UnfoldSAITable` |
