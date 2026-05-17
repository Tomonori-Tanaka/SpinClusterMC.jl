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
| タイリング数 | `h.repeat` |
| 基本セルの並進対称性 | `h.map_sym`（`base_n_atoms × n_trans`） |
| スーパーセル原子インデックス変換 | `supercell_atom_index(base_atom, ti, tj, tk, base_n_atoms, repeat)` |
| クラスターインスタンスの列挙 | `_build_cluster_instances(h)` |
| 並進の重複除去 | `_foreach_translated_instance(f, h, cbc)` |
