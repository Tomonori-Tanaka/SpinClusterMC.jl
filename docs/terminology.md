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
