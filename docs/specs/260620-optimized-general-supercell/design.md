# Design — Optimized 版の一般スーパーセル対応 (Phase 1)

> 関連 spec: [requirements.md](requirements.md) / [tasklist.md](tasklist.md)

## 1. 全体方針

Simple と同じ分離を optimized でも採る:

- **対角 `repeat` → legacy パス** (`_build_supercell_geometry` +
  `_foreach_translated_instance` + 既存カーネル)。**一切変えない**。
- **`supercell_matrix = M` → 新 primitive パス** (`:tensor` カーネル限定)。
  付番は primitive cell-major (Simple と一致)、エネルギーは Simple と parity。

整数線形代数・primitive 抽出・自己重なり判定は Simple と完全に同じなので
**共有モジュール `SupercellCommon`** に括り出し、両モジュールから使う。

## 2. 共有モジュール `src/supercell_common.jl`

新規 `module SupercellCommon`。`SpinClusterMC.jl` で **両サブモジュールより前に**
`include`。`JPhiMagestyCarlo` / `Simple` は `using ..SupercellCommon: ...`。

移管する関数 (現 `src/simple/supercell.jl` から移動、純粋・型非依存):

```
# 整数線形代数 (3×3)
_int_det3, _adjugate3, _col_hermite, _wrap_offset_into_supercell
# primitive 幾何
struct PrimitiveCell  (lattice, pos_frac, n_prim, base_to_prim, prim_to_base, reshape_base)
extract_primitive(lattice, pos_frac, map_sym, n_trans) -> PrimitiveCell
_shortest_independent3
# タイリング基盤
_cluster_base_stabilizer(atoms, map_sym, n_trans) -> Int
_enumerate_cells(M, adjM, detM) -> (cell_index, cells_by_id)
_supercell_from_repeat(reshape_base, repeat) -> SMatrix
# クラスタ幾何ヘルパ (新規, 両モジュール共用)
_cluster_offsets(atoms, prim) -> (pivot_subl, site_subl::Vector{Int}, site_delta::Vector{NTuple{3,Int}})
```

- **`extract_primitive` を生配列引数版に refactor** (現 Simple 版は `SystemData`
  を取る)。`periodicity` は現実装で未使用なので **引数に含めない**。
  Simple は `SystemData` の (`lattice`, `pos_frac`, `map_sym`, `n_trans`)、
  optimized は `SystemXMLInfo` の同名フィールドを渡す (両者ともこの4つを持つ)。
- `_cluster_offsets`: `atoms` (base 原子) → pivot 相対の (副格子, primitive
  offset) 列。Simple の `build_templates` と optimized のタイリングが共用。
- スタイル: 共有ファイルは SciML Style (新規ファイルなので)。

## 3. Simple 側の refactor (挙動不変)

- `src/simple/supercell.jl`: 上記の共有関数を **削除し `using
  ..SupercellCommon` で取り込む**。残すのは Simple 固有の `ClusterTemplate`
  型・`build_templates` (内部で `_cluster_offsets` / `_cluster_base_stabilizer`
  を共有から呼ぶ)・`_generate_instances_matrix` (Simple `ClusterInstance` を
  emit)。
- `src/simple/types.jl` / `mc.jl`: シンボル参照を共有モジュール経由に。
  **方針確定**: Simple 内で `using ..SupercellCommon: _int_det3, _adjugate3,
  _col_hermite, _wrap_offset_into_supercell, PrimitiveCell, extract_primitive,
  _cluster_base_stabilizer, _enumerate_cells, _supercell_from_repeat,
  _cluster_offsets` と **明示 import** する。これで既存テストの
  `Simple._int_det3` 等の qualified アクセスはそのまま通る (using で取り込んだ
  名前は `Simple.foo` で解決可能)。re-export はしない。
- **検証**: 既存 Simple テスト (M1-M6 含む) が無改変 pass = 挙動不変の担保。

## 4. Optimized 側の一般 M パス (`:tensor` カーネル)

### 4.1 instance 生成 (`_build_cluster_instances` の M 版)

新 `_build_cluster_instances_matrix(salc_list::Vector{Vector{CoupledBasis_with_coefficient}},
jphi, map_sym, n_trans, prim, M)` (optimized 専用):

各 `(salc group s, cbc)` について:
1. `s_base = _cluster_base_stabilizer(cbc.atoms, map_sym, n_trans)`。
   `mod(cbc.multiplicity, s_base) == 0` を assert。
   `eff_mult = cbc.multiplicity ÷ s_base`。
2. `(pivot_subl, site_subl, site_delta) = _cluster_offsets(cbc.atoms, prim)`。
3. `_enumerate_cells(M, ...)` の各 cell で全サイトを wrap して
   supercell atom index (`subl + n_prim*(cell-1)`) に解決。**同一 sorted-atoms
   に落ちる配置は 1 instance にまとめ、重なり数を `eff_mult` に乗じて加算**
   (Simple `_generate_instances_matrix` と同じ規約)。
4. optimized の `ClusterInstance` を生成 (prefactor =
   `jphi[s] * eff_mult_total * scaling`、`scaling = _cluster_scaling(N)`、
   coeff_flat 等は既存 `_build_cluster_instances` と同じ流用)。

`_tensor_contract_instance*` などの低レベルカーネルは **不変** (instance を
渡す側だけ変える)。

### 4.2 SCEHamiltonian / load_sce_hamiltonian

```
load_sce_hamiltonian(xml; repeat=(1,1,1), supercell_matrix=nothing, jphi_threshold=0.0)
```
- `supercell_matrix === nothing` → legacy (現状のまま)。
- else → `repeat==(1,1,1)` を要求 (二重指定エラー)、`M=SMatrix{3,3,Int}`、
  `det(M)!=0` を要求、`extract_primitive` →
  `_build_cluster_instances_matrix` で instances 構築。geometry
  (lattice/pos_frac) は primitive×M で構築 (`_build_supercell_geometry` の
  M 版、または reference path 用に最小限)。
- `SCEHamiltonian` に **`supercell_matrix::Union{Nothing,Matrix{Int}}`
  フィールドを追加** (legacy は `nothing`)。`repeat` フィールドは legacy 用に
  保持 (M パスでは `(0,0,0)` 番兵)。構築箇所 (load + MPI deserialize) を更新。
- **`init!` の `_tile_base_spins!` 注意**: M パスは primitive cell-major 付番
  なので base-cell サイズの `initial_spins` をタイリングすると副格子が崩れる。
  Simple の `init!` ガードと同様、M パス + base-cell サイズ `initial_spins` は
  **明示エラー** (full supercell サイズ配列か乱数初期化を要求)。

### 4.3 カーネル選択 (JPhiSpinMC)

- `params[:supercell_matrix]` を読み、M パスでは **kernel を `:tensor` に強制**。
- `params[:energy_kernel] == :tensor_template` かつ `supercell_matrix` 指定 →
  **明示 `ArgumentError`** (「Phase 2 未実装。:tensor を使うか repeat を使え」)。
- sweep! の `use_template` 分岐: M パスでは `:tensor` 側
  (`_related_instances_by_atom` + `_tensor_contract_instance_cached_changed!`)
  を通る。`build_local_energy_template` は **呼ばない**。

### 4.4 キャッシュ & MPI

- `_HAM_CACHE` / `_ECACHE_CACHE` / `_DERIVED_CACHE` のキーに `M` を追加。
  既存キー `(xml, repeat, thr[, ...])` に **第4要素 `Union{Nothing,
  NTuple{9,Int}}`** (flatten した `M`、legacy は `nothing`) を足す。repeat
  パスのキー値・挙動は実質不変 (nothing が付くだけ)。
- `_mpi_build_ham_and_cache` / `_get_or_build_derived` のシグネチャに `M` を
  追加。
- `JPhiSpinMC` に `supercell_matrix::Union{Nothing,Matrix{Int}}` フィールド +
  `Serialization.serialize/deserialize` に追加。deserialize 側で
  `load_sce_hamiltonian(xml; supercell_matrix=M, ...)` を呼んで再構築。
- `register_evaluables` の `n_atoms` 取得も M 対応。**ローカル key 構築
  (`key = (xml, rep, thr)`) にも `M` を足す** (M を変えて同一プロセスで再呼び
  出ししたとき古い `n_atoms` を返さないように)。

### 4.5 total energy (parity 用)

- **注意**: `sce_energy(h, spins)` は `coupled_cluster_energy(...; repeat =
  h.repeat)` のタイルループ実装なので、M パス (`repeat=(0,0,0)`) では `0:(-1)`
  ループになり **0 を返す (壊れる)**。
- 対応: `sce_energy` に分岐を追加し、M パス
  (`h.supercell_matrix !== nothing`) では **instance 総和**
  (`_energy_from_instances(instances, spins)`) を返す。`instances` は
  `_build_cluster_instances_matrix` の結果 (または `LocalEnergyCache` 経由)。
  legacy 対角パスは従来通り `coupled_cluster_energy` 総和 (不変)。
- `coupled_cluster_energy` リファレンス自体は **対角のまま** (M 非対応)。M パスの
  reference total は instance 総和を正とする。

## 5. テスト

- `test/parity/test_parity_supercell_matrix.jl` (新規): bcc/fege で複数 `M`
  (base 倍数 `reshape_base*diag`、非倍数 `diag(3,2,2)`、非対角) について
  **optimized `sce_energy` ≈ Simple matrix path total** を ferro + random 配置で
  検証 (両者 primitive cell-major で同付番なので random も直接比較可)。
- `test/optimized/test_supercell_matrix.jl` (新規): load/エラー (3×3・det0・
  二重指定・`:tensor_template`+M)、MPI serialize round-trip (`M` 保持)、
  sweep! が走る。
- 既存 optimized/parity/JET テスト無改変 pass (legacy 不変の担保)。
- 重い fege/ferh は `"slow" in ARGS` 節へ。`runtests.jl` に include 追加。

## 6. 連動箇所 (CLAUDE.md)

- **タイリングロジック**: legacy の `_foreach_translated_instance` /
  `coupled_cluster_energy` は不変。M パスは独立した
  `_build_cluster_instances_matrix` を新設 (共有 `_enumerate_cells`/wrap 利用)。
- **2 パス整合性**: M パスは `:tensor` のみ。`:tensor_template` (デフォルト) は
  対角専用のまま。M で template を要求したらエラー (silent な不一致を防ぐ)。
- **observables per atom 規約**: `measure!` / `register_evaluables` は不変
  (n_atoms 取得経路だけ M 対応)。
- **低レベルカーネル**: `_tensor_contract_instance*` は不変。

## 7. Phase 2 (本 spec 外, future work)

`:tensor_template` 高速パスの一般 M 対応: `_foreach_base_instance` /
`_build_sai_table_n` / `_tile_coords` / `_template_local_energy!` を
primitive cell-major (cell_id + subl) に作り直す。別 spec で扱う。
`docs/design_notes.md` に future-work として記録。
