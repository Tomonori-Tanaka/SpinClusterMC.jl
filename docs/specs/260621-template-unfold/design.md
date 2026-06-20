# Design — Phase 2: template kernel un-fold + `repeat` 統一

> 関連 spec: [requirements.md](requirements.md) / [tasklist.md](tasklist.md)

## 1. 方針

`:tensor` パス (Phase 1, `_build_cluster_instances_matrix`) が既に un-fold で
正しい。Phase 2 は同じ un-fold 幾何を**高速 template カーネル**に実装し、
`repeat` を含めて全経路を un-fold に統一する。

統一後の唯一の supercell 指定は **整数行列 M (primitive 単位)**:
- `supercell_matrix = M` → そのまま。
- `repeat = (n1,n2,n3)` → `M = reshape_base * diag(n)` (糖衣)。
- 内部はすべて `prim`(PrimitiveCell) + `M` + 共有 `SupercellCommon` ヘルパ。

## 2. template カーネルの un-fold 化 (`src/template_energy.jl`)

### 2.1 BaseClusterInstance の作り直し

現状は base 原子インデックス + tile_delta:
```
BaseClusterInstance{,2,3}: base_atoms::Vector{Int}, tile_deltas::Vector{NTuple{3,Int}}, ...
```
→ primitive 幾何に:
```
pivot_subl::Int
site_subl::Vector{Int}            # 各サイトの primitive 副格子
site_delta::Vector{NTuple{3,Int}} # pivot 相対の primitive cell offset (site_delta[pivot]= (0,0,0))
ls / Lseq / coeff_flat / dims / strides / Mf_size / prefactor   # 既存どおり
```
prefactor は `js * eff_mult * scaling` (`eff_mult = multiplicity ÷ s_base`)。
これらは共有 `_cluster_offsets(cbc.atoms, prim)` と
`_cluster_base_stabilizer(cbc.atoms, map_sym, n_trans)` で得る。

### 2.2 関連テーブル: base 原子 → primitive 副格子

`LocalEnergyTemplate` の 3 フィールド
`related_by_base_atom` / `related2_by_base_atom` / `related3_by_base_atom`
(各 長さ base_n) を
`related_by_subl` / `related2_by_subl` / `related3_by_subl`
(各 長さ n_prim) に置き換える (N=2/3/≥4 でそれぞれ独立)。

**重要 (pivot_k は参加サイトで可変)**: 原子 `i` の局所エネルギーには、`i` が
**どのサイトとして参加するか**に関わらず `i` に触れる全クラスタが必要。よって
`related_by_subl[subl]` には、**各 instance の各サイト `k` について `site_subl[k]
== subl` なら `(inst_idx, k)` を登録**する (既存 `related_by_base_atom` が base
原子 `b` を含む全 instance を `rc.pivot_k`=その factor 位置とともに列挙するのと
同じ意味)。`RelatedBaseCluster{inst_idx, pivot_k}` は流用 (`pivot_k` = 参加サイト
`k`、テンプレートの pivot=`k=1` とは別概念)。

テンプレートの幾何 pivot は常に `site=1` (`_cluster_offsets` が `atoms[1]` 固定、
`site_delta[1]=(0,0,0)`)。`related_by_subl` の索引はそれと独立に「参加サイトの
副格子」で行う。

### 2.3 SAI テーブル: cell-major precompute

`_build_sai_table_n` を作り直す。supercell 原子 `i` を
`(cell_id, subl) = ((i-1) ÷ n_prim + 1, (i-1) % n_prim + 1)` に分解。
各 `(cell_id, subl)` と各 `rc=(inst_idx, pivot_k) ∈ related_by_subl[subl]` に
ついて、参加サイト `pivot_k` を cell_id に置き、全サイト k' の supercell 原子:
```
abs_off = cells_by_id[cell_id] .+ (site_delta[k'] - site_delta[pivot_k])
w       = _wrap_offset_into_supercell(abs_off, M, adjM, detM)
sai_k'  = site_subl[k'] + n_prim * (cell_index[w] - 1)
```
を flat 配列に格納 (`pivot_k` 相対なので `sai_{pivot_k} == i` になる)。
`offsets` のレイアウトは従来どおりフラット `i = subl + n_prim*(cell_id-1)`
インデックス (長さ `n_atoms+1`)。`_tile_coords` / `supercell_atom_index` は廃止。
`cells_by_id` / `cell_index` は `_enumerate_cells(M, adjM, detM)` から
(template に `adjM`/`detM` とともに保持し N≥4 の on-the-fly wrap で再利用)。
`coeff_flat` は既存どおり `objectid(cbc)` keying で cbc 間共有。

### 2.4 `_template_local_energy!` (JPhiMagestyCarlo.jl)

`i → (cell_id, subl)` 分解 → `related{2,3}_by_subl[subl]` を走査 →
N=2/3 は precomputed SAI を読む (既存 `_tensor_contract_template{2,3}_changed!`
は不変)。N≥4 は on-the-fly に上記 wrap で SAI を計算 (`_tensor_contract_
template_changed!` 不変、`atoms_buf` に書く側だけ変更)。

### 2.5 build_local_energy_template

`h.prim` / `h.supercell_matrix`(or repeat→M) を入力に、各 (salc, cbc) で
`_cluster_offsets` + `_cluster_base_stabilizer` から BaseClusterInstance を作り、
SAI テーブルを `_build_sai_table_n` で precompute。`LocalEnergyTemplate` に
`prim` / `M`(or adjM/detM/cell_index/cells_by_id) を保持。

## 3. `repeat` の un-fold 統一

- `load_sce_hamiltonian` / `Simple.SpinClusterHamiltonian`: `repeat` 指定時に
  `prim = extract_primitive(...)`, `M = _supercell_from_repeat(prim.reshape_base,
  repeat)` として **un-fold パスに合流**。`supercell_matrix` 指定と同じ実装に。
- これにより `SCEHamiltonian` は常に `prim` / `supercell_matrix`(=M) を持つ
  (legacy の `repeat` フィールドは記録用に残すか撤去)。
- `:tensor_template` も `:tensor` も un-fold M で動く → **両カーネルとも一般 M で
  選択可能** (Phase 1 の「M では :tensor 強制」「:tensor_template+M はエラー」を
  解除)。

### 原子番号付けレイアウトの変更 (破壊的)

`repeat` を M に変換すると、supercell 原子の列順が変わる:
- folded 旧経路: **tile-major** `i = base_atom + base_n*(ti + n1*tj + n1*n2*tk)`
  (`supercell_atom_index`)。
- un-fold 新経路: **cell-major** `i = subl + n_prim*(cell_id-1)`
  (`cell_index` / `_enumerate_cells`)。

`mc.spins` の列順が `repeat>1` で変わるため、列順に依存する以下を更新する
(open decision 2 の base-cell タイリング廃止と連動):
- `src/simple/spin_proposal.jl` の `_tile_base_matrix`
  (コメント "tile-major order" 前提) を撤去。
- tile-major を前提にした既存テスト
  (`test/bcc_2x2x2`: "initial_spins tiling" / "cross-tile",
  `test/simple/test_simple_spin_proposal.jl`: Matrix tiling) を
  un-fold cell-major 期待値に書き換えるか撤去 (§6 / tasklist P2-M5)。

`repeat=(1,1,1)` は tile-major ≡ cell-major (cell 1 個) なので列順不変。

### folded コード撤去 (open decision 1 に従う)

撤去候補 (file:line は実装時に確定):
- `JPhiMagestyCarlo`: `supercell_atom_index`, `_foreach_translated_instance`,
  `coupled_cluster_energy`, folded `_build_supercell_geometry`(M 版に一本化),
  folded `_build_cluster_instances`(→ matrix 版に一本化)。
- `template_energy`: `_tile_coords`, `_foreach_base_instance`(folded)。
- `Simple`: folded `_generate_instances`, `_supercell_atom_index`。
- `sce_energy`: M 分岐のみ残す (instance 総和)。`coupled_cluster_energy` 依存を除去。
  - **注意 (init! コスト)**: `:tensor_template` の `init!` は `sce_energy(mc.ham,
    mc.spins)` でエネルギー初期化する。撤去後これは `_build_cluster_instances`
    (M 版) を都度呼ぶ O(n_instances)。**許容**する (init! は 1 回/構築)。
    過大なら template から直接 total を計算する補助を後で足す。
- `register_evaluables` / `JPhiSpinMC` constructor: `repeat` を M に変換する箇所と
  キャッシュキー (`_parse_repeat_param` 経由) も M ベースに合わせて更新する
  (§5)。

## 4. `init_spins` / `_tile_base_spins!` (open decision 2)

primitive cell-major では base-cell tiling が崩れる。案:
- (a) base-cell `initial_spins` タイリングを廃止し、`:random` か full
  `3×n_atoms` 配列のみ許可 (matrix パスの既存ガードと統一)。← 推奨
- (b) primitive 副格子ベースのタイリングに作り直す。
撤去/変更は `src/spin_utils.jl`(optimized) / `src/simple/spin_proposal.jl`。

注: `repeat` を M に変換すると常に `mc.supercell_matrix !== nothing` になるため、
optimized の既存 `init!` ガード (`initial_spins` + `supercell_matrix` → エラー)
が自動的に全ケースに効く。案 (a) なら追加実装はほぼ不要 (Simple 側も同様の
ガードを `repeat` 経路に拡張するだけ)。

## 5. キャッシュ / MPI

- `repeat` も M に変換されるので、キャッシュキーは `_scm_key(M)` に一本化
  (repeat 由来でも M で keying)。`_HAM_CACHE`/`_ECACHE_CACHE`/`_DERIVED_CACHE`/
  `register_evaluables`/serialize を M ベースに整理。
- serialize は `supercell_matrix`(M) を保持 (repeat は M に変換済みなので冗長)。
- **互換性 (open decision 4)**: `repeat` を M 変換し `mc.repeat`/`mc.supercell_
  matrix` の格納が変わるため、**Phase 2 の serialize は Phase 1 チェックポイントと
  非互換**になる。serialize は「同一バージョン内のみ」前提 (既存方針) なので許容と
  するが、`mc.repeat` を撤去するか記録用に残すかを deserialize とペアで確定する。

## 6. 2 パス整合性 & テスト (CLAUDE.md 連動箇所)

- **`:tensor_template` ≡ `:tensor`**: 一般 M・random 配置で総エネルギー一致を
  新テストで担保 (両カーネルとも un-fold)。`init!` の両カーネル energy 初期化、
  `sweep!` の ΔE 計算が一致すること。
- **既存テスト更新**:
  - `test/bcc_2x2x2/test_bcc_2x2x2.jl`: repeat=(2,2,2) の非共線エネルギー期待値を
    un-fold 値に更新。"reference path agrees with fast path" は両 un-fold で再確認。
    ferro/extensive/cross-tile 系は再検証 (ferro は不変)。
  - `test/runtests.jl`: `supercell_atom_index` 直接テストを撤去。
  - `test/ferh|fege`: (2,1,1) load テストは n_atoms 不変なので構造チェックは pass、
    energy 値があれば更新。
  - parity (bcc/fege/ferh, 主に (1,1,1)) は不変。
- `test/parity/test_parity_supercell_matrix.jl`: `:tensor_template` も比較対象に
  追加。"intended divergence" テストは **撤去**(repeat も un-fold になり folded
  比較が消える)。
- `test/optimized/test_supercell_matrix.jl`: `:tensor_template`+M がエラーでなく
  動くことに変更。

## 7. マイルストーン (tasklist 参照)

P2-M1 (template un-fold 構築) → P2-M2 (sweep + 2 パス整合) → P2-M3
(repeat 統一 + folded 撤去) → P2-M4 (init_spins / cache / MPI) →
P2-M5 (テスト全更新) → P2-M6 (docs + レビュー)。各段階で `make test` 維持。
