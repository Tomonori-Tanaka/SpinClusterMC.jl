# Design — General supercell (任意 3×3 整数行列 M)

> 関連 spec: [requirements.md](requirements.md) / [tasklist.md](tasklist.md)

## 1. 用語と座標系

- **base cell**: `jphi.xml` が記述するセル (= Magesty 訓練スーパーセル)。
  `SystemData.lattice` (列 = `[a1 a2 a3]`, Å)、`pos_frac` (3×base_n)、
  `map_sym` (base_n × n_trans)、`n_trans` を持つ。
- **primitive cell**: base cell 内の並進対称性 (`map_sym`) が定める最小周期。
  primitive 格子 `Lp` (3×3, 列 = primitive ベクトル)、`n_prim = base_n/n_trans`
  副格子。base = primitive × `reshape_base` (`reshape_base` は整数 3×3,
  `det = n_trans`)。
- **supercell**: primitive を一般整数行列 `M` (3×3) でタイリングしたセル。
  primitive 格子に対し `L_super = Lp * M`。セル数 `N_cells = |det(M)|`、
  原子数 `n_atoms = n_prim * N_cells`。
- **cell offset**: primitive cell 単位の整数ベクトル `c ∈ ℤ³`。
- **後方互換**: `repeat=(n1,n2,n3)` は base cell の対角タイリング。primitive
  単位では `M = reshape_base * diagm([n1,n2,n3])` に対応する。

座標規約は CLAUDE.md 準拠 (spins `3×n_atoms`、列 = 原子、`spins[:,i]` は単位
方向ベクトル)。本 spec はジオメトリ (atom 付番) のみを扱い、これらは不変。

## 2. モジュール構成

新規ファイル `src/simple/supercell.jl` に以下を集約 (`SpinClusterMC` の
`include` 順は `types.jl` の前)。すべて内部関数 (leading `_`) または
非 export 型。

```
# --- 整数線形代数 (3×3 限定) ---
_int_det3(M::SMatrix{3,3,Int})::Int
_adjugate3(M::SMatrix{3,3,Int})::SMatrix{3,3,Int}   # M * adj(M) = det(M) I
_col_hermite(M::SMatrix{3,3,Int}) -> (H, U)         # M = H*U, H 下三角>0, U unimodular
_wrap_offset_into_supercell(c::NTuple{3,Int}, M, adjM, detM)::NTuple{3,Int}

# --- primitive 抽出 ---
struct PrimitiveCell
    lattice::Matrix{Float64}                         # Lp (3×3, 列 = prim ベクトル)
    pos_frac::Matrix{Float64}                        # 3×n_prim, prim セル内 frac
    n_prim::Int
    base_to_prim::Vector{Tuple{Int, NTuple{3,Int}}}  # base atom → (副格子 s, prim cell offset)
    reshape_base::Matrix{Int}                        # base = prim * reshape_base, det = n_trans
end
extract_primitive(sys::SystemData)::PrimitiveCell

# --- クラスタの primitive テンプレート化 ---
struct ClusterTemplate
    pivot_subl::Int
    site_subl::Vector{Int}              # 各サイトの副格子
    site_delta::Vector{NTuple{3,Int}}   # 各サイトの pivot 相対 cell offset (site_delta[1] = (0,0,0))
    ls::Vector{Int}
    Lf::Int
    Lseq::Vector{Int}
    weights::Vector{Float64}
    J::Float64
    multiplicity::Int
end
build_templates(salcs, jphi, sys, prim; jphi_threshold)::Vector{ClusterTemplate}

# --- 一般 M タイリング ---
_generate_instances_matrix(templates, prim, M; compat_repeat=nothing)::Vector{ClusterInstance}
```

`energy.jl` / `cg.jl` / 観測量は変更しない。

## 3. アルゴリズム詳細

### 3.1 整数線形代数

3×3 限定で閉形式・整数演算のみ (浮動小数 `LinearAlgebra.det` のロバスト性
問題を避ける)。

- `_int_det3`: サラスの公式を `Int` で。
- `_adjugate3`: 余因子行列の転置 (整数)。`M^{-1} = adj(M)/det(M)`。
- `_col_hermite`: 列基本変形 (整数 gcd・列入替・列加算) で下三角化。
  `H[i,i] > 0`、`H[i,j] = 0 (i<j)`、`0 ≤ H[i,j] < H[i,i] (i>j)`。
  `U` は unimodular (`|det U| = 1`)、`M = H*U`。
  Sunny `MatInt.col_hermite` (`Crystal.jl:561`) の 3×3 移植。
- `_wrap_offset_into_supercell(c, M, adjM, detM)`: cell offset `c` を `M` の
  列が張る格子で mod する。`f = adjM * c` (整数ベクトル, `= detM * M^{-1} c`)、
  各成分を `mod(f_i, detM)` ではなく、`M` 格子の代表に落とすため
  `frac_i = f_i / detM` の小数部を取って `c - M * floor.(frac)` を整数で計算。
  結果は §3.3 の `cell_index` に登録された代表 offset の1つ。
  Sunny `position_to_atom_and_offset` + `wrap_to_unit_cell` 相当。
- **`det(M) < 0` の扱い**: `_col_hermite` は `H[i,i] > 0` を満たすので
  `det(H) > 0`、よって `det(U) = sign(det(M))` (左手系 `M` では `det(U) = -1`)。
  HNF 自体は符号に依らず正常に構成できる。`_wrap_offset_into_supercell` の
  `frac_i = f_i / detM` は **Float64 除算**で計算し (`f_i`, `detM` は整数)、
  `floor` 後に整数へ戻す。これで `detM` の符号に依らず正しい代表 offset を返す。

### 3.2 primitive 抽出 (`extract_primitive`)

入力 `sys::SystemData`。

0. **前提 assert**: `rep0 = map_sym[1,1]` は恒等並進での atom 1 の像。
   `map_sym[1,1] == 1` (より堅牢には
   `pos_frac[:,map_sym[1,1]] ≈ pos_frac[:,1]`) を assert する。Magesty 生成の
   全 fixture で成立するが、仕様として保証されていないため明示チェックする。
1. **並進ベクトル収集**: 各 `t ∈ 1:n_trans` で
   `df = pos_frac[:, map_sym[1,t]] - pos_frac[:, rep0]`、`df .-= round.(df)`
   (最小像)、`v = lattice * df` を候補に push。さらに **`periodicity` に依らず
   格子ベクトル3本 `lattice[:,j]` (j=1,2,3) を常に候補に追加** (Magesty
   `_sunny_primitive` と同じ。3D 全周期材料では n_trans 由来候補が独立3本を
   含むので実害なし、開放方向のフォールバックも兼ねる)。
2. **最短独立3本**: `sort!(cands; by=norm)`、零ベクトル除去後に貪欲選択
   (1本目 = 最短非零、2本目 = 1本目と非共線、3本目 = 2本と非共面、
   閾値 `tol = 1e-8 * 平均長`)。`Lp = hcat(b1,b2,b3)`。`det(Lp) < 0` なら
   3本目を反転して右手系化 (`_sunny_primitive` SunnyExport.jl:283-285 と同じ)。
3. **副格子分類**: `Lpi = inv(Lp)`。各 base atom `a` を
   `g = Lpi * (lattice * pos_frac[:,a])` で primitive 座標へ。
   `frac = mod.(g, 1)` が一致する base atom 群を1副格子に (tol `1e-5`)。
   各群の代表 (最小 base index) を `prim_rep[s]`、
   `prim_frac[:,s] = mod.(Lpi*(lattice*pos_frac[:,prim_rep[s]]), 1)`。
4. **base_to_prim**: 各 base atom `a` に `(s, Δ)` を割当。`s` は frac 一致で
   決定、`Δ = round.(Int, g - prim_frac[:,s])`。
5. **reshape_base**: `reshape_base = round.(Int, Lpi * lattice)`。
   整数性を assert (`Lpi*lattice ≈ 整数`)。
6. **整合性 assert**:
   - `n_prim = base_n / n_trans` (割り切れること)。
   - 各副格子に属する base atom 数 = `n_trans`。
   - `|det(reshape_base)| == n_trans`。
   失敗時は明示的 `ErrorException` (XML が想定外の対称構造)。

### 3.3 クラスタの primitive テンプレート化 (`build_templates`)

各 `(salc, basis)` を **1 個の** `ClusterTemplate` に変換する (並進ループは
不要; 並進が生む base-cell コピーはタイリングが再生成する)。

1. `jphi_threshold` で `abs(J) < thr` の SALC を skip (既存と同じ短絡;
   `thr == 0.0` で全 keep)。
2. **offset**: site `k` を `base_to_prim[basis.atoms[k]] = (s_k, δ_k)` に変換し、
   pivot (site 1) 相対に `Δ_k = δ_k - δ_1` (`Δ_1 = (0,0,0)`)。これは listed
   atoms 間の実際の primitive 変位 (最小像でなくてよい; wrap が吸収)。
3. **自己重なり補正 (重要)**: XML の `multiplicity` は base cell での自己重なり
   (等距離 ±Δ 像の畳み込み) を encode している。半周期=面上ペア (2Δ≡0) は
   `multiplicity ≥ 2`。一般 supercell では ±Δ が別原子に分かれるため、
   **un-fold した有効値**
   ```
   s_base = _cluster_base_stabilizer(atoms, map_sym, n_trans)
          = count(t -> sort(map_sym[atoms,t]) == sort(atoms), 1:n_trans)
   effective_mult = basis.multiplicity ÷ s_base   (割り切れることを assert)
   ```
   を `ClusterTemplate.multiplicity` に格納する。これで任意 `M` で base cell と
   同じエネルギー密度になる (Magesty 規約 `E = Σ contract·multiplicity` に一致)。
   - 根拠: Magesty は `1/stabilizer` 除算をせず multiplicity に等距離像数を
     畳み込む (`Fitting.jl:778-833`, `Clusters.jl:438-448`,
     `SALCBases.jl:1069-1082`)。bcc/fege で検証済み (`s_base`, multiplicity とも
     面上=2/内部=1、effective=1)。

### 3.4 一般 M タイリング (`_generate_instances_matrix`)

入力: `templates`, `prim`, `M::SMatrix{3,3,Int}`。

1. `detM = _int_det3(M)`、`detM != 0` を要求。`N_cells = abs(detM)`、
   `adjM = _adjugate3(M)`。
2. **cell 列挙**: `_enumerate_cells` が HNF `H` の対角ボックス
   `[0,H11)×[0,H22)×[0,H33)` を `_wrap_offset_into_supercell` で正規化し、
   `cell_index::Dict{offset→cell_id}` と `cells_by_id::Vector{offset}`
   (`cell_id ∈ 1:N_cells`) を返す (三角行列の補題で coset 代表を網羅)。
3. **付番**: `super_index(cell_id, subl) = subl + n_prim*(cell_id-1)`、
   `n_atoms = n_prim * N_cells` (純 primitive cell-major)。
4. **instance 生成 + 自己畳み込み accumulate**: 各 template について、全 cell
   `c0` で全サイト `k` を
   ```
   wrapped = _wrap_offset_into_supercell(c0 .+ site_delta[k], M, adjM, detM)
   atom_k  = site_subl[k] + n_prim*(cell_index[wrapped]-1)
   ```
   で解く。**同一 sorted-atoms に落ちる配置は 1 instance にまとめ、
   multiplicity を加算** (`effective_mult × 重なり数`)。これは `M` レベルの
   追加の自己畳み込み (小さいセルでクラスタが自身に重なる場合) を Magesty の
   sorted-multiset 規約通りに処理する。`order` で初出順を保ち決定的に emit。

## 4. 後方互換 (diagonal `repeat` は legacy をそのまま使う)

§3 の primitive タイリングは **legacy とは異なる atom 付番** (`subl + n_prim·
(cell_id-1)`) を使うが、ferro 等価検証で **エネルギーは legacy と一致** する
(M4 で全 fixture 確認)。既存テスト・`init_spins`・PT は legacy の base-cell
付番に依存するため、**bit-exact 互換は付番を変えない方法で達成する**:

- **`repeat=(n1,n2,n3)` (対角 base 倍数) → 既存 `_generate_instances` を
  そのまま使う** (コード不変 ⇒ bit-exact 自明)。
- **`supercell_matrix=M` (一般 `M`) → §3 の primitive タイリング** (新付番、
  エネルギーは legacy と整合)。

この分離により、既存の挙動・テスト・PT serialization は**完全に無改変**で、
新機能は純粋な追加となる。primitive 付番を旧 base-cell 付番に一致させる
互換層 (旧 design の §4) は **不要** になった。`prim_to_base` 逆引きは
当面使わないが `PrimitiveCell` に保持しておく (将来の `:initial_spins`
一般化等に有用)。

## 5. API / 配線

### 5.1 `SpinClusterHamiltonian` (types.jl)

```
SpinClusterHamiltonian(xml_path;
    repeat=(1,1,1),
    supercell_matrix=nothing,        # AbstractMatrix{<:Integer} 3×3, or nothing
    jphi_threshold=0.0)
```

- 解決規則:
  - `supercell_matrix !== nothing` かつ `repeat != (1,1,1)` → `ArgumentError`
    (二重指定)。
  - `supercell_matrix === nothing` (= `repeat` 指定) → **legacy パス**:
    `_generate_instances(...; repeat)` をそのまま使う (bit-exact 互換)。
  - `supercell_matrix !== nothing` → **primitive パス**:
    `M = SMatrix{3,3,Int}(supercell_matrix)` で
    `extract_primitive` → `build_templates` → `_generate_instances_matrix`。
    `det(M) == 0` で `ArgumentError`。
- 構造体フィールド: 既存の `repeat::NTuple{3,Int}` は legacy パスで従来通り使う。
  新フィールド `supercell_matrix::Matrix{Int}` (3×3) を追加 (legacy パス時は
  番兵 `diagm(repeat)` か `zeros` 等で記録)。`n_atoms` は legacy では
  `base_n * prod(repeat)`、primitive では `n_prim * |det(M)|`。`base_n_atoms`
  は従来通り。

### 5.2 `SCEMC` (mc.jl)

- param 解決に `:supercell_matrix` を追加 (`_params_supercell_matrix` 的
  ヘルパ)。`SCEMC` が受け付けるのは `:repeat` (対角・従来通り) と
  `:supercell_matrix` (一般 `M`) の2つのみ。両者の二重指定は `ArgumentError`。
  - optimized 側の `:supercell` (対角 alias) は **`SCEMC` には追加しない**
    (本 spec のスコープ外)。
- `SCEMC` 構造体に `supercell_matrix::Matrix{Int}` を持たせ、`init!` /
  `register_evaluables` での `n_atoms` 取得を新 Hamiltonian から行う。
- `init_spins` / `_tile_base_matrix` (spin_proposal.jl): M=diag では付番不変の
  ため無改変で動く。非対角 `M` の `:initial_spins` タイリングは
  `base_to_prim` 経由の一般化が必要 (M5 で対応; 当面は非対角時に
  `:initial_spins` 未対応なら明示エラーでも可 — M5 で判断)。

## 6. 整数線形代数の依存

- HNF / 整数 det / adjugate は既存依存 (`LinearAlgebra`, `StaticArrays`) に
  無い。3×3 限定で**自前実装** (`Project.toml` 変更なし)。
- Sunny の `MatInt` は standalone でなく依存追加は過剰。
- `SMatrix{3,3,Int}` を使い割当を避ける (Simple はリファレンスだが、無駄な
  Dict/alloc は避ける)。

## 7. 検証戦略

### M=diag (parity 維持)
- 既存 `test/parity/test_parity_{bcc,fege,ferh}.jl` が無改変で pass
  (互換層で付番不変)。
- 追加: `supercell_matrix = reshape_base*diag(2,2,2)` を渡した結果が
  `repeat=(2,2,2)` と instance 集合・total/local/ΔE で完全一致。

### 非対角 M (Simple 単独)
1. **対角等価**: `M` と unimodular `U` で `M' = M*U` が対角になるケースで、
   ferro / ランダム配置の `total_energy` が一致 (U による付番置換を許容)。
2. **並進不変性**: spins を 1 primitive cell シフトしても `total_energy` 不変
   (§3.1 wrap の正しさの強い検証)。
3. **スケーリング**: ferro 配置で `total_energy ∝ |det(M)|`
   (既存 runtests.jl の線形スケール則と同型)。
4. (任意・重) Magesty `SunnyExport` で同 `jphi.xml` を primitive 展開 →
   Sunny で同 `M` reshape し pair-only SALC の energy を独立比較。

### テスト配置
- `test/simple/test_simple_supercell.jl` (新規) を `runtests.jl` の simple
  include 群へ。重い非対角 ferh は `if "slow" in ARGS` 節へ。
- `make test` / `make test-slow` で実行。JET 静的解析も通す。

## 8. 連動箇所 (CLAUDE.md 準拠)

- 本 spec が触る連動箇所は **atom 付番生成のみ**。`energy.jl` の縮約・
  観測量 (`measure!` / `register_evaluables`) / `cg.jl` / `Φᵥ` 定義は不変。
- Optimized 側 (`_foreach_translated_instance` / `coupled_cluster_energy` /
  `template_energy.jl`) は据え置き。Simple と optimized の parity は M=diag
  ケースでのみ成立 (互換層が保証)。非対角 M は Simple 専用。
