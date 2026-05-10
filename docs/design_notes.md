# 設計メモ

保留中の設計判断や将来の実装方針を記録する。

---

## Carlo アダプタのアルゴリズム分離（保留中）

### 背景

現在 `JPhiMagestyCarlo.jl` には単スピン Metropolis 以外のアルゴリズム（Wolff クラスター更新など）を
追加したい場合、`Carlo.sweep!` 以外のボイラープレート（`init!`, `measure!`, チェックポイント, シリアライズ）
が全部重複することになる。

### 方針

struct の入れ子（`mc.base.spins`）は既存コードの変更量が大きいため採用しない。
**ヘルパー関数の共有**で実現する。

```
src/
  carlo_helpers.jl   ← アルゴリズム非依存な共通ロジック（引数渡しの純粋関数）
  carlo_mc.jl        ← JPhiSpinMC struct + constructor + init!/measure!/checkpoint/serialize
  metropolis.jl      ← Carlo.sweep! のみ（Metropolis アルゴリズム本体）
  wolff.jl（将来）   ← JPhiWolffMC <: AbstractMC + Carlo.sweep!
```

`carlo_helpers.jl` に切り出す関数：

| 関数 | 切り出し元 | 用途 |
|---|---|---|
| `_init_spins_from_params!` | `Carlo.init!` 内 | スピン初期化（random / 指定値） |
| `_compute_initial_energy` | `Carlo.init!` 内 | エネルギー初期値計算（両カーネル対応） |
| `_mc_measure!` | `Carlo.measure!` 内 | Energy・Magnetization の記録 |
| `_register_standard_evaluables` | `Carlo.register_evaluables` 内 | 比熱・感受率・Binder 比の登録 |

新アルゴリズム追加時のテンプレート：

```julia
# wolff.jl
mutable struct JPhiWolffMC <: AbstractMC
    T::Float64
    ham::SCEHamiltonian
    spins::Matrix{Float64}
    energy::Float64
    # ... 共通フィールド ...
    cluster_buf::Vector{Int}   # Wolff 固有
end

Carlo.init!(mc::JPhiWolffMC, ctx, params) = begin
    _init_spins_from_params!(mc.spins, params, ctx.rng, mc.ham.base_n_atoms)
    mc.energy = _compute_initial_energy(...)
end
Carlo.sweep!(mc::JPhiWolffMC, ctx) = ...   # Wolff アルゴリズム本体
Carlo.measure!(mc::JPhiWolffMC, ctx)       = _mc_measure!(ctx, mc.energy, mc.spins, mc.ham.n_atoms)
```

---

## Sunny.jl を参考にした実装案（保留中）

> 用語の対応：Sunny.jl の "unit cell"（conventional cell）= 本コードの「基本セル」、
> Sunny.jl の `dims` = 本コードの `repeat`。
> Sunny.jl の primitive cell（`primitive_cell()` で取得）= 本コードの「プリミティブセル」。
> 参照: [Sunny.jl GitHub](https://github.com/SunnySuite/Sunny.jl)、[`docs/terminology.md`](terminology.md)

### Sunny.jl の設計との比較

| 観点 | Sunny.jl | SpinClusterMC.jl（現状） |
|---|---|---|
| 相互作用テンプレート | 基本セル1個分のみ保存、全セルをオンザフライでループ | 全インスタンスを事前列挙（`_build_cluster_instances`） |
| ΔE 計算 | テンプレート + オンザフライでインデックス計算 | 事前列挙済みインスタンスを直接参照 |
| メモリ | O(n_base_instances) | O(n_atoms × n_base_instances) |
| 二重カウント防止 | `isculled` フラグ + ソート済みリストへの `break` | `_foreach_translated_instance` の `seen::Set` |

### 案1：BaseClusterInstance テンプレート

Sunny.jl と同様に基本セル1個分のテンプレートのみ保存し、`sweep!` でオンザフライにスーパーセルインデックスを計算する。

```julia
struct BaseClusterInstance
    base_atoms::Vector{Int}      # 基本セル内原子インデックス（jphi.xml の atoms= 属性）
    cbc::CoupledBasis_with_coefficient
    prefactor::Float64
    dims::Vector{Int}
    strides::Vector{Int}
    coeff_flat::Vector{Float64}
    Mf_size::Int
end

# 基本原子 b ごとに「どのテンプレートが b を含むか」+「相対タイルオフセット」を事前計算
struct RelatedBaseCluster
    inst_idx::Int
    tile_offset::NTuple{3,Int}   # actual_tile = (tile_i .+ tile_offset) .% repeat
end

struct LocalEnergyTemplate
    base_instances::Vector{BaseClusterInstance}
    related_by_base_atom::Vector{Vector{RelatedBaseCluster}}  # indexed by base_atom b
end
```

`sweep!` での ΔE 計算：

```julia
b      = ((i-1) % base_n_atoms) + 1     # 基本原子
tile_i = tile_of(i, base_n_atoms, repeat) # タイル座標

for rc in template.related_by_base_atom[b]
    inst = template.base_instances[rc.inst_idx]
    tile = mod.(tile_i .+ rc.tile_offset, repeat)
    atoms = [supercell_atom_index(map_sym[ba, t], tile..., base_n_atoms, repeat)
             for (ba, t) in zip(inst.base_atoms, ...)]
    e += inst.prefactor * _tensor_contract_cached(inst, zlm_cache, atoms)
end
```

**効果：**
- メモリが O(n_base_instances) に削減（スーパーセルサイズ非依存）
- `coeff_flat` がキャッシュに乗り続けるため大きいスーパーセルで有利
- テンソル収縮コスト自体は変わらない（支配項のため影響なし）

### 案2：提案関数の差し替え可能化

Sunny.jl の `LocalSampler(propose=propose_uniform)` パターン。
Carlo アダプタ分離（`metropolis.jl` への `sweep!` 切り出し）と同時に実施するのが自然。

```julia
mc.propose::Function  # (rng, sx, sy, sz) -> (sx', sy', sz')
```

### 実装時の注意

- `related_by_base_atom` の事前計算には `map_sym` と `_foreach_translated_instance` を使う
- タイルオフセットの符号・modulo の扱いが細かい。CLAUDE.md「連動箇所」参照
- 実装前に `benchmark_sce.jl` でベースラインを取り、変更後に同条件で比較すること

---

## StaticArrays による高速化（保留中）

### 背景

`_tensor_contract_template_changed!` が `sweep!` の支配的コストであり、
その内側ループで `inst.dims`, `inst.strides`, `inst.base_atoms`, `inst.tile_deltas`
を繰り返し参照している。これらはすべてヒープ上の `Vector` であり、要素数は N=2〜5 程度と小さい。

### 候補1：`BaseClusterInstance` を本体数 N でパラメトリック化（最優先）

`dims::Vector{Int}` / `strides::Vector{Int}` / `base_atoms::Vector{Int}` /
`tile_deltas::Vector{NTuple{3,Int}}` を `SVector{N,…}` にする。

効果：
- `for k in 1:N` ループがコンパイル時にアンロールされる（N=2の2体項が大多数の系で大きい）
- これら小配列がスタックに乗る（ヒープ参照のオーバーヘッド消滅）
- `@inbounds` なしでも境界チェックが消える

**実装上の問題点**：`BaseClusterInstance{N}` は型パラメータを持つため、
`Vector{BaseClusterInstance}` が抽象型のコンテナになり型不安定になる。
対策：
- `Union{BaseClusterInstance{2}, BaseClusterInstance{3}, BaseClusterInstance{4}, BaseClusterInstance{5}}`
  で dispatch する（N≤5 で場合分け）
- あるいは N=2 のみ特殊化し、残りは現状維持

### 候補2：`spins` のレイアウト変更（変更範囲が広い）

現在 `spins::Matrix{Float64}`（3×n_atoms）。列アクセスが `@view(mc.spins[:, atom])` でヒープ参照。
`Vector{SVector{3,Float64}}` に変えると `mc.spins[atom]` がスタック値になり、
zlm 計算の x/y/z バラシコストが消える。

**トレードオフ**：`sce_energy`, `_tile_base_spins!`, `coupled_cluster_energy` など
`spins` を受け取るすべての関数の型注釈を変える必要がある。変更範囲が広く、
候補1より優先度は低い。

### 候補3：`zlm_row_buf` を `MVector` に（効果は小さい）

`zlm_row_buf::Vector{Float64}`（サイズ `(l_max+1)^2`、l_max=2 なら 9 要素）は
毎スピン提案ごとに書いて読む。`MVector{K,Float64}` にするとスタックに乗るが、
サイズが実行時パラメータのためパラメトリック型が必要。効果は小さいと見込まれる。

### 実装方針

候補1 → 候補2 の順で検討する。
実装前に `profiler` エージェントでボトルネックを確認し、変更前後でベンチマークを比較すること。

---

## ClusterInstance のテンプレート化（保留中）

> 用語の定義は [`docs/terminology.md`](terminology.md) を参照。

### 背景

`_build_cluster_instances` は jphi.xml で定義されたクラスターをスーパーセル内の全並進に展開し、
全インスタンスを `ClusterInstance` として列挙する。2×2×2 タイリングでは同じテンソルデータ
（`coeff_flat`・`dims`・`strides`・`prefactor`）を持つインスタンスが 8 コピー生成され、
メモリが 8 倍、`LocalEnergyCache` の構築時間も 8 倍になる。

ベンチマーク結果（`test/ferh_4x4x4/jphi.xml`）:
- 1×1×1: instances=840,256、sweep≈1,573 ms
- 2×2×2: instances=6,722,048（8倍）、sweep≈13,972 ms（8.9倍）

スケーリングは O(n_atoms) で正常だが、大きいスーパーセルでキャッシュ圧迫が問題になりうる。

### 提案する設計

「テンプレート + オンザフライ並進」:

jphi.xml の各クラスターは基本セル内の原子インデックスで定義されており（`atoms="i j ..."`）、
プリミティブセル中の原子を必ず1つ含む。この情報を1コピーだけ保持し、
sweep! 時にスーパーセル原子インデックスをオンザフライで計算する。

```
BaseClusterInstance {
    base_atoms   ← 基本セル内の原子インデックス（jphi.xml の atoms 属性）
    coeff_flat   ← 1コピーのみ（全並進で共有）
    dims, strides, prefactor
}

sweep! で atom i の局所エネルギーを計算するとき：
  b = base_atom(i),  tile = tile_of(i)
  → map_sym で base_atoms を並進 → supercell_atom_index で実インデックスを計算
```

### トレードオフ

| | 現状（展開済み） | 提案（テンプレート） |
|---|---|---|
| メモリ | O(n_atoms × n_base_instances) | O(n_base_instances) |
| キャッシュ効率 | coeff_flat が n_tiles 箇所に散在 | 同じバッファを n_tiles 回再利用 |
| sweep! のインデックス計算 | ゼロ（展開済み） | 毎回 supercell_atom_index を呼ぶ |
| related_instances の構築 | 単純（原子インデックスで引ける） | タイルをまたぐ探索が必要 |

大きいスーパーセルほどテンプレート方式が有利（coeff_flat がキャッシュに乗り続ける）。
小さい系では現状の方がシンプルで速い可能性がある。

### 実装時の注意

CLAUDE.md「連動箇所」参照。タイリングロジックは以下 3 箇所を同期して変更する必要がある：
- `_foreach_translated_instance`
- `_build_cluster_instances`
- `coupled_cluster_energy`（リファレンスパス、独立実装）

実装前に `benchmark_sce.jl` / `benchmark_sce_reference.jl` でベースラインを計測し、
変更後に同条件で比較すること。
