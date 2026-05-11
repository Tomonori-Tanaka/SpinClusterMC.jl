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
    spins::Vector{SVector{3,Float64}}
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

### 提案関数の差し替え可能化（同時に実施）

Sunny.jl の `LocalSampler(propose=propose_uniform)` パターン。Carlo アダプタ分離（`metropolis.jl` への
`sweep!` 切り出し）と同時に実施するのが自然。

```julia
mc.propose::Function  # (rng, sx, sy, sz) -> (sx', sy', sz')
```

---

## さらなるパフォーマンス改善候補（保留中）

`:tensor_template` カーネル（N=2 / N=3 SVector 特殊化済み）と `Vector{SVector{3,Float64}}` スピン
ストレージの実装後に残っている候補。優先度順。

### 候補A：`zlm_cache` レイアウト転置（最優先・低コスト）

**現状**: `Matrix{Float64}(n_atoms, ncols)`（`_alloc_zlm_cache` in `src/spin_utils.jl`）。
Julia は column-major のため、hot path の `zlm_cache[atom, col]` を内側 SIMD で col 方向に
varying させると **stride = n_atoms** の strided アクセスになる。

**問題のループ**: `_kernel3_chg` (src/template_energy.jl)
```julia
@simd for m_chg in 1:d_chg
    inner += coeff_flat[base_m1 + (m_chg - 1) * s_chg] *
             zlm_cache[changed_atom, chg_col_base + m_chg]   # ← n_atoms stride
end
```

**提案**: `(ncols, n_atoms)` に転置 → 内側ループが連続メモリアクセスに。

**影響箇所**:
- `_alloc_zlm_cache` (src/spin_utils.jl)
- `_kernel3_chg` / `_tensor_contract_template2_changed!` /
  `_tensor_contract_template_changed!` (src/template_energy.jl) のすべての `zlm_cache[...]` index
- `Carlo.sweep!` 内の zlm_row_buf save/restore (src/JPhiMagestyCarlo.jl line 1340, 1368)
- `_update_atom_zlm_cache!` (src/spin_utils.jl)

**期待効果**: 内側 SIMD ループ 2〜3×、3 体支配系で全体 1〜2%
**実装規模**: 小（indexing 反転のみ、テストは既存で十分）

### 候補B：`coeff_flat` の stride 別 dispatch（中規模・3 体系で 5〜10%）

**現状**: `_kernel3_chg` の内側ループ `coeff_flat[base + (m_chg-1)*s_chg]` は changed atom の
位置（sitepos）で stride が変わる：
- sitepos=1: s_chg=strides[1]=1（連続、SIMD OK）
- sitepos=2: s_chg=strides[2]=dims[1]=3（strided）
- sitepos=3: s_chg=strides[3]=dims[1]*dims[2]=9（更に strided）

3 体評価の 67% は sitepos=2 or 3 のため、その分 SIMD vectorization が崩れる。

**提案**: build 時に sitepos 別の coeff_flat（changed 軸を最下位ストライドに置いた並べ替え版）を
用意するか、sitepos=1 専用と sitepos>1 専用の kernel に分けて常に stride=1 を保証する。

**期待効果**: sitepos=2,3 ケースが SIMD 化、3 体支配系で 5〜10%
**実装規模**: 中（テンソル並べ替えロジック追加 + dispatch 変更）

### 候補C：`Union{Nothing,LocalEnergyTemplate}` 解消（低優先・< 1%）

**現状**: `mc.local_template::Union{Nothing,LocalEnergyTemplate}` (src/JPhiMagestyCarlo.jl:765)。
hot path で `tpl = mc.local_template::LocalEnergyTemplate` の typeassert が必要 (line 920)。

**提案**: 関数バリアで `_template_sweep!(mc)` / `_tensor_sweep!(mc)` を分けて、各々の中で
field type を確定させる。あるいは Carlo アダプタ分離と同時に struct を分ける。

**期待効果**: < 1%（タグチェック 1 命令）
**実装規模**: 小〜中（`Carlo.sweep!` の構造変更）

### 候補D：`zlm_row_buf` を `MVector` に（効果は小さい）

`zlm_row_buf::Vector{Float64}`（サイズ `(l_max+1)^2`、l_max=2 なら 9 要素）は毎スピン提案ごとに
書いて読む。`MVector{K,Float64}` にするとスタックに乗るが、サイズが実行時パラメータのため
パラメトリック型が必要。効果は小さい。

### 実装方針

候補 A → B → C の順を推奨。実装前に `profiler` エージェントで現状のボトルネックを実測し、
変更前後で `bccFe 2x2x2` と `ferh 1x1x1` のベンチマークを比較すること。

---

## プロファイル結果（2026-05-11, bcc_2x2x2 + 2x2x2 タイリング, 128原子）

`profiler` エージェントで `:tensor_template` パスを計測。条件：max_l=1、N=2 base instances=88、
T=0.02585 eV、spin_theta_max=0.5。

### Before / After 比較

「Before」は何も手を入れていない時点。「After」は以下 2 件を適用後：

1. `sweep!` 内の `related_instances = Int[]`（`:tensor_template` パスで未使用）を削除
2. Magesty.jl の buffered `Zₗₘ_unsafe(l, m, u, buf)` を採用。`JPhiSpinMC` に `zlm_dnpl_buf`
   フィールド（長さ `max_l+1`）を持たせ、`_update_atom_zlm_cache!` 経由で渡す

| 指標 | Before | After | 変化 |
|---|---|---|---|
| sweep（`:tensor_template`, GC 除く） | 57.9 μs | **51.1 μs** | **1.13×** |
| sweep（`:tensor`） | 69.1 μs | 62.0 μs | 1.11× |
| allocs/sweep（`:tensor_template`） | 1152 | **0** | 完全消失 |
| memory/sweep（`:tensor_template`） | 38.9 KB | **0 B** | 完全消失 |
| GC overhead | 3.1 μs (5.1%) | 0 | 消失 |
| `_update_atom_zlm_cache!`（128 calls/sweep） | 24.7 μs (43%) | **4.8 μs (9%)** | **5.1×** |
| `_update_atom_zlm_cache!`（per call） | ~66.7 ns | 37.5 ns | 1.78× |

### After の sweep 内訳（51.1 μs/sweep, allocs 0, GC 0）

| 処理 | 呼び出し回数/sweep | 時間 | 割合 |
|---|---|---|---|
| `_tensor_contract_template2_changed!`（テンソル収縮） | 2816 | **23.7 μs** | **46%** |
| `supercell_atom_index`（mod + 乗算） | 5632 | **14.1 μs** | **28%** |
| `_update_atom_zlm_cache!` | 128 | 4.8 μs | 9% |
| ループ・分岐・tile_coords 等の残差 | — | 6.2 μs | 12% |
| `_propose_spin_geodesic` | 128 | 1.6 μs | 3% |
| その他（zlm_row_buf コピー, Metropolis rand+exp） | — | 0.9 μs | 2% |

### 残る主要ボトルネックと候補との対応

- **収縮カーネル `_tensor_contract_template2_changed!`（46%）** ← 候補 A・B が対応。
  特に候補 A（`zlm_cache` レイアウト転置）は数値規約を変えない安全な改善で、
  内側 SIMD ループの stride を 1 にできるため再優先候補。
- **`supercell_atom_index` の mod + 乗算（28%）** ← 新規候補（下記「候補F」を追加）。
  数値結果に影響しないリファクタリングで、`repeat` が小さい本ベンチでは比較的大きな割合を
  占める。

候補 E（`Zₗₘ_unsafe` のバッファ事前確保）は完了。Magesty.jl 側で API が実装され、
SpinClusterMC 側で受け取り側を更新済み（commit pending）。

---

## 候補F：`supercell_atom_index` の mod 削減（新規, 28%, 数値不変）

**現状**: `_template_local_energy!` 内で 1 sweep あたり 5632 回 `supercell_atom_index(base_atom,
ti, tj, tk, base_n_atoms, repeat)` を呼び、その都度 `mod` と乗算でスーパーセル原子インデックスを
計算している。bcc_2x2x2 + 2x2x2 タイリング系では 14.1 μs/sweep（28%）を占める。

**提案**:
- (a) `repeat` を `Val{rep}` として伝搬し、`mod` をコンパイル時定数化する。`repeat` のバリエー
  ションは少数（典型は (2,2,2) や (4,4,4) など固定）なので、一般化と特殊化のバランスを取りやすい。
- (b) タイル座標 (ti,tj,tk) のループ自体を `_template_local_energy!` の外側に持ち上げ、
  事前計算したインデックステーブル（`Vector{Int}` of length `n_atoms × max_sites`）を引くだけに
  する。メモリは増えるが mod が消える。

**期待効果**: 14.1 μs → 数 μs（推定 sweep 全体 1.1〜1.2× 改善）
**実装規模**: 中。`_template_local_energy!` 周辺と関連ヘルパーの変更が必要。
**注意**: 数値結果は不変であるべきだが、インデックステーブル方式 (b) はタイリング規約と
密結合になるため `_foreach_translated_instance` との連動確認が必要（CLAUDE.md「連動箇所」）。
