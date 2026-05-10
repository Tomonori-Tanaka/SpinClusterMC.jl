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
