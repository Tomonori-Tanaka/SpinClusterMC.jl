# Design — Simple Zlm Workspace

> 関連 spec: [requirements.md](requirements.md) / [tasklist.md](tasklist.md)

## 全体方針

`src/simple/` を 3 層 API に整理する。

```
Layer 1: Bufferless reference   (現行 API、教育・テスト用)
   total_energy(h, spins)
   local_energy(h, spins, i)
   delta_local_energy(h, spins, i, S_new)
   gradient(h, spins, i)

Layer 2: Workspace-aware (pure buffer)
   total_energy(h, spins, ws)
   local_energy(h, spins, i, ws)
   delta_local_energy(h, spins, i, S_new, ws)
   gradient(h, spins, i, ws)

Layer 3: Stateful cache (MC hot path)
   sync_workspace!(ws, h, spins)
   delta_local_energy_cached!(h, spins, i, S_new, ws)
   commit_workspace_column!(ws, i, S_new)
```

レイヤ間の関係:

- **Layer 1 は Layer 2 の薄いラッパ**。`f(h, spins) = f(h, spins, SpinClusterWorkspace(h))`。
  数値結果は完全に同一。alloc は内部の workspace 1 個分だけ増えるが、Layer 1 は
  cold path 専用なので問題にならない。
- **Layer 2 は毎回 `sync_workspace!` 相当の再計算を内部で行う**。precondition なし、
  呼び出し側は workspace の中身を気にしなくてよい。alloc を削るだけが目的。
- **Layer 3 は precondition あり**。「`ws.zlm` が `spins` と同期している」ことを呼び
  出し側が保証する。代わりに「変わった 1 列だけ更新」する。MC hot path 専用。

## 型定義

新規ファイル `src/simple/workspace.jl`。

```julia
"""
    SpinClusterWorkspace(h::SpinClusterHamiltonian)

Preallocated scratch space for repeated energy/gradient calls on the same
`SpinClusterHamiltonian`. Holds the SpheriCart evaluator plus a `Zlm` cache
sized `K × n_atoms` (where `K = (h.max_l + 1)^2`) and per-column save
buffers used by the stateful cache API.

# Fields

- `sph`: `SphericalHarmonics(h.max_l)`. Persistent SH evaluator.
- `zlm`: `Matrix{Float64}` of size `(K, n_atoms)`. Tesseral Z_l^m for every
  atom in the supercell. **In the stateful cache API its contents must
  match the spin configuration passed to `sync_workspace!` / mutated in
  place via `commit_workspace_column!`.** In the pure-buffer API the
  contents are scratch — every call overwrites it from `spins`.
- `zlm_col_buf`: `Vector{Float64}` of length `K`. Scratch buffer used by
  `delta_local_energy_cached!` to save the column that is about to be
  overwritten so it can be restored on reject.
- `dzlm_i`: `Vector{SVector{3, Float64}}` of length `K`. Scratch buffer
  used by `gradient(..., ws)` for the per-atom gradient column.

# Invariants

- `size(ws.zlm) == (K, n_atoms)` where `K = (h.max_l + 1)^2`.
- `ws` is bound to one Hamiltonian's `max_l` and `n_atoms`; mixing across
  Hamiltonians of different shape is unsupported and not checked.
- In the stateful cache API: between calls, `ws.zlm` represents the Zlm of
  `spins` (the matrix the caller is iterating over). The library never
  reads `spins` to detect drift — the caller is responsible.

# Lifetime

Typical use is one workspace per `SCEMC` (constructed once in the MC
constructor, lives for the whole run). One-shot users of `total_energy`
can omit it; the bufferless overload allocates a workspace internally.
"""
mutable struct SpinClusterWorkspace
    sph::SphericalHarmonics
    zlm::Matrix{Float64}
    zlm_col_buf::Vector{Float64}
    dzlm_i::Vector{SVector{3, Float64}}
end

function SpinClusterWorkspace(h::SpinClusterHamiltonian)
    K = (h.max_l + 1)^2
    return SpinClusterWorkspace(
        SphericalHarmonics(h.max_l),
        Matrix{Float64}(undef, K, h.n_atoms),
        Vector{Float64}(undef, K),
        Vector{SVector{3, Float64}}(undef, K),
    )
end
```

レイアウト確認:

- Simple 版は `zlm[k, atom]` (列 = 原子)。Optimized 側は `zlm_cache[atom, k]`
  (行 = 原子) と逆。本 spec では **Simple の現行レイアウト `zlm[k, atom]` を維持**。
  混乱を避けるため `_zlm_index(l, m)` ヘルパーは現行のまま流用。
- レイアウト変更は別問題で本 spec では扱わない。

## API 設計

### Layer 1 (bufferless reference)

```julia
"""
    total_energy(h, spins) -> Float64

Total SCE energy ``E = Σ_{inst} E_inst`` for the supercell spin
configuration `spins`. ...

# Notes

This bufferless overload is **not used by the Monte Carlo hot path**; it
allocates a fresh `SpinClusterWorkspace` on every call and is intended as
an educational / testing reference. Production MC code (`SCEMC.sweep!`)
calls the workspace-aware overload `total_energy(h, spins, ws)` (or, for
single-spin proposals, `delta_local_energy_cached!`).
"""
total_energy(h, spins) = total_energy(h, spins, SpinClusterWorkspace(h))
```

`local_energy` / `delta_local_energy` / `gradient` も同形式で書く。Notes 節
の "not used by the Monte Carlo hot path" の文言は 4 つすべてに入れる。

### Layer 2 (pure buffer)

```julia
"""
    total_energy(h, spins, ws) -> Float64

Workspace-aware variant of `total_energy`. Overwrites `ws.zlm` from
`spins`, then sums every cluster's contribution. Returns the same value
as the bufferless overload but does not allocate on the heap (apart from
whatever `_instance_energy` allocates internally; see `design_notes.md`).

# Mirrors

`JPhiMagestyCarlo._energy_from_instances_cached` (which assumes
`zlm_cache` is already in sync; this variant ensures sync inside).

# Preconditions

- `size(spins) == (3, h.n_atoms)`.
- `ws` was constructed from the same Hamiltonian shape (`max_l`, `n_atoms`).

# Postconditions

- Returns `Σ_inst E_inst`.
- `ws.zlm` is overwritten with the Zlm of `spins`. (Side effect: callers
  who relied on the previous contents must re-sync.)
"""
function total_energy(h, spins, ws::SpinClusterWorkspace)::Float64
    _validate_spin_matrix(h, spins)
    _compute_zlm_all!(ws.zlm, ws.sph, spins, h.n_atoms)
    E = 0.0
    for inst in h.instances
        E += _instance_energy(inst, ws.zlm, h.cg_table)
    end
    return E
end
```

`_compute_zlm_all!` は現行の `_compute_zlm_all` を in-place 版に変更したもの:

```julia
function _compute_zlm_all!(zlm, sph, spins, n_atoms)
    K = size(zlm, 1)
    for a in 1:n_atoms
        z = compute(sph, _spin_svector(spins, a))
        @inbounds for k in 1:K
            zlm[k, a] = z[k]
        end
    end
    return zlm
end
```

`local_energy` / `delta_local_energy` / `gradient` も同様に Layer 2 化。

`delta_local_energy(h, spins, i, S_new, ws)` の中身:

```julia
function delta_local_energy(h, spins, i, S_new, ws)::Float64
    # Same algorithm as the bufferless version, but using ws.zlm as the
    # scratch matrix instead of allocating one.
    _compute_zlm_all!(ws.zlm, ws.sph, spins, h.n_atoms)
    # Save column i, write new, sum delta, restore column.
    K = size(ws.zlm, 1)
    @inbounds for k in 1:K
        ws.zlm_col_buf[k] = ws.zlm[k, i]
    end
    z_new = compute(ws.sph, SVector{3, Float64}(S_new[1], S_new[2], S_new[3]))
    @inbounds for k in 1:K
        ws.zlm[k, i] = z_new[k]
    end
    delta = 0.0
    for idx in h.atom_to_instance_indices[i]
        inst = h.instances[idx]
        E_new = _instance_energy(inst, ws.zlm, h.cg_table)
        # restore one column temporarily to compute E_old; this is the
        # simplest implementation. A swap-free version is possible via
        # _instance_energy_with_column but only useful in Layer 3.
        @inbounds for k in 1:K; ws.zlm[k, i] = ws.zlm_col_buf[k]; end
        E_old = _instance_energy(inst, ws.zlm, h.cg_table)
        @inbounds for k in 1:K; ws.zlm[k, i] = z_new[k]; end
        delta += E_new - E_old
    end
    # restore for caller hygiene
    @inbounds for k in 1:K
        ws.zlm[k, i] = ws.zlm_col_buf[k]
    end
    return delta
end
```

Layer 2 では `_instance_energy_with_column` を使わない (アルゴリズムの読みやすさ
優先)。Layer 3 で導入する。

### Layer 3 (stateful cache)

```julia
"""
    sync_workspace!(ws, h, spins) -> ws

Rebuild `ws.zlm` from scratch against `spins`. Call once before a run of
`delta_local_energy_cached!` invocations, and again whenever `spins` is
mutated outside `ws` (for example, after `_renorm_and_drift_check!`).

# Mirrors

`JPhiMagestyCarlo._rebuild_zlm_cache!` (`compute!(mc.zlm_cache, mc.sph, mc.spins)`).

# Postconditions

- `ws.zlm[:, a] == Z_l^m(spins[:, a])` for every atom `a`.
"""
function sync_workspace!(ws, h, spins)
    _compute_zlm_all!(ws.zlm, ws.sph, spins, h.n_atoms)
    return ws
end

"""
    delta_local_energy_cached!(h, spins, i, S_new, ws) -> Float64

Change in local energy at site `i` when its spin is replaced by `S_new`,
computed by updating only column `i` of `ws.zlm` in place.

# Mirrors

`JPhiMagestyCarlo` Metropolis trial body in `_jphi_metropolis_sweep!`
(save row, swap to s_new, compute E_new, decide, restore on reject).

# Preconditions

- `ws.zlm` is in sync with `spins` (caller invariant).
- `1 ≤ i ≤ h.n_atoms`, `length(S_new) == 3`.

# Postconditions

- Returns `local_energy_after - local_energy_before` for the proposal
  S_i → S_new.
- `ws.zlm[:, i]` holds the **new** column (computed from `S_new`).
- `ws.zlm_col_buf` holds the **old** column (so the caller can restore
  via `restore_workspace_column!(ws, i)` on reject).
- All other columns of `ws.zlm` are unchanged.

The caller decides what to do next:

- **Accept**: leave `ws.zlm` as-is; the new column is already installed.
  Update `spins[:, i] .= S_new` so the sync invariant is preserved.
  (Helper: `commit_workspace_column!(ws, i, S_new)` is a no-op that
  exists only for naming symmetry; nothing in `ws` needs to change.)
- **Reject**: call `restore_workspace_column!(ws, i)` to copy the
  saved old column back into `ws.zlm[:, i]`. `spins` is unchanged.
"""
function delta_local_energy_cached!(h, spins, i, S_new, ws)::Float64
    K = size(ws.zlm, 1)
    @inbounds for k in 1:K
        ws.zlm_col_buf[k] = ws.zlm[k, i]
    end
    z_new = compute(ws.sph, SVector{3, Float64}(S_new[1], S_new[2], S_new[3]))
    delta = 0.0
    for idx in h.atom_to_instance_indices[i]
        inst = h.instances[idx]
        # E_old uses ws.zlm as-is (column i is still old).
        E_old = _instance_energy(inst, ws.zlm, h.cg_table)
        # E_new substitutes column i with z_new on the fly; ws.zlm is not
        # mutated inside this helper.
        E_new = _instance_energy_with_column(inst, ws.zlm, i, z_new, h.cg_table)
        delta += E_new - E_old
    end
    # Install the new column so the postcondition holds.
    @inbounds for k in 1:K
        ws.zlm[k, i] = z_new[k]
    end
    return delta
end

"""
    commit_workspace_column!(ws, i, S_new)

Accept-side helper. Currently a no-op because `delta_local_energy_cached!`
already installs the new column in `ws.zlm` before returning; this
function exists for symmetry with `restore_workspace_column!` and so that
the call site reads `delta → if accept commit else restore`.
"""
commit_workspace_column!(ws, i, S_new) = ws

"""
    restore_workspace_column!(ws, i) -> ws

Reject-side helper. Copies `ws.zlm_col_buf` (the column saved by the
most recent `delta_local_energy_cached!`) back into `ws.zlm[:, i]`.
"""
function restore_workspace_column!(ws, i)
    K = size(ws.zlm, 1)
    @inbounds for k in 1:K
        ws.zlm[k, i] = ws.zlm_col_buf[k]
    end
    return ws
end
```

`_instance_energy_with_column` の signature (新規、内部用):

```julia
# Internal helper: E_inst evaluated as if zlm column `swap_col` were
# replaced by `swap_col_data`, without mutating zlm. Used by the stateful
# cache API to compute E_old and E_new from a single zlm without
# round-tripping the column through memory.
function _instance_energy_with_column(
        inst::ClusterInstance,
        zlm::AbstractMatrix{Float64},
        swap_col::Int,
        swap_col_data::AbstractVector{Float64},
        cg_table::CGTable,
)::Float64
    # Same loop as _instance_energy, but the lookup
    #   zlm[_zlm_index(l, m), atoms[k]]
    # is replaced by
    #   atoms[k] == swap_col ? swap_col_data[_zlm_index(l, m)] :
    #                          zlm[_zlm_index(l, m), atoms[k]]
    ...
end
```

数値結果は `_instance_energy` と完全一致 (`swap_col` 列を別 vector から読むだけで
算術は同一)。

## `SCEMC` への組み込み

```julia
mutable struct SCEMC <: Carlo.AbstractMC
    h::SpinClusterHamiltonian
    external::Union{Nothing, ExternalTerm}
    moments::MomentModel
    spins::Matrix{Float64}
    T::Float64
    theta_max::Float64
    renorm_every::Int
    energy::Float64
    sweep_count::Int

    # New field: persistent workspace bound to h.
    ws::SpinClusterWorkspace
end
```

`init!` の変更:

```julia
function Carlo.init!(mc::SCEMC, ctx, params)
    # ... (existing initial_spins setup) ...
    sync_workspace!(mc.ws, mc.h, mc.spins)   # ← new
    mc.energy = _full_energy(mc)             # uses Layer 2 internally
    mc.sweep_count = 0
    return nothing
end
```

`_full_energy` も Layer 2 を使う:

```julia
function _full_energy(mc::SCEMC)::Float64
    return total_energy(mc.h, mc.spins, mc.ws) +
           _external_total_energy(mc.external, mc.spins)
end
```

`metropolis_sweep!`:

```julia
function metropolis_sweep!(mc::SCEMC, ctx)
    n = mc.h.n_atoms
    rng = ctx.rng
    @inbounds for _ in 1:n
        i = rand(rng, 1:n)
        S_old = SVector{3, Float64}(mc.spins[1, i], mc.spins[2, i], mc.spins[3, i])
        S_new = _propose_spin_geodesic(rng, S_old, mc.theta_max)

        ΔE_sce = delta_local_energy_cached!(mc.h, mc.spins, i, S_new, mc.ws)
        ΔE = ΔE_sce + _external_delta_local(mc.external, mc.spins, i, S_new)

        if ΔE ≤ 0.0 || rand(rng) < exp(-ΔE / mc.T)
            mc.spins[1, i] = S_new[1]
            mc.spins[2, i] = S_new[2]
            mc.spins[3, i] = S_new[3]
            mc.energy += ΔE
            commit_workspace_column!(mc.ws, i, S_new)   # currently no-op
        else
            restore_workspace_column!(mc.ws, i)
        end
    end
    return nothing
end
```

`_renorm_and_drift_check!` の変更:

```julia
function _renorm_and_drift_check!(mc::SCEMC)
    # ... renormalize spins in place ...
    sync_workspace!(mc.ws, mc.h, mc.spins)    # ← new: re-sync after spin mutation
    E_full = _full_energy(mc)
    # ... existing drift check ...
end
```

不変条件: `mc.spins` を直接書き換えた直後は必ず `sync_workspace!`。これは
`init!` と `_renorm_and_drift_check!` の 2 箇所だけ。

## ドキュメント規約

### 1. Bufferless 版の `# Notes` 節

Layer 1 の 4 関数 (`total_energy(h, spins)` 等) の docstring に必須:

```
# Notes

This bufferless overload is not used by the Monte Carlo hot path; it
allocates a fresh `SpinClusterWorkspace` on every call and is intended
as an educational / testing reference. Production MC code goes through
the workspace-aware overload `<fname>(h, spins, ..., ws)` (or, for
single-spin proposals, `delta_local_energy_cached!`).
```

### 2. Layer 2/3 関数の `# Preconditions` / `# Postconditions`

Layer 3 ではマスト。Layer 2 でも contract を明示する (`ws.zlm` を上書きする
副作用は呼び出し側にとって surprise になりうる)。

### 3. `# Mirrors` 行

Layer 2/3 の各関数に、対応する `JPhiMagestyCarlo` 内の関数名を 1 行で
リンクする。

```
# Mirrors

`JPhiMagestyCarlo._update_atom_zlm_cache!` (column-in-place update on the
optimized side).
```

### 4. `src/simple/workspace.jl` モジュール先頭

10〜20 行の design rationale を英語で:

- なぜ Workspace を引数で受け渡すのか (副作用を型で見せる)
- なぜ `SpinClusterHamiltonian` に持たせなかったのか (純データ維持)
- Layer 1 / 2 / 3 の関係
- Optimized 側 (`JPhiMagestyCarlo`) との対応

## パフォーマンス見込み

| パス | Before (M10 baseline) | After (本 spec) | 削減源 |
|---|---|---|---|
| `SCEMC.metropolis_sweep!` (bcc, 1 sweep) | 計測予定 | alloc / call は数十オーダーに | (A) の解消 |
| `total_energy(h, spins)` (fege) | 84M allocs / 1.76 GiB | inner-loop alloc が残るので 84M に近い | (B) の解消は Zlm 行列 1 個分のみ |
| `delta_local_energy_cached!(h, spins, i, S_new, ws)` (fege) | (該当パスなし、これが新規) | 0 alloc を目標 | (A) + (B) の解消 |

`total_energy` の絶対 alloc は本 spec では追求しない (inner-loop が支配。別 spec)。

## リスクと緩和

| リスク | 緩和 |
|---|---|
| `mc.spins` を直接書き換えた箇所で `sync_workspace!` を呼び忘れる | sync 必要箇所を 2 つ (init / renorm) に閉じ込める。テストで drift check を有効に保つ |
| `delta_local_energy_cached!` の accept 後に `commit_workspace_column!` 呼び忘れ | commit は現実装では no-op。将来別 layout に変えた時に効くよう、呼び出しは残す慣習にする |
| `_instance_energy_with_column` が `_instance_energy` と数値乖離する | 専用 unit test を追加。同じ zlm + 列 i に対し `_instance_energy(inst, zlm)` と `_instance_energy_with_column(inst, zlm, i, zlm[:, i])` が bit-exact 一致 |
| Workspace 引数版と無引数版で算術順序が変わって parity が壊れる | Layer 1 を Layer 2 の単純 wrapper として実装すれば順序は同じ。順序保存は実装時に念押し |
| SciML Style + JET が新コードに警告 | M10 までと同様、書いたら `format` + `make test` (JET 含む) で確認 |

## テスト計画

新規テストファイル:

- `test/simple/test_simple_workspace.jl`: Workspace 構築・形状・型のスモーク
- `test/simple/test_simple_workspace_layer2.jl`: Layer 1 vs Layer 2 の bit-exact 一致
- `test/simple/test_simple_workspace_cached.jl`: `sync_workspace!` 後の Layer 3
  vs Layer 1 ΔE の一致 (`rtol = 1e-14`)、accept/reject 経路の不変条件検証
- 既存 `test/parity/test_parity_*.jl` は無修正のまま全 pass を維持

詳細は [tasklist.md](tasklist.md) の各マイルストーンに割り当てる。
