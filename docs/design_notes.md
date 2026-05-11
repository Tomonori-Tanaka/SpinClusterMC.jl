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

### Before / After 比較（4 段階）

| 段階 | 内容 |
|---|---|
| Before | 何も手を入れていない時点 |
| 中間1 | (1) `related_instances = Int[]` 削除 (2) Magesty buffered `Zₗₘ_unsafe(l, m, u, buf)` 採用 |
| 中間2 | (3) SpheriCart.jl 採用（parametric `JPhiSpinMC{S<:SphericalHarmonics}`）— bit-exact 一致を確認の上 |
| After | (4) SAI フラットテーブル化（候補F (b)）— N=2/N=3 ホットパスの `supercell_atom_index` + `_tile_coords` を `tpl.sai{2,3}_flat` lookup に置換 |

| 指標 | Before | 中間1 | 中間2 (SpheriCart) | After (SAI table) | 累積変化 |
|---|---|---|---|---|---|
| sweep（`:tensor_template`, GC 除く） | 57.9 μs | 51.1 μs | 45.2 μs | **26.0 μs** | **2.23×** |
| allocs/sweep | 1152 | 0 | 0 | **0** | 完全消失 |
| memory/sweep | 38.9 KB | 0 B | 0 B | **0 B** | 完全消失 |
| `_update_atom_zlm_cache!` per call | ~66.7 ns | 37.5 ns | 2.0 ns | **2.5 ns** | **27×** |
| SAI + `_tile_coords` 寄与 | 28% | 28% | 41.5% | **~1.3%** | 実質除去 |
| Zlm 寄与 | 43% | 9% | 0.8% | **1.2%** | 実質除去 |

### After の sweep 内訳（26.0 μs/sweep, allocs 0, GC 0, 2026-05-11）

| 処理 | μs/sweep | % total |
|---|---|---|
| `_template_local_energy!` 全体（2 パス、ループオーバヘッド含む） | **17.1** | **65.6%** |
| &emsp;└ `_tensor_contract_template2_changed!`（収縮、推定 6.7 ns × 11 × 128 × 2） | ~18.8 | 〜支配 |
| &emsp;└ SAI テーブル lookup（read-only） | 0.34 | 1.3% |
| `_propose_spin_geodesic` | 0.96 | 3.7% |
| Metropolis（`exp` + `rand`） | 0.99 | 3.8% |
| `_update_atom_zlm_cache!`（SpheriCart Zlm） | 0.32 | 1.2% |
| `zlm_row_buf` save+restore | 0.27 | 1.0% |
| 残差（loop/rng/分岐予測など） | ~6.4 | 24.6% |

### 残る主要ボトルネックと候補との対応

- **収縮カーネル `_tensor_contract_template2_changed!`**（推定 ~18.8 μs, 主要） ← 候補 A（zlm_cache 転置）と候補 B（coeff_flat stride dispatch）が対応
- **残差** ~6.4 μs（loop/rng/分岐）— 現状の sweep 構造で削るには Metropolis の内側ループそのものを見直す必要

完了済み：
- 候補 E: Magesty buffered `Zₗₘ_unsafe` 採用
- SpheriCart 採用（Zlm hot path 実質除去）
- 候補 F (b): SAI フラットテーブル化（SAI hot path 実質除去）— 想定 1.1〜1.2× を上回り **1.74×** 改善

---

## 候補F：完了（2026-05-11, 1.74× sweep 改善）

**実装**: `LocalEnergyTemplate` に `sai2_flat::Vector{Int}` + `sai2_offsets::Vector{Int}` と
N=3 用ペアを追加。`build_local_energy_template` 末尾の `_build_sai_table_n` ヘルパーで、
全 `(i, rc, k)` の組み合わせを 1 度だけ計算してフラット配列に詰める。
`_template_local_energy!` の N=2 と N=3 ループ内の `_tile_coords` + `supercell_atom_index` を
`sai{2,3}_flat[sai{2,3}_offsets[i] + N*(rc_idx-1) + k - 1]` 1 命令の lookup に置換。

**測定結果**: SAI + `_tile_coords` が 19.6 μs → ~0.34 μs（57× 高速化）。
sweep 全体は 45.2 μs → 26.0 μs（**1.74×**）。想定 1.1〜1.2× を上回ったのは、SAI 除去だけでなく
`_tile_coords` 呼びとそれに付随する `rc.pivot_k` 分岐や中間変数アクセスも一緒に消えたため。

**N≥4 path**: 事前計算テーブルなし、on-the-fly のまま。現行 2 つのテスト問題
（bcc_2x2x2, ferh_4x4x4）はどちらも N≥4 インスタンス 0 なので影響なし。`_tile_coords` 呼びは
`related_other` が空のとき skip する分岐を入れて 0 コストにした。

**メモリ・init コスト（実測, 2026-05-11）**:

| 問題 | repeat | n_atoms | sai2 / sai3 | 合計 mem | init 時間 |
|---|---|---|---|---|---|
| bcc_2x2x2 | (1,1,1) | 16 | 352 / 0 ints | 2.8 KB | <0.01 s |
| ferh_4x4x4 | (1,1,1) | 128 | 32 K / 7.49 M ints | 57.4 MB | 0.84 s |

ferh の 57 MB は許容範囲（test 用 / 開発時の絶対量）。`repeat=(1,1,1)` のため理論上は
`base_atoms[k]` をそのまま使えば 0 メモリで済むが、現実装は repeat に依らず一律で
テーブルを作る。将来 repeat=(1,1,1) 用の degenerate path を追加すれば 57 MB は消せる。

bcc_2x2x2 の `repeat` フィールドは XML 上は (2,2,2) 設定の派生だが、test 用の
`jphi.xml` は repeat=(1,1,1) 版を保持しているため init は瞬時。本ドキュメントの sweep
ベンチは MC 構築時に `repeat = (2, 2, 2)` を渡した結果（n_atoms=128）。
