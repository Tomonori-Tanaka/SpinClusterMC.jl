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
