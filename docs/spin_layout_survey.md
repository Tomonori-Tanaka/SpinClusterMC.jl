# スピン配列レイアウト調査（simple版 設計の参考資料）

`src/simple/` 参照実装で `spins` をどの型で持つかの判断材料として、既存の古典スピン MC パッケージのスピン保持方式と更新パスを調査した記録。
（調査日: 2026-05-12）

design_notes.md 側の `src/simple/` 設計セクションから参照される位置づけ。重複を避けるため、結論の詳細だけ design_notes.md に書き、本ファイルは「なぜその結論か」のエビデンスを置く。

---

## 用語: AoS と SoA

メモリ上にデータをどう並べるかの2つのパターン。

### AoS (Array of Structures)

1要素 = 1セットのデータを連続に並べる。

```
spins = [S₁, S₂, S₃, ...]   # Vector{SVector{3,Float64}}
```

メモリ上:
```
[x₁ y₁ z₁] [x₂ y₂ z₂] [x₃ y₃ z₃] ...
└─site 1─┘ └─site 2─┘ └─site 3─┘
```

- 1原子分 (x,y,z) が同じ cache line に乗る → **1サイトに着目する計算が速い**
- 全原子の x だけを舐めるのは1要素飛ばし（strided）になり SIMD が効きにくい

### SoA (Structure of Arrays)

同じ成分だけを集めた配列を並べる。

```
x_array = [x₁, x₂, x₃, ...]
y_array = [y₁, y₂, y₃, ...]
z_array = [z₁, z₂, z₃, ...]
```

メモリ上:
```
[x₁ x₂ x₃ ... xₙ] [y₁ y₂ y₃ ... yₙ] [z₁ z₂ z₃ ... zₙ]
```

- 「全原子の x を順に処理」が連続メモリアクセス → **SIMD / GPU で爆速**
- 1サイトの (x,y,z) を読むには3本の別 cache line を踏む

### SpinClusterMC.jl 現状との対応

最適化版の `spins::Matrix{Float64}` (3×N, column-major) は1列=1原子で連続 → **実質 AoS**。
もし `(N, 3)` だったら SoA になっていた。CLAUDE.md の「転置すると壊れる」はまさにこの境界の話。

---

## 各パッケージの方式

### 1. Sunny.jl (Julia, SU(N) + dipole) — 最も近い

**データ構造**: `Array{Vec3, 4}`（`Vec3 = SVector{3,Float64}`）。軸は `(a, b, c, sublattice)` で supercell 座標を indexing に陽に含める。
- `src/MathBasics.jl`: `const Vec3 = SVector{3, Float64}`
- `src/System/Types.jl`:
  ```julia
  const dipoles          :: Array{Vec3, 4}
  const coherents        :: Array{CVec{N}, 4}
  const dipole_buffers   :: Vector{Array{Vec3, 4}}   # LL 積分器の predictor/corrector 用
  ```

**Update path**: `src/MonteCarlo/Samplers.jl` の single-site Metropolis `step!`:
```julia
site = rand(eachsite(sys))
state = sampler.propose(sys, site)    # state.S::Vec3 を作るだけ
ΔE = local_energy_change(sys, site, state)   # sys.dipoles は触らない
accept && setspin!(sys, state, site)         # 受理時だけ書き込み
```
**ロールバック路は存在しない**。

**設計理由**:
- dipole（3-vector）と coherent state（`CVec{N}`）を同一 indexing で扱える AoS が自然
- 4D indexing は Ewald/FFT-based 長距離 dipolar 計算でも有利
- GPU 化（KernelAbstractions）で immutable `SVector` は乗りやすい

### 2. VAMPIRE (C++, 原子論的 LLG + MC)

**データ構造**: 純 SoA。`hdr/atoms.hpp`:
```cpp
extern std::vector<double> x_spin_array;
extern std::vector<double> y_spin_array;
extern std::vector<double> z_spin_array;
```

**Update path**: `src/montecarlo/montecarlo.cpp`:
```cpp
const double S[3] = {atoms::x_spin_array[atom],
                     atoms::y_spin_array[atom],
                     atoms::z_spin_array[atom]};
// trial を Snew[3] に作る → neighbor list で ΔE → 受理時に書き戻し
```

**設計理由**: 数十億原子の LLG / OpenMP / MPI / CUDA 用に「全原子に対する SIMD pass」と「GPU バッファに丸ごとコピー」を最優先。代償として single-site read で3本の別 cache line を踏む。
参考: Evans et al., JPCM 26 103202 (2014)。

### 3. Spirit (C++, Eigen-based)

**データ構造**: `core/include/engine/Vectormath_Defines.hpp`:
```cpp
using Vector3 = Eigen::Matrix<scalar, 3, 1>;
template<typename T> using field = std::vector<T>;
using vectorfield = field<Vector3>;
```
AoS（24B stride）。CUDA ビルドでは managed memory に差し替わる。

**Update path**: `core/src/engine/Method_MC.cpp::Iteration()`:
```cpp
auto & spins_old = *this->systems[0]->spins;
auto   spins_new =  spins_old;                  // sweep 頭に全コピー
...
spins_new[ispin] = local_basis * local_spin_new;
Eold = Energy_Single_Spin(ispin, spins_old);
Enew = Energy_Single_Spin(ispin, spins_new);
if (rejected) spins_new[ispin] = spins_old[ispin];  // ロールバック
```

**設計理由**: メインソルバが LLG/GNEB で `Eigen::Vector3` 演算が中心。`float3`/`double3` への GPU 移植性も意識。
参考: PRB 99 224414 (2019)。

### 4. ALPS spinmc / simplemc (C++, 古典 MC)

**データ構造**: `std::vector<Spin>` で `Spin` は model（Ising/XY/Heisenberg/q-Potts）でテンプレ特殊化。オブジェクト AoS。

**Update path**: `lattice[i]` を読み、spin policy class が trial を値で返す。シャドウ配列なし。

**設計理由**: 性能ではなくモデル横断の汎用性が主目的（一つの `simplemc` driver で Ising bool/XY angle/Heisenberg 3-vector/q-Potts integer を扱う）。

### 5. SpinMC.jl (Buessen) — Julia の比較対象

`src/Lattice.jl`:
```julia
spins::Matrix{Float64}   # 3 × N_sites
```
`getSpin`/`setSpin!` accessor 経由。**現状の SpinClusterMC.jl 最適化版と同じ規約**。

---

## 比較表

| パッケージ | レイアウト | trial の扱い | 主な動機 |
|---|---|---|---|
| Sunny.jl | `Array{SVector{3,Float64}, 4}` (AoS) | commit-on-accept、シャドウなし | dipole/SU(N) 共通 indexing、GPU |
| VAMPIRE | SoA (3本の `std::vector<double>`) | local 3要素 array に集める | 数十億原子 LLG + SIMD/GPU |
| Spirit | `std::vector<Eigen::Vector3>` (AoS) | sweep 頭に全コピー、棄却時ロールバック | LLG/GNEB、`Eigen::Vector3` 演算 |
| ALPS spinmc | `std::vector<Spin>` (オブジェクト AoS) | spin policy が値で返す | モデル横断の汎用性 |
| SpinMC.jl (Buessen) | `Matrix{Float64}` 3×N | accessor 経由 | SpinClusterMC 最適化版と同じ |

---

## simple版への推奨と判断根拠

### (A) レイアウト: `Vector{SVector{3,Float64}}`

最適化版の 3×N とは**意図的に分離**して AoS を採用する。

1. **読みやすさ**: `S_i = spins[i]` がそのまま3-vectorで `Z_l^m(S_i)` や回転に直接流れる。教材性（design_notes.md の目的 (C)）と整合。
2. **転置事故ゼロ**: CLAUDE.md が `3 × n_atoms` の転置で全計算が壊れると明記。`SVector` なら次元が型レベルで固定。
3. **single-site MC ではむしろ速い**: 1サイト24Bを1cache line で読める。3×N が勝つのは「全サイトに対する SIMD pass」のみで、これは template fast path の領分。
4. **`NTuple{3,Float64}` は不可**: `LinearAlgebra` overload が無くアドホックなブロードキャストが必要。

### (B) trial spin: Sunny.jl 流 commit-on-accept

```julia
S_old = spins[i]
S_new = propose(rng, S_old)::SVector{3,Float64}
ΔE = delta_local_energy(h, spins, i, S_new)   # spins は触らない
rand() < exp(-ΔE/T) && (spins[i] = S_new)
```

- `delta_local_energy` は `spins` を書き換えず `S_new` を引数で受ける。
- Spirit 流のシャドウ配列・ロールバックは sweep ごとの全コピーが無駄。
- `zlm_cache` も持たない方針（design_notes.md）なのでバッファ管理を増やさない方が simple さに整合。

### (C) 将来の update への影響

- **Overrelaxation / heatbath**: `setspin!` 系のスカラー API のままで足りる。
- **Wolff cluster**: 反射行列 `R` を `spins[i] = R * spins[i]` for `i ∈ cluster` で一掃。AoS が自然。
- **HMC**: 共役運動量 `π::Vector{SVector{3,Float64}}` を同形で持てる（Sunny の `dipole_buffers` と同じ流儀）。symplectic integrator は `spins .+= Δt .* π` の1行。
- **CMC**: 多サイト同時提案。`delta_local_energy_pair(h, spins, i, S_i_new, j, S_j_new)` を増やすだけでレイアウトは変えなくてよい。

### 採らなかった選択肢

- **VAMPIRE 流 SoA**: 数十億原子 SIMD/GPU 用の選択であり、design_notes.md で GPU は対象外と明示されている本パッケージでは動機が薄い。
- **Spirit 流シャドウ配列**: sweep ごとの全コピーが simple さに反する。Sunny 流の引数渡しで同じ「spins を speculatively 書き換えない」性質が得られる。
- **最適化版と型を共有 (`Matrix{Float64}` 3×N)**: 軸3で完全独立が決まっているため対象外。CI で「同一 XML + 同一初期スピン → energy 値一致」を境界で検証する。

---

## 参考リンク

- [Sunny.jl GitHub](https://github.com/SunnySuite/Sunny.jl)（`src/System/Types.jl`, `src/MathBasics.jl`, `src/MonteCarlo/Samplers.jl`）
- [Sunny.jl paper (arXiv:2501.13095)](https://arxiv.org/html/2501.13095v1)
- [VAMPIRE GitHub](https://github.com/richard-evans/vampire)（`hdr/atoms.hpp`, `src/montecarlo/montecarlo.cpp`）
- [VAMPIRE paper (Evans et al., JPCM 26 103202)](https://iopscience.iop.org/article/10.1088/0953-8984/26/10/103202)
- [Spirit GitHub](https://github.com/spirit-code/spirit)（`core/include/engine/Vectormath_Defines.hpp`, `core/src/engine/Method_MC.cpp`）
- [Spirit paper (PRB 99 224414)](https://dx.doi.org/10.1103/PhysRevB.99.224414)
- [ALPS spinmc docs](https://alps.comp-phys.com/documentation/models/spinmc/intro/)
- [SpinMC.jl (Buessen)](https://github.com/fbuessen/SpinMC.jl)
