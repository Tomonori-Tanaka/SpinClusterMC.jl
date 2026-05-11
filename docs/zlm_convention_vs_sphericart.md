# `Zₗₘ` 規約比較メモ — Magesty.jl vs SpheriCart.jl

SpheriCart.jl など外部の球面調和関数パッケージへの差し替えを検討する際、必ず最初に確認
すべき規約差をまとめる。**実装に進む前にこの差を埋める変換式を確定させること**。

調査日: 2026-05-11

---

## なぜ規約差が問題か

`Zₗₘ` の値は SpinClusterMC のホットパス（`_update_atom_zlm_cache!`）だけでなく、
**Magesty.jl 内の `BasisSets.jl` が SALC 係数（XML 内 `<jphi>` の各 cluster の係数テンソル）を
構築する際にも使われている**。つまり既存の `jphi.xml` に書かれている SCE 係数は Magesty の
`Zₗₘ` 規約で fitting されたものであり、`Zₗₘ` の符号や規格化を変えると：

1. クラスタエネルギーの値が定数倍ずれる（規格化のみ変えた場合）
2. m の符号反転で項間相殺が壊れる（位相規約を変えた場合）
3. SALC 係数の対称性破れ（テンソル基底の直交性喪失）

最悪のシナリオは「テストは通るがエネルギー絶対値が物理的にずれる」「特定対称系のみ間違う」
など**サイレントな数値破壊**。CLAUDE.md にも「**`Φᵥ` の定義はMagesty.jl側**」と明記されており、
規約変更は本リポジトリ単独で判断してはならない領域。

---

## Magesty.jl `Zₗₘ_unsafe` の定義（実装基準）

ソース: `Magesty.jl/src/utils/MySphericalHarmonics.jl` (commit `bc79460`)

### `P̄ₗₘ`（normalized associated Legendre, Drautz 規約）

docstring (行 105):

```
P̄ₗₘ = (-1)ᵐ √((2l+1)/(4π) × (l-m)!/(l+m)!) × d^m/d(r̂z)^m Pₗ
```

- 規格化定数: √((2l+1)/(4π) × (l-m)!/(l+m)!)
- **Condon-Shortley 位相 `(-1)ᵐ` を含む**
- 内部実装は `LegendrePolynomials.dnPl(x, l, m)` を呼び、自前で `(-1)ᵐ × √((2l+1)/(4π) (l-m)!/(l+m)!)` を掛ける
- 参考文献: R. Drautz, Phys. Rev. B 102, 024104 (2020)

### `Zₗₘ_unsafe(l, m, uvec)`（実テッサー型）

ソース (行 471-480):

```julia
function Zₗₘ_unsafe(l::Integer, m::Integer, uvec::AbstractVector{<:Real})::Float64
    m == 0 && return P̄ₗₘ(l, 0, uvec[3])

    n = abs(m)
    plm = P̄ₗₘ(l, n, uvec[3])
    c = _parity(n) * √2 * plm            # _parity(n) = (-1)^n
    z_pow = ComplexF64(uvec[1], uvec[2])^n
    return m > 0 ? c * real(z_pow) : c * imag(z_pow)
end
```

docstring (行 446-450) の数式表現:

```
m = 0:  Zₗ₀  = P̄ₗ₀(r̂z)
m > 0:  Zₗₘ = (-1)ᵐ √2 P̄ₗₘ(r̂z) ∑ₖ (-1)ᵏ C(m,2k)   r̂ₓ^(m-2k)   r̂ᵧ^(2k)
m < 0:  Zₗₘ = (-1)ⁿ √2 P̄ₗₘ(r̂z) ∑ₖ (-1)ᵏ C(n,2k+1) r̂ₓ^(n-2k-1) r̂ᵧ^(2k+1)    (n=|m|)
```

### Magesty の規格化（m≠0 ケースの全係数）

`P̄ₗₘ` に含まれる `(-1)ᵐ` と外側の `_parity(n) = (-1)^n` で **`(-1)²ᵐ = 1`** となり、最終的な
`Zₗₘ` の規格化部分は

```
|Zₗₘ| の係数 (m≠0) = √2 × √((2l+1)/(4π) × (l-|m|)!/(l+|m|)!)
                  = √((2l+1)/(2π) × (l-|m|)!/(l+|m|)!)
```

m=0 では √2 が掛からないので

```
|Zₗ₀| の係数 = √((2l+1)/(4π))
```

---

## SpheriCart.jl の規約（公式ドキュメント記載）

出典: https://sphericart.readthedocs.io/en/latest/maths.html
（取得日: 2026-05-11、Julia 版 API は `julia/README.md`、性能 benchmark の数値は未公開）

### 提供される harmonics 種類

- **実球面調和関数** `Y_l^m_real`（単位ベクトル `r̂` の関数）
- **実立体調和関数** `r^l × Y_l^m_real`（`r` 込みの多項式）
- Racah 正規化（"Racah normalization" と明記）

### Racah 正規化の係数

公式ドキュメントによれば、SpheriCart の real spherical harmonics の規格化因子は

```
F_l^m = (-1)^m √((2l+1)/(2π) × (l-m)!/(l+m)!)        (m ≠ 0)
F_l^0 = √((2l+1)/(4π))                                  (m = 0)
```

これは Magesty の `Zₗₘ` の規格化と **数式上は一致する**（√2 因子は Racah で内蔵済み）。

### API（要点）

- `compute!(sph, r̂_batch, output)` — in-place、`l_max` まで一括計算、allocation 0
- メモリレイアウト: `output[i, l*(l+1)+m+1]` 形式（1-indexed Julia）
- SIMD batch 実装あり（複数入力で恩恵）

---

## 一致判定 — **bit-exact 一致を実測で確認済み（2026-05-11）**

上の規格化を見比べる限り **Magesty Zₗₘ ≡ SpheriCart Y_l^m_real**（同一）に見えるが、念のため
数値検証を実施。

### 検証スクリプト

ブランチ `experiment/sphericart-zlm-compat` 上で `/tmp/sphericart-zlm-check/compare.jl` を作成。
- `L_MAX = 3`
- N = 25 個の単位ベクトル（軸方向 6 点 + 対角 + 任意点 + 18 ランダム点、seed 固定）
- 全 (l, m) ∈ {l ≤ 3, |m| ≤ l}（16 個）で `Magesty.Zₗₘ_unsafe(l, m, u)` と
  `SpheriCart.compute(SphericalHarmonics(L_MAX), u)[l*(l+1)+m+1]` を比較

### 結果

**全 (l, m) で bit-exact**（`max |Δ| ≤ 3.3 × 10⁻¹⁶`、機械精度の丸め誤差レベル）。

| l | m | max abs diff |
|---|---|---|
| 0 | 0 | 0.0 |
| 1 | -1, 0, 1 | 5.6e-17 |
| 2 | -2, -1, 0, 1, 2 | ≤ 2.2e-16 |
| 3 | -3, …, 3 | ≤ 3.3e-16 |

SpheriCart の値順序は flat index `l*(l+1) + m + 1`（m = -l, …, +l）。

### 含意

- **規約差なし**: 定数倍補正不要、符号反転なし、位相規約一致、m<0 sin 系の符号一致
- **既存 `jphi.xml` の SCE 係数はそのまま有効**（Magesty `BasisSets.jl` が構築する SALC は
  SpheriCart の値で再現可能）
- 上の章で挙げた 3 つの不確実性（位相規約・m<0 sin 系の符号・Condon-Shortley の位置）は
  すべてクリア

### SpheriCart 側の規約確認（事後）

bit-exact だったということは、SpheriCart も以下を満たす：
- Condon-Shortley `(-1)ᵐ` を含む実テッサー型
- m<0 で sin 系（Magesty の `imag(z^|m|)` と同符号）
- Racah 正規化（√((2l+1)/(2π) × (l-|m|)!/(l+|m|)!) × √2 由来）が Magesty と一致

---

## もし規約が一致した場合の置き換え見積もり

### SpinClusterMC 側で必要な変更（影響範囲は局所）

- `src/spin_utils.jl::_update_atom_zlm_cache!`: `compute!(sph, [u], output_row)` に置き換え
- `src/JPhiMagestyCarlo.jl::_tensor_contract_instance` 内 (行 370, 415): SpheriCart 評価に
  置き換え
- `JPhiSpinMC` の `zlm_dnpl_buf` → SpheriCart の前計算 struct (`SphericalHarmonics{Float64}`)
  に差し替え
- Magesty の `Zₗₘ_unsafe` import を削除

### Magesty.jl 側で必要な変更（**広い**）

- `BasisSets.jl` の SALC 構築で使われている `Zₗₘ_unsafe` 呼び出しを SpheriCart に切替（既存
  XML との互換のために規約完全一致が必須）
- `Optimize.jl` の勾配計算で使われている `∂ᵢZlm_unsafe` も SpheriCart の derivative API
  (`compute_with_gradients!`) に置換
- 既存テストのリグレッション確認

つまり **SpinClusterMC 単独では完結しない**。Magesty 側の球面調和関数依存箇所をすべて
洗い出して同時に切り替える必要がある。

### 規約が一致しなかった場合（変換ラッパー方式）

- SpinClusterMC 内で SpheriCart の出力を Magesty 規約に補正する thin layer を作る
- Magesty.jl は無変更、既存 XML 互換維持
- ただし定数倍補正の `* multiplier_table[l, m]` が SIMD 一括計算の恩恵を一部削る
- Magesty 内 `Zₗₘ_unsafe` も並行残存するので「2 つの規約が共存」する状態になり、保守上は
  健全とは言いがたい

---

## 性能比較（実測, 2026-05-11）

`/tmp/sphericart-zlm-check/bench.jl` を BenchmarkTools.jl で実行。allocs はどちらも 0。

### Scenario A: 1-site cache update（hot path 想定）

`_update_atom_zlm_cache!` 相当。Magesty は buffered `Zₗₘ_unsafe` を `(l, m)` ループで呼び、
SpheriCart は `compute(sph, u)` で `SVector` 一括返しを cache row に書き戻す。

| max_l | n_values | Magesty buffered loop | SpheriCart `compute` | SpheriCart `compute!(1)` |
|---|---|---|---|---|
| 1 | 4  | min 35.3 ns | **min  2.8 ns** | min 42.4 ns |
| 2 | 9  | min 95.5 ns | **min  5.2 ns** | min 61.7 ns |

- 1-site では `compute(sph, u)`（SVector 返し）が **10〜20× 高速**
- `compute!(out, sph, [u])`（1-site batched）は dispatch overhead で逆に遅い

### Scenario B: 128-site full rebuild（初期化や global update 想定）

| max_l | Magesty loop | SpheriCart `compute!(N)` | 比 |
|---|---|---|---|
| 1 | min 4.33 μs | **min 0.43 μs** | 10× |
| 2 | min 12.0 μs | **min 0.61 μs** | 20× |

128 site batched 呼びは SIMD と無 dispatch で 1 site あたり ~5 ns。Carlo の `init!` での
`_build_zlm_cache` 相当処理に効く。

### Sweep への寄与

現状の sweep プロファイル: `_update_atom_zlm_cache!` 128 calls/sweep = 4.8 μs (9%)。
SpheriCart `compute` 換算で 128 × 2.8 ns = 0.36 μs (<1%)。
→ sweep 51.1 μs → 推定 47 μs（**約 1.09× 改善**, max_l=1 のとき）

### 出力レイアウト一致性

SpheriCart の flat index `l*(l+1) + m + 1`（m = -l..+l）は SpinClusterMC の
`_zlm_col(l, m_idx) = l² + m_idx`（m_idx = 1..2l+1, m = m_idx - l - 1）と
**列の対応関係まで含めて完全一致**：

| (l, m) | SpinClusterMC `_zlm_col` | SpheriCart index |
|---|---|---|
| (0, 0)   | 1 | 1 |
| (1, -1)  | 2 | 2 |
| (1, 0)   | 3 | 3 |
| (1, +1)  | 4 | 4 |
| (2, -2)  | 5 | 5 |
| (2, +2)  | 9 | 9 |

→ ループの読み替え不要、メモリレイアウトもそのまま使える。

---

## 結論（2026-05-11, 採用済み）

採用判断 3 点：

1. ~~**数値同一性検証**~~ — **完了**。bit-exact（max |Δ| ≤ 3.3e-16, l ≤ 3 全 (l,m), 25 単位ベクトル）
2. ~~**マイクロベンチ**~~ — **完了**。1-site で 12〜18×、128-site batched で 10〜20× 高速
3. ~~**スコープ確認**~~ — **完了**。規約一致により SpinClusterMC 単独差し替えで OK

### 実装結果（sweep ベンチ on bcc_2x2x2 + 2x2x2 タイリング）

| 指標 | Magesty buffered (baseline) | SpheriCart 採用後 |
|---|---|---|
| sweep min | 51.1 μs | **45.2 μs**（1.13×） |
| allocs/sweep | 0 | **0**（維持） |
| `_update_atom_zlm_cache!` per-call | 37.5 ns | **2.0 ns**（19×） |
| Zlm の sweep 寄与 | 9% | **< 0.5%** |

865/865 tests pass、数値結果は bit-exact 一致。Zlm は実質ボトルネックから外れた。

### 採用時のハマりどころ

`JPhiSpinMC` を **parametric** にする必要があった：

```julia
mutable struct JPhiSpinMC{S<:SphericalHarmonics} <: AbstractMC
    ...
    sph::S
    ...
end
```

abstract typed field `sph::SphericalHarmonics` のままだと `compute(sph, u)` が返す
`SVector{(L+1)²,Float64}` のサイズ L を静的に知れず、毎回 ~39 bytes ヒープ割り当て
（~5 KB/sweep × 128 atoms）が発生した。parametric 化で concrete 型に変わり、SVector が
スタック割り当てに戻って 0 alloc を回復。

副作用：
- `::Type{JPhiSpinMC}` 直接ディスパッチを `::Type{<:JPhiSpinMC}` に変更
  （`Carlo.register_evaluables` と `Serialization.deserialize`）
- 他の `mc::JPhiSpinMC` シグネチャは UnionAll マッチで無変更で済む

### 採用に至った変更（実装済み）

- `Project.toml` に `SpheriCart = "0.2"` を追加
- `JPhiSpinMC` を parametric struct `JPhiSpinMC{S<:SphericalHarmonics}` に変更、`sph::S`
  フィールドを追加、`zlm_dnpl_buf::Vector{Float64}` フィールドを削除
- `_update_atom_zlm_cache!` の中身を `compute(sph, u)` ベースに差し替え（Matrix view 用と
  SVector 用の 2 メソッドを提供）
- `_build_zlm_cache` を `compute!(cache, sph, spins)` 一括版に差し替え
- `coupled_cluster_energy` / `_tensor_contract_instance`（reference path）の `Zₗₘ_unsafe`
  呼びも SpheriCart 化（site 別 `l` を `l*l + m_idx` で取り出す形）
- `Magesty.MySphericalHarmonics: Zₗₘ_unsafe` の import を削除
- `_alloc_zlm_dnpl_buf` ヘルパーを削除
- `::Type{JPhiSpinMC}` ディスパッチを `::Type{<:JPhiSpinMC}` に変更（2 箇所）
- test plumbing（5 箇所）を `sph = JMCC.SphericalHarmonics(max_l)` ベースに更新

### 残るリスク

- SpheriCart のバージョン上げで Racah 正規化の規約が変わる可能性 → `compat = "0.2"` で固定
- SpheriCart 側で `SphericalHarmonics(L)` の生成コストが ~数 KB の lookup table を持つ
  ことがある（init で 1 回なので問題なし）
