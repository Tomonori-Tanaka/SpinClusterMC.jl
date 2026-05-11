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

## 一致判定 — **「定数倍で済むかは未検証」**

上の規格化を見比べる限り **Magesty Zₗₘ ≡ SpheriCart Y_l^m_real**（同一）に見える。ただし
以下の点で **数値検証なしに同一とは断言できない**：

1. **位相規約**: Magesty は `(-1)ᵐ` を `P̄ₗₘ` 側に持たせ、`Zₗₘ` 側で `_parity(n)` を掛ける構造。
   SpheriCart 側で `(-1)^m` がどのレイヤに入っているかで、m が奇のとき符号が反転する可能性が
   ある。
2. **m<0 の sin 系成分の符号**: Magesty は `imag((r̂x + ir̂y)^n)` を取る。SpheriCart の m<0 が
   `sin(|m|φ)` 系なのか `-sin(|m|φ)` 系なのか、ドキュメント明記なし。
3. **`d^m Pₗ/d(cos θ)^m` vs Associated Legendre `P_l^m`** の符号差: Condon-Shortley の
   `(-1)^m` がどちら側に乗っているかで違いが出る。

**確認手順（採用判断前の必須作業）**:

```julia
using SpheriCart, Magesty
# l_max を 2 か 3 に固定
# 単位ベクトル列 (N=20 程度) で両者を評価
# 各 (l, m) について比 sphericart / magesty を計算
# - すべて 1.0 ± 1e-14 ならビット同一
# - 一律 -1.0 → 全体符号差（補正容易）
# - m に応じて ±1 が交互 → 位相規約差
# - 比が無理数（√2 など）→ 規格化差
```

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

## 性能ゲインの不確実性

- SpheriCart Julia 版の公開 benchmark なし（C++ 版の論文はあるが Julia 性能は別）
- 現状の Magesty buffered `Zₗₘ_unsafe`: 37.5 ns/call、allocation 0、sweep 寄与 9%
- `max_l=1` では計算要素は (l_max+1)² = 4 個のみ → SIMD batch の恩恵が小さい可能性
- Metropolis hot path は「1 原子分の 4 要素」を計算する形なので SpheriCart の「N サイト分
  一括」とは噛み合わない。`compute!` の 1 サイト呼びが Magesty 単純呼びより速い保証はない

---

## 結論（メモとしての判断材料）

採用判断には最低限以下が必要：

1. **数値同一性検証**（10 行スクリプト）— Magesty と SpheriCart の `Zₗₘ` が bit-exact か、
   定数倍か、規約差ありかを確定
2. **マイクロベンチ** — 1 サイト `l_max=1〜2` 評価で SpheriCart が Magesty buffered 版を
   上回るか（少なくとも互角か）
3. **スコープ確認** — SpinClusterMC 単独で済むか、Magesty.jl 側にも波及するか

上記 3 点を確認しないまま実装ブランチを切ると、**SCE 係数の意味が変わるサイレントな数値
破壊**または **大量のコード変更後に性能改善が出ないという無駄足**のいずれかになる。
