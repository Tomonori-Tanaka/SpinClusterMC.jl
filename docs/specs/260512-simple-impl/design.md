# Design — `src/simple/` Reference Implementation

ブレスト開始: 2026-05-11 / spec 化: 2026-05-12。

> 関連 spec ドキュメント:
> - 要件: [requirements.md](requirements.md)
> - マイルストーン: [tasklist.md](tasklist.md)
> - スピン配列レイアウト調査: [../../spin_layout_survey.md](../../spin_layout_survey.md)

## 動機
現状の `src/JPhiMagestyCarlo.jl`（1600行）には最適化機構（instance/derived cache,
MPI 分散ビルド, template fast path, PT, HDF5 ckpt, グローバル辞書キャッシュ）が
集中しており、構造が追いにくい。読みやすく拡張しやすい参照実装を `src/simple/` に
別実装として置く。完全な別パッケージにはせず、同じパッケージ内で型・XMLパーサを
共有し、CI で「最適化版と数値一致」を常に検証できる構成を狙う。

## **不変条件（user 指定, 絶対に守る）**
スピンクラスター展開の強みは「ハミルトニアンの形を事前に決めず、`l_max` と
`N-body` だけ変えれば任意のスピンハミルトニアンを表現できる」点。
simple 版でもこの**拡張性は犠牲にしない**。
- 任意の `l_max` を受け取れること。
- 任意の N-body（1体, 2体, 3体, ...）を受け取れること。
- ハミルトニアンの「形」は `jphi.xml` から駆動され、コード側に固定しない。
- → 結果として、テンソル縮約のループは N-body に対して可変長で書く必要がある
  （現実装の `_tensor_contract_instance` と同じ性質を保つ）。
  l_max 固定 / N-body 固定の特殊化路線は不可。

## 議論の軸

**軸1: simple版の目的**（決定済: (B) 主, (C) 副）
- (B) 拡張の足場 — **ただし拡張したいのは「スピン更新アルゴリズム」**。
  SCE は任意項を表現できるため、新項（クラスター項）の追加は本質的に不要。
  追加したい update 例: Wolff 系クラスター更新, overrelaxation, heatbath,
  microcanonical, replica-exchange variants 等。
- (B') **外場の寄与は SCE の外**。Zeeman・external time-dependent field は
  SCE Hamiltonian と**加法的に**プラグインできる構造にする。
  （注: 単一イオン異方性は SCE で N=1, l_max=2 として表現可能なので外場ではない）
- (C) 副目的として教材性: **Magesty.jl の用語・構造との対応**を取る。
  `Cluster`, `BasisSet`, `coupled_basislist`, `Φᵥ`(SALC), `tensor_inner_product`
  などの呼称を踏襲し、コードを Magesty docs と並べて読めるようにする。

**軸2: 削る候補**
- `_HAM_CACHE` / `_ECACHE_CACHE` / `_DERIVED_CACHE` 等のグローバル辞書 — 削るほぼ確定
- Template energy fast path (`_template_local_energy!`) — 削って `coupled_cluster_energy`
  ベースのみに
- MPI 分散ビルド (`_mpi_build_ham_and_cache`) — 削る（rank0で全部組む）
- Instance precompute (`supercell_atom_index` テーブル) — 削って毎回計算
- HDF5 checkpoint / Serialization 拡張 — Carlo デフォルトに任せる
- Parallel Tempering — 残すか切るか未決
- `zlm_cache` の事前計算 — 削る（その場で `compute_zlm` 呼ぶ）

**軸3: コード構造**
- (i) 完全独立: `src/simple/SimpleJPhi.jl` 1ファイル、XMLパーサだけ共有、型も別
- (ii) 型は共有・ロジックだけ別: `kernel=:simple` で切替
- (iii) サブモジュール: `SpinClusterMC.Simple` として並列 export

**軸4: テスト戦略**
- 同一 `jphi.xml` + 同一初期スピン配置で `sce_energy(simple) == sce_energy(optimized)`
- 乱数固定の短い MC run で `:Energy` 系列の一致を確認
- 既存の「2パス整合性」（`tensor_template` vs `tensor`）の延長として CI に組み込み

## 参考文献
- SCE 論文: https://arxiv.org/abs/2512.04458 （user の論文）
- Magesty.jl technical notes: https://Tomonori-Tanaka.github.io/Magesty.jl/technical_notes/
- スピン HMC: https://arxiv.org/abs/1902.02116 （Hamiltonian Monte Carlo for spins）

## 計画中の update アルゴリズム（拡張対象）
- Metropolis single-spin flip（既存・最低限）
- Overrelaxation（gradient 経由）
- Heatbath（SCE 非線形性のため要工夫）
- Wolff 系クラスター更新（2体寄与に限定すれば素直）
- **Hamiltonian Monte Carlo (HMC)**: arXiv:1902.02116 流。
  - 必要なのは `gradient(h, spins, i)` のみ。提案中の API でそのまま実装可能。
  - 全サイト同時更新なので Carlo の sweep 単位とは粒度が違う点に注意。
- Replica exchange / parallel tempering（既存 PT との関係を整理）
- **拘束付き Monte Carlo (Constrained MC)**: Asselin et al., arXiv:1006.3507。
  温度依存磁気異方性の計算用。全磁化 Σ S を所与の方向 n̂ に固定して torque を測定。
  詳細は下の「stateful update の拡張ガイドライン」参照。

## Stateful update の拡張ガイドライン（CMC 等を見据えて）

CMC のような「sweep 関数だけでなく **MC 状態自体に追加情報** を要する」update を
将来足すときの想定：

1. **追加 state**: 拘束方向 n̂・Lagrange 乗数 λ など update 固有の状態を持つ場合は、
   `SCEMC` に optional フィールド（例: `constraint::Union{Nothing, ConstraintState}`）
   を追加するか、`ConstrainedSCEMC` を別型として作る。初版で先回りして仕込む必要はなく、
   実装時に最小限の field を足せばよい。

2. **多サイト同時提案 API**: CMC のキモは「site i を回転 → site j を補正回転」で
   2スピン同時更新になる。現状の `delta_local_energy(h, spins, i, S_new)` は単一
   サイト想定なので、必要になったら以下のいずれかで拡張:
   - **buffer pattern**: `spins` を変更してから `delta_local_energy(h, spins, j, S_j_new)`
     を呼ぶ。仮想的な変更を巻き戻せるよう temporary buffer を用意。
   - **専用関数追加**: `delta_local_energy_pair(h, spins, i, S_i_new, j, S_j_new)` を
     CMC 専用に足す。1ペアの差分を一気に評価。

3. **観測量の拡張**: CMC では `:Energy` ではなく **拘束方向の torque**（≈ 異方性
   エネルギー）が主観測量。`Carlo.measure!` / `register_evaluables` を update_scheme
   に応じて分岐させるか、CMC 専用型に別 dispatch させる。

4. **拡張は API がもう揃っていることに乗る**: `local_energy / delta_local_energy /
   gradient` の分割そのものが (B) の足場。新 update を作るとき、これら energy API は
   そのまま使い回せる — Hamiltonian 評価を書き直す必要はない。CMC は API 拡張も
   伴うが、その他多くの update（overrelaxation, heatbath, Wolff, HMC 等）は現状 API
   だけで足りる。

## エネルギーモジュール API（決定）

最小 API は以下4つ。update アルゴリズム拡張に必要なものは原則これで賄える：

```julia
total_energy(h, spins)::Float64                          # 初期化用
local_energy(h, spins, i)::Float64                       # site i を含む全項の和
delta_local_energy(h, spins, i, S_new)::Float64          # Metropolis 用
gradient(h, spins, i)::SVector{3,Float64}                # ∂E/∂S_i
```

**`gradient` から派生する量**:
- 有効場: `h_eff(h, spins, i) = -gradient(h, spins, i)`
- 接平面成分（spin の長さを保つ微分）: `g_⊥ = g - (g·S_i)S_i`

**SCE 特有の注意**:
- l_max > 1 では `gradient(h, spins, i)` は **S_i 自身にも依存する**
  （Z_l^m(S_i) が S_i の非線形関数なので）。
- → Heisenberg 流の「実効場は他スピン依存・自分には依存しない」前提は使えない。
  heatbath のサンプリング分布や、エネルギー保存型 overrelaxation の構成は
  アルゴリズム側で SCE の非線形性を考慮して書く必要がある。
- ただしこれは **API の問題ではなく実装側の問題**。energy module は
  `gradient` を返す義務だけ負う。

**外場 (ExternalTerm) も同じ4つを実装**:
- 任意の外場項は `local_energy / delta_local_energy / gradient` を実装した型として
  プラグインし、update 側は SCE 寄与と加算する。
- Zeeman は線形なので `gradient` が S_i 非依存になり、heatbath/overrelaxation が
  単純化される（特殊化の余地）。

## update アルゴリズムの抽象化レベル（決定: (a)）
- (a) **抽象型なし、関数を並べる**: `metropolis_sweep!(mc, ctx)`, `overrelaxation_sweep!(mc, ctx)`,
  `hmc_sweep!(mc, ctx)` のように個別関数を `src/simple/updates/` 下に並べる。
- params の `update_scheme::Symbol` で `sweep!` から dispatch。
- 理由: HMC のように site ループ粒度を持たない update も含めて素直に書ける。
  抽象型を入れても今のところ得が薄い。後で必要になったら (b) へ昇格可能。

## 軸3: コード構造（決定: (i) 完全独立）
- `src/simple/` 配下だけ読めば全てが理解できる状態を作る。
- **XML パーサ・型定義も simple 側で独自に持つ**（既存 `xml_io.jl`・`SCEHamiltonian`
  等は再利用しない）。
- 理由: user の最優先要件「Simple だけ読んでわかる」。教材性 (C) と整合。
- トレードオフ:
  - XML パーサ重複（~150行）— 許容。XML スキーマは安定。
  - 型が別なので「型を共有して値だけ比較」はできない。CI は
    「同じ XML を両実装に食わせ、energy の値が一致」で確認する。
- 利点: 二重独立実装による正当性の交差検証が CI で常時走る。

## モジュール露出（決定: (i-1)）
- `SpinClusterMC.Simple` サブモジュール。`using SpinClusterMC.Simple` で利用。
- 依存（Carlo, MPI, HDF5, EzXML 等）は親パッケージと共通。重複インストール無し。
- Carlo 用の MC 型として `Simple.???`（命名未定）を露出。

## jphi.xml の構造（fege_2x2x2 で確認, 2026-05-11）

```
<SALC index="s" body="N" Lf="L">
  <basis multiplicity="m_b" atoms="..." ls="..." Lseq="...">
    w_{s,b}[0] w_{s,b}[1] ... w_{s,b}[2L]   ← (2L+1) 個の数値
  </basis>
  ...
</SALC>
<JPhi unit="eV">
  <ReferenceEnergy>...</ReferenceEnergy>   ← simple版では読まない (CLAUDE.md j0方針)
  <jphi salc_index="s">J_s</jphi>           ← user-facing 結合定数 (SALC 毎 1 スカラー)
</JPhi>
```

Lf 別の `<basis>` 値数（fege 確認）:
- Lf=0: 1 個 (等方)
- Lf=1: 3 個
- Lf=2: 5 個 (異方性)
- Lf=4: 9 個

`Lseq` 属性: 中間結合 path (`l1 ⊗ l2 → L12, L12 ⊗ l3 → L123, ...`) を指定。
N=2 では空（中間結合不要）。N≥3 では必要。Magesty の
`enumerate_paths_left_all(ls)` が同じ列挙を行う。

## エネルギー縮約（Magesty `predict_energy` で検証, 2026-05-11）

```
E = Σ_s J_s · (4π)^(N_s/2) · Σ_{b ∈ SALC_s} m_b ·
        Σ_{Mf=-Lf}^{Lf} w_{s,b}[Mf] · Σ_{m1..mN} T_real[l1 m1; ..; lN mN | Lf Mf]
            · ∏_i Z_{l_i}^{m_i}(S_{a_i})
```
- `T_real`: **tesseral CG テンソル**
  = `complex_to_real_tensor(coeff_tensor_complex(ls, Lf, path), ls, Lf)`
- `Z_{l}^{m}`: tesseral spherical harmonics (実 Zlm, NOT 複素 Ylm)

Magesty 内部マッピング (`Optimize.jl:design_matrix_energy_element` で確認):
- `cbc.coeff_tensor[m1,...,mN, mf]` = T_real
- `cbc.coefficient[mf]` = w_{s,b}[Mf] (= `<basis>` text content)
- `cbc.multiplicity` = m_b
- `h.jphi[s]` = J_s (= `<JPhi>`)
- `_cluster_scaling(N)` = (4π)^(N/2)

**重要な区別**: `<JPhi>` の J_s（user-facing 結合定数）と `cbc.coefficient[mf]`
（SALC 内部の対称化重み）は**別物**。前者は SALC ごとの 1 スカラー、後者は
SALC 内の各 basis ごとの (2Lf+1) ベクトル。

## ClusterInstance / CGTable（決定: CG を CGTable に切り出し）

CG 係数は `(ls, Lf, path)` のみに依存し全 instance で共有可能なので、
Hamiltonian レベルの read-only `CGTable` に切り出す。`ClusterInstance` は
jphi.xml 由来データのみを保持する。

```julia
# 読み取り専用 CG テーブル
struct CGTable
    entries::Dict{Tuple{Vector{Int}, Int, Vector{Int}}, Array{Float64}}
    # key:   (ls, Lf, Lseq)   Lseq は <basis Lseq=""> から (length == N-2)
    # value: tesseral CG テンソル T_real
    #        shape: (2l1+1, ..., 2lN+1, 2Lf+1)
end

# jphi.xml 由来のデータのみ
struct ClusterInstance
    atoms::Vector{Int}             # スーパーセル内原子インデックス
    ls::Vector{Int}                # 各原子の角運動量
    Lf::Int                        # SALC 最終角運動量 (0=isotropic, ≥1=anisotropy)
    Lseq::Vector{Int}              # 中間結合 path (<basis Lseq="...">, length == N-2)
    salc_weights::Vector{Float64}  # <basis> 内 (2Lf+1) 個の重み (= w_{s,b})
    J::Float64                     # <JPhi> から (eV)
    multiplicity::Int              # <basis multiplicity="...">
end

struct SpinClusterHamiltonian
    # ... 格子・スピン情報 ...
    instances::Vector{ClusterInstance}
    cg_table::CGTable              # 構築時に1回作って以降 read-only
end
```

## CG 計算の出処（決定: Magesty.AngularMomentumCoupling に依存）

- **球面調和関数は tesseral（実 Zlm）**。標準の Racah 公式で得られる複素CGは
  そのまま使えず、複素→実変換と位相補償が必要。CLAUDE.md にも明記されている。
- Magesty に既に正しく実装されている:
  - `coeff_tensor_complex(ls, Lf, path)`: 複素CG（Racah formula）
  - `complex_to_real_tensor(Ccx, ls, Lf)`: tesseral 変換
  - `build_all_real_bases(ls)`: 全 Lf・path の tesseral CG を一括構築
  - `c2r_sph_harm_matrix(l)`: 複素Y→実Z 変換行列
  - `enumerate_paths_left_all(ls)`: 中間結合 path 列挙 (`<basis Lseq="">` に対応)
- 自前実装すると 150-300 行、tesseral 位相補償でバグりやすい。
- → **「完全独立」原則の修正**: simple 版は SCE 実装本体（XML 構文・型・
  energy/update 機構）を独自に持つが、**角運動量結合の数学
  (`Magesty.AngularMomentumCoupling`) には依存してよい**。これは SCE 固有
  ではない数学ユーティリティであるため。Magesty docs との対応もより明確になる。

## measure! の拡張ポリシー（決定: callback in params）

ユーザー定義観測量（副格子磁化, staggered M, 構造因子の特定 q, ...）への対応。
simple さを保ちつつ拡張可能にするため、**MC に callback フィールドを持たせて
params 経由でユーザーが渡す方式**を採用する。

```julia
mutable struct SCEMC <: Carlo.AbstractMC
    # ...
    extra_measure::Function       # default: (mc, ctx) -> nothing
    extra_evaluables::Function    # default: (eval, params) -> nothing
end

function Carlo.measure!(mc::SCEMC, ctx)
    measure!(ctx, :Energy, mc.energy / n_atoms)
    measure!(ctx, :Magnetization, total_magnetization(mc.spins))
    mc.extra_measure(mc, ctx)        # ← user hook
    return nothing
end

function Carlo.register_evaluables(::Type{<:SCEMC}, eval, params)
    # 組み込み (Energy 由来の比熱等)
    Carlo.evaluate!(:SpecificHeat, eval, (:Energy,)) do E
        n_atoms * E.var / T^2
    end
    (get(params, :extra_evaluables, (e,p) -> nothing))(eval, params)
end
```

ユーザー側使用例（副格子磁化）:
```julia
params[:A_indices] = [1, 3, 5, ...]
params[:B_indices] = [2, 4, 6, ...]
params[:extra_measure] = function(mc, ctx)
    M_A = sum(@view mc.spins[:, params[:A_indices]]) / length(params[:A_indices])
    M_B = sum(@view mc.spins[:, params[:B_indices]]) / length(params[:B_indices])
    measure!(ctx, :M_A, M_A)
    measure!(ctx, :StaggeredM, norm(M_A - M_B))
end
```

**採用理由**:
1. simple さ: `measure!` を読めば「組み込み」「user hook」の2層構造が即座にわかる。
2. ユーザー負担最小: 関数1個書いて params に詰めるだけ。MC 型の継承・override 不要。
3. (B) 拡張の足場: フィールド名 `extra_measure` 自体が「ここに足せる」と示している。
4. パフォーマンス: measure は sweep に1回程度。Function field の indirection は無視可能。

**却下した代替案**:
- (m1) ハードコード: 系依存観測量（副格子磁化等）に絶対に届かない。
- (m3) サブタイプで override: `init!/sweep!/write_checkpoint!` 等の delegate が
  ユーザー側に必要で、observable 1つ足すために定型コードが増える。

## Parallel Tempering（決定: 初版では切る）
- PT 関連の周辺機能（`Serialization.serialize/deserialize` 拡張 ~50行,
  MPI 対応 `write_checkpoint` ~30行, Carlo バージョン互換パッチ ~25行）が
  「simple」の読みやすさを圧迫するため、初版では切る。
- ただし**後付け可能な構造を保つ**:
  - `T` は mutable field として保持（PT 時に変わるため）
  - `energy` を field として保持（`parallel_tempering_log_weight_ratio` から直読）
  - `xml_path`, `repeat` を保持（後の serialize/deserialize の再構築起点）
- 後で PT が必要になったら以下を足すだけで動かせる:
  - `parallel_tempering_log_weight_ratio` (1関数, ~2行)
  - `parallel_tempering_change_parameter!` (1関数, ~2行)
  - `Serialization.serialize/deserialize` 拡張 (~30行, MPI ビルドが無い分シンプル)
  - 必要なら MPI ckpt

## 初版に入れる update（決定: Metropolis のみ）
- 初版は **Metropolis single-spin flip のみ**を実装。
- `src/simple/updates/metropolis.jl` に置き、後で `overrelaxation.jl`, `hmc.jl`,
  `heatbath.jl`, `cmc.jl` を同じディレクトリに追加できる構造にしておく。
- (B) の「拡張の足場」は **拡張点が明示されていること**であって、初版から全部
  入っていることではない、という整理。

## MC 型の命名（決定）
`SpinClusterMC.Simple.SCEMC` を Carlo の `AbstractMC` 継承型として露出。

## Magesty 用語との対応（決定）

| 既存 (`JPhiMagestyCarlo.jl`) | simple 版 |
|---|---|
| `SCEHamiltonian` | `SpinClusterHamiltonian` |
| `coupled_cluster_energy` | `cluster_energy` |
| `ClusterInstance` | `ClusterInstance`（中身は簡素化） |
| `zlm_cache` | 持たない（その場計算） |
| `JPhiSpinMC` | `SCEMC` |
| （CG係数は ClusterInstance に内包） | `CGTable` として分離 |

## ディレクトリ構成（決定）
```
src/simple/
├── Simple.jl                 # サブモジュール本体・include 集約・export
├── xml_io.jl                 # jphi.xml の読み込み（独自実装）
├── types.jl                  # SpinClusterHamiltonian, ClusterInstance, CGTable
├── cg.jl                     # CGTable 構築 (Magesty.AngularMomentumCoupling 経由)
├── energy.jl                 # total_energy, local_energy, delta_local_energy, gradient
├── external.jl               # ExternalTerm 抽象 + Zeeman 等
├── spin_proposal.jl          # _rand_unit_spin, _propose_spin_geodesic, init helpers
├── mc.jl                     # SCEMC 型 + Carlo.init!/sweep!/measure!/register_evaluables
└── updates/
    └── metropolis.jl         # metropolis_sweep!
    # 将来: overrelaxation.jl, hmc.jl, heatbath.jl, cmc.jl
```

---

## 全決定事項サマリ（着手前確認用）

| カテゴリ | 項目 | 決定 |
|---|---|---|
| **目的** | 主 | (B) update 拡張の足場 |
|  | 副 | (C) Magesty.jl との対応で教材性 |
| **不変条件** | 拡張性 | 任意 `l_max` / N-body / Lf を犠牲にしない |
| **構造** | コード | (i) 完全独立（XMLパーサ・型も独自） |
|  | 露出 | (i-1) `SpinClusterMC.Simple` サブモジュール |
|  | 外部依存（緩和） | `Magesty.AngularMomentumCoupling` (CG計算) のみ依存可 |
| **API** | Energy | `total_energy / local_energy / delta_local_energy / gradient` |
|  | 外場 | `ExternalTerm` 抽象 + 同じ4関数を実装 |
|  | Update 抽象化 | (a) 関数並列、`params[:update_scheme]` で dispatch |
|  | 観測量拡張 | callback (`params[:extra_measure]`, `params[:extra_evaluables]`) |
| **初版機能** | Update | Metropolis のみ |
|  | PT | 切る（後付け可能な field 配置は維持） |
|  | MPI 分散ビルド | 切る |
|  | グローバル辞書キャッシュ | 切る |
|  | Template fast path | 切る |
|  | Instance precompute | 切る |
|  | zlm cache | 切る（その場計算） |
|  | HDF5 ckpt 拡張 | 切る（Carlo デフォルト） |
| **型** | MC 型名 | `SpinClusterMC.Simple.SCEMC` |
|  | Hamiltonian 型 | `SpinClusterHamiltonian` |
|  | クラスター項 | `ClusterInstance` (atoms, ls, Lf, Lseq, salc_weights, J, multiplicity) |
|  | CG | `CGTable` (Hamiltonian 直下、`(ls, Lf, Lseq)` → tesseral CG) |
|  | Zlm | SpheriCart 直接依存（自動正規化あり） |
|  | Spin proposal | 既存 params 規約踏襲 (`:initial_spins`, `:spin_theta_max`, `:renorm_every=1000`) |
|  | `:initial_spins` 拡張 | 案A: Symbol (`:random`/`:ferromagnetic`) / Tuple / SVector / Matrix (base or supercell) を型 dispatch |
| **検証** | CI | 同一 XML + 同一初期スピンで最適化版と energy 値一致 |
| **入口** | サンプル | `examples/` 直下に番号付き `.jl` を配置（01_quickstart → 05_custom_observable） |
| **将来拡張** | Stateful update (CMC等) | MC 型に optional field 追加、`delta_local_energy_pair` 追加 |
|  | HMC | 既決 API (`gradient`) でそのまま実装可 |
|  | Overrelaxation/Heatbath | `gradient` 経由、SCE 非線形性は実装側で対応 |

---

## 着手前に最終確認した決定（2026-05-12）

ブレスト段階で解像度を上げきれていなかった 3 点 + 周辺事項を、調査結果を踏まえて
最終判断したもの。すべて確定済み。

### (1) Spin proposal の規約（決定: 既存規約踏襲, default `renorm_every=1000`）

**調査結果**:
- 既存 (src/spin_utils.jl): `_rand_unit_spin(rng)`（球面一様）と
  `_propose_spin_geodesic(rng, ux, uy, uz, theta_max)`（接平面に乱数接線 → 角度
  θ ∈ [-θ_max, θ_max] 一様で測地線回転）。
- params 規約: `:initial_spins` (3 × base_n_atoms), `:spin_theta_max`
  (未指定なら uniform proposal), `:renorm_every` (default 1000).

**決定**:
- params 規約（`:initial_spins` / `:spin_theta_max` / `:renorm_every`）はそのまま踏襲。
  default も既存と同じ（`renorm_every = 1000`、`0` で無効化）。
- proposal 関数の実装本体は simple 側で独自に書き直す（`src/simple/spin_proposal.jl`）。
  シグネチャは `(rng, ux, uy, uz, theta_max) -> (ux', uy', uz')`。
- 理由: CI で「同一 params + 同一 seed → energy 一致」が一番素直に書ける。
  proposal 関数を独自実装することで教材性 (C) は確保できるので、API を変える動機は薄い。

**`:initial_spins` の受け入れ型を拡張**（決定: 案A — 単一 param に型 dispatch）:

| 値 | 動作 |
|---|---|
| 未指定 / `nothing` / `:random` | 全 supercell スピンを球面一様にサンプル |
| `:ferromagnetic` | 全 site を `+z` 方向に揃える（強磁性研究の事実上の標準） |
| `NTuple{3,<:Real}` / `SVector{3,<:Real}` | 受け取った方向を正規化して全 site 揃え |
| `AbstractMatrix{<:Real}` size `(3, base_n_atoms)` | 既存どおり base cell 配置を tile |
| `AbstractMatrix{<:Real}` size `(3, n_atoms)` | supercell 全体を直接指定（新規） |
| その他 | `ArgumentError` |

実装側:
- 行列は列数で base/supercell を dispatch。`size(M, 1) == 3` をまず assert。
- Symbol / NTuple / SVector ルートは内部で `_uniform_direction_init!(spins, direction)`
  に統一（強磁性プリセットは `direction = SVector(0,0,1)`）。
- ゼロベクトル方向は `ArgumentError`（normalize で NaN になるため事前に弾く）。
- 既存最適化版でも将来同じ拡張を入れたくなる可能性が高いので、API は最初から simple 版で
  揃えておく。最適化版への back-port は別タスクとして切り出し可能。

利点:
- 1 つの param で全モードを表現、設定ファイルが読みやすい。
- 教材性: `:ferromagnetic` 1 語で動くのは初学者に親切。
- 拡張性: 将来 `:neel_z`, `:helix(q)` 等を symbol/struct で追加可能。

**`renorm_every` の役割整理**（SpheriCart 確認結果を踏まえて, 2026-05-12）:
- 浮動小数点 drift 対策の周期的再正規化（`sweep_count % renorm_every == 0` で `s/|s|`）。
- **Zlm 計算自体には drift が伝播しない**: SpheriCart は `compute(basis, 𝐫)` の冒頭で
  必ず `𝐫̂ = 𝐫 / norm(𝐫)` を行う（`SpheriCart/src/spherical.jl:55-58`）。
  実測でも `compute(basis, k * u)` は `compute(basis, u)` と max |Δ| ≤ 1.7e-16 で一致。
- したがって `renorm_every` の必要性は **`mc.spins` に格納される S 自体を unit に保つ**
  ためにある。具体的には:
  - 将来の HMC / gradient ベース update で `S · ∇E` のような ‖S‖=1 前提の量を
    使うため。
  - checkpoint や CMC（`Σ S = const`）で stored spin の整合性を保つため。
- simple 版実装上の利点: zlm cache を持たないので、再正規化後の cache rebuild が
  不要 → 3 行（全 spin に `s/|s|` を回すだけ）。
- docstring に「Zlm 計算は SpheriCart 側で自動正規化されるので drift は伝播しない。
  この renormalization は stored spin direction を unit に保つための保険である」と明記。

**SpheriCart の非正規化入力の振る舞い**（実測, 2026-05-12）:
- non-unit ベクトル（同方向）→ 正しい結果（自動正規化）。
- drifted（unit ± 1e-12）→ 約 1e-12 だけずれた値が返る（線形応答）。
- **ゼロベクトル** → l=0 のみ定数、l≥1 は **全 NaN**。
- → simple 版の `_propose_spin_geodesic` 相当でも、結果が unit から離れすぎないこと
  は実装側で気にする必要なし。ただしゼロ入力だけは事前に弾く（既存実装の
  `nrm < 1e-14 then fallback to random unit` 規約を踏襲）。

### (2) `<basis Lseq>` の規約と CG path 抽出（決定: Magesty に合わせて `Lseq`）

**調査結果**:
- Magesty 型 `CoupledBasis` の assertion (types/Basis.jl:62-63):
  **`length(Lseq) == N - 2`** （N=2: 空, N=3: 1個, ...）
- XML 例:
  - fege_2x2x2 (全 N=2): `Lseq=""` (空)
  - ferh_4x4x4 SALC index 23 (N=3, ls=[1,1,2]): `Lseq="2"` (= l1⊗l2 → L12=2)
- Magesty `build_all_real_bases(ls)` は `(Lf, Lseq) → tesseral_CG_tensor` のテーブル
  を返す（BasisSets.jl:229-235）。

**決定**:
- `ClusterInstance` の field 名を `Lseq::Vector{Int}` にする。
- `CGTable.entries` のキーも `(ls, Lf, Lseq)` に揃える。
- 理由: 教材性 (C) の主目的「Magesty docs と並べて読める」を満たすため。XML 属性名
  `Lseq` と Magesty 型 field と simple 版 field を一語で揃える。
- パース実装:
  ```julia
  parse_lseq(s::AbstractString) = isempty(strip(s)) ? Int[] : parse.(Int, split(s))
  ```
- CGTable 構築: unique な `ls` ごとに `Magesty.AngularMomentumCoupling.build_all_real_bases(ls)`
  を 1 回呼んで `(Lf, Lseq) → tensor` を取得し、`entries[(ls, Lf, Lseq)]` に詰める。
  実装時に CGTable のキー長 invariant `length(Lseq) == N - 2` をコンストラクタで
  assert する。

### (3) Zlm 計算の出処（決定: SpheriCart）

**調査結果**:
- `Magesty.MySphericalHarmonics.Zₗₘ_unsafe(l, m, uvec)::Float64` が export されている
  (src/utils/MySphericalHarmonics.jl:471)。
- 既存 SpinClusterMC.jl は `SphericalHarmonics` (sphericart) を使用し、Magesty
  `Zₗₘ_unsafe` との数値一致は docs/zlm_convention_vs_sphericart.md で
  bit-level 検証済み (max |Δ| ≤ 3.3e-16, l ≤ 3)。

**決定: (z2) SpheriCart を直接依存**。
- 最適化版と同じ実装を使うので、Zlm 単体は bit-exact に揃う。energy は集約順序の
  違いで bit-exact にはならないが、Zlm が一致する分だけドリフトの解析が容易になる。
- SpheriCart は `compute(basis, 𝐫)` で **入力の自動正規化** を行う
  (`spherical.jl:55-58`)。non-unit ベクトル入力でも正しく動く（max |Δ| ≤ 1.7e-16）。
  ただしゼロベクトルは l=0 を除き **NaN** を返す。
- simple 版は zlm cache を持たないので、`compute(sph, S_i)` をその場で呼ぶだけ。
  SVector 入力なら `STATIC=true`（`max_l ≤ 15`）でスタック上に SVector を返すため
  allocation も無い。
- Magesty CG モジュール (`AngularMomentumCoupling`) には依存するが、Zlm については
  既存最適化版と同じ SpheriCart で揃える方針。「数学ユーティリティとして
  Magesty に揃える」原則の例外だが、CI の数値一致のしやすさを優先。

### (4) CI の数値一致閾値（決定）

simple 版と最適化版は同じ XML / 同じ初期スピンを食わせても、ループ順序とアキュムレータ
構造が違うので **bit-exact にはならない**（既存の `:tensor` / `:tensor_template` の
2 パスでも summation order までしか一致しない）。simple 版の目的は「読みやすさ・教材性」で
あって最適化追従ではないので、bit-exact を要求しない。

既存テストの規約に合わせる:

| 比較 | 規約 | 出処 |
|---|---|---|
| `total_energy(simple) ≈ sce_energy(optimized)` | `rtol = 1e-8` | 既存 `sce_energy` ref vs fast (test/runtests.jl:144) |
| `local_energy_i(simple) ≈ ...(optimized)` | `rtol = 1e-10` | 既存 cached vs ref (test/runtests.jl:165, 194) |
| `delta_local_energy(simple) ≈ ...(optimized)` | `rtol = 1e-7` | 既存 ΔE consistency (test/runtests.jl:255) |
| `compute(sph, S)` 出力 (Zlm) | `atol = 0` (bit-exact) | 同じ SpheriCart 呼び出しなので可能 |

### 実装中に判断する事項

- `enabled_bodies` field の要否（最適化版にあるが simple 版で必要かは実装してから判断）。
  判断トリガは M7 (`mc.jl` 着手時) — tasklist.md 参照。

## コーディング方針: 数式を含む docstring（決定）

simple 版の主目的 (C) 教材性のため、**数式を実装する関数には markdown 形式の
docstring で数式そのものを書く**。Julia の docstring は LaTeX math
（`$...$` インライン, ` ```math ` ブロック）を直接サポートし、Documenter.jl で
そのまま docs にレンダリングされる。

**対象**:
- エネルギー縮約 (`cluster_energy`, `local_energy`, `total_energy`, `delta_local_energy`)
- 勾配 (`gradient`)
- CG テーブル構築 (`build_cg_table`)
- SALC 関連の補助関数
- 外場項 (`Zeeman`等)
- update アルゴリズム (`metropolis_sweep!`, 将来 `hmc_sweep!` 等)

**対象外**: 数式と無関係なユーティリティ（XMLパーサ、I/O、buffer 管理など）。
これらは通常の docstring（型と引数の説明）のみで OK。

**書式例**:
```julia
\"\"\"
    cluster_energy(ins::ClusterInstance, spins, cg::CGTable) -> Float64

Compute the energy contribution of one cluster instance.

Implements:

```math
E_{\\\\rm ins} = J \\\\cdot m \\\\cdot (4\\\\pi)^{N/2} \\\\cdot
              \\\\sum_{M_f=-L_f}^{L_f} w[M_f] \\\\cdot
              \\\\sum_{m_1..m_N} T_{\\\\rm real}[l_1 m_1; ..; l_N m_N | L_f M_f] \\\\cdot
              \\\\prod_i Z_{l_i}^{m_i}(S_{a_i})
```

where `T_real` is the tesseral CG tensor stored in [`CGTable`](@ref).

# Arguments
- `ins::ClusterInstance`: cluster term from jphi.xml.
- `spins::AbstractMatrix`: 3 × n_atoms spin directions.
- `cg::CGTable`: precomputed tesseral CG tensors, keyed by `(ls, Lf, Lseq)`.

# Reference
- SCE paper: arXiv:2512.04458 Eq. (X).
- Magesty technical_notes: \\\$\\\\Phi_v\\\$ の定義
  (https://Tomonori-Tanaka.github.io/Magesty.jl/technical_notes/).
\"\"\"
function cluster_energy(ins::ClusterInstance, spins, cg::CGTable)
    ...
end
```

**ポリシー**:
1. **数式を実装する関数**: docstring に LaTeX 数式 + Magesty docs / SCE 論文への
   参照リンクを必ず含める。コード本体は数式と1対1対応するよう書く（変数名は
   数式の記号と揃える）。
2. **CLAUDE.md との整合**: 既存の「エクスポートされる API には明示的な型アノテー
   ションと docstring」を満たしつつ、simple 版では「数式を含む API には数式も」
   と一段強める。
3. **インラインコメントは「種類で判断」**: CLAUDE.md の「Default to writing no
   comments」は WHAT 説明や変更履歴的なコメントに対する規制で、simple 版でも
   それは維持する。ただし以下は教材性 (C) のために**書く**:
   - **数式の構造を示す section header**: 多段の Σ / Π を実装するとき、「今どの
     和の中にいるか」を navigation aid として明示する。例:
     ```julia
     # ---- Σ_{Mf}: SALC 最終角運動量成分の和 ----
     for mf in 1:(2*Lf+1)
         # ---- Σ_{m1..mN}: CG 縮約 ----
         for mtuple in CartesianIndices(...)
             ...
         end
     end
     ```
   - **非自明な規約・罠**: tesseral CG の位相補償、index shift (`m_idx = m + l + 1`)、
     SALC 規格化、`atoms` のソート規約など、読者が黙ってバグを埋めうる箇所。
   - **アルゴリズム上の判断**: なぜこの順序か、なぜここで shuffle か、など WHY。

   一方、**書かない**:
   - 変数名や型から明らかなこと（`# compute Zlm` の直後に `Z = compute_zlm(...)`）
   - docstring の数式と重複する WHAT 説明
   - 現在のタスクや変更履歴的コメント

4. **コメント・docstring・コミット・PR はすべて英語で書く**: simple 版に限らず
   本リポジトリ全般の方針。`docs/` 下の design notes は日本語可だが、ソースコード
   中のコメント・docstring、Git コミットメッセージ、GitHub PR タイトル/本文は
   英語に統一する。
5. **参照の付け方**: 「論文の式番号」「Magesty docs の対応セクション URL」を
   docstring 末尾の "References" 節に明記。Magesty 側のドキュメントが更新
   されたら、こちらも追随する（数式 docstring は **Magesty docs と独立に書く
   のではなく対応付ける**ことが目的）。

## ベンチマーク方針（決定）

**目的**: simple 版を**速くする**ためではなく、**どこが遅いのか説明できる**ため。
simple 版が最適化版より遅いことは設計上当然なので、絶対速度ではなく
**構造に即した遅さの分布**が見えればよい。

**フレームワーク**: BenchmarkTools.jl を採用。
- 理由: 標準・情報多・`Profile.jl` 連携。生 `@elapsed` だと per-flip µs オーダーの
  比較で GC/JIT ノイズが乗ってしまい不適。
- 却下: PkgBenchmark.jl / AirspeedVelocity.jl（regression tracking 目的なので過剰）。

**ボトルネック特定のため、コンポーネント別に切る** — end-to-end の wall time だけ
だとどの関数が重いか追えないので、各 API 単位で `@benchmark` する。

**ディレクトリ構成**:
```
benchmark/simple/
├── README.md             # 使い方・結果の読み方
├── fixtures.jl           # bcc_2x2x2 / fege_2x2x2 / ferh_4x4x4 のロード共通化
├── bench_construction.jl # 構築フェーズ: XML / CG / Hamiltonian
├── bench_energy.jl       # total / local / delta_local / gradient
├── bench_sweep.jl        # metropolis_sweep! end-to-end + per-flip cost
├── bench_compare.jl      # 同じ fixture で simple vs 最適化版の比率を出す
└── runbench.jl           # 全部走らせて1つのサマリ表に集約
```

**測る粒度（最低限）**:

| フェーズ | 関数 | 頻度 | 何を測る |
|---|---|---|---|
| 構築 (1回) | `parse_jphi_xml` | 1 | wall, alloc |
|  | `build_cg_table` | 1 | wall, alloc |
|  | `SpinClusterHamiltonian(...)` | 1 | wall, alloc |
| Energy (per measurement) | `total_energy(h, spins)` | O(n_meas) | wall, alloc |
|  | `local_energy(h, spins, i)` | O(n_atoms · n_meas) | wall/site, alloc/site |
| Hot path (per flip) | `delta_local_energy(h, spins, i, S_new)` | O(n_atoms · n_sweep) | **wall/flip, alloc/flip** |
|  | `metropolis_sweep!` 全体 | O(n_sweep) | wall/sweep, **wall/flip** |
| HMC/Overrelax (将来) | `gradient(h, spins, i)` | O(n_atoms · n_sweep) | wall/site, alloc |

**Fixture の選び方**（スケール・SCE 構造をカバー）:
- `bcc_2x2x2` (16 atoms, repeat=(2,2,2)→128, Lf=0 のみ): 軽量・等方
- `fege_2x2x2` (64 atoms, Lf=0/1/2/4): **異方性込み**, 中規模
- `ferh_4x4x4` (128 atoms): 大規模・重い path で検証用

**出力サンプル**:
```
=== Construction (fege_2x2x2, n_atoms=64) ===
  XML parsing            12.3 ms   alloc:  120 KiB
  CG table build          5.1 ms   alloc:   45 KiB
  Hamiltonian assembly   45.7 ms   alloc:  3.2 MiB

=== Energy evaluation ===
  total_energy            2.1 ms   alloc:  18 KiB
  local_energy / site    15.2 µs   alloc:   1.4 KiB
  delta_local_energy     16.0 µs   alloc:   1.5 KiB
  gradient / site        45.1 µs   alloc:   2.1 KiB

=== sweep ===
  metropolis_sweep!       2.1 ms   alloc: 180 KiB
  per-flip cost          16.4 µs

=== vs optimized (ratio = simple / optimized) ===
  total_energy           18.5×
  delta_local_energy     22.0×
  sweep                  19.8×
```

**実装メモ**:
- `BenchmarkTools.@benchmark` を使う。`samples=...`/`evals=...` を fixture サイズで切替。
- alloc を必ず測る（時間より allocation がボトルネックの主因になることが多い）。
- 比較は `simple / optimized` の比率を出す。「2× 程度に収まっていれば simple 版として
  OK」「100× だったら設計の問題」のような判断指標として使う。
- 各 `bench_*.jl` は独立して走らせられるよう、`include("fixtures.jl")` だけで
  完結させる。`runbench.jl` は順次 include して総合表を出す。
- 将来 HMC を実装するときは `gradient` のベンチが指針になる。CMC を入れるなら
  `delta_local_energy_pair` の項目を追加する。

## サンプル (examples/) の方針（決定）

**目的**: 「すぐに試せる・読んで分かる」リファレンスを提供。教材性 (C) を実装側だけでなく
ユーザー入口にも広げる。

**ディレクトリ**: リポジトリ直下の `examples/`（test fixture とは独立）。

```
examples/
├── README.md                        # 各サンプルの目的・期待出力・実行方法・読む順序
├── 01_quickstart.jl                 # bcc_2x2x2 を load → SCEMC 構築 → 数 sweep → E を print
├── 02_cooling_run.jl                # T を高→低でスキャン、E(T)・|M|(T) を CSV 出力
├── 03_anisotropy_demo.jl            # fege_2x2x2 (Lf>0) で M_z の方向選好を観察
├── 04_initial_spin_presets.jl       # :random / :ferromagnetic / SVector / Matrix 全モードのデモ
└── 05_custom_observable.jl          # ferh_4x4x4 で Fe/Rh 副格子磁化を extra_measure callback で追加
```

**方針**:
- 1 ファイル `julia --project=. examples/01_quickstart.jl` で完結。プロット依存は入れない
  （CSV/print 出力のみ、可視化はユーザー側に委ねる）。
- test fixture (`test/bcc_2x2x2`, `test/fege_2x2x2`) の XML を再利用する。重複ファイル無し。
- `README.md` に「30秒で動かす quickstart」「30分かけて読む順序」を併記。01 → 04 →
  02 → 03 → 05 のような難易度順を提示。
- (B') 外場サンプルや HMC サンプルは **初版に含めない**（初版 Metropolis only と整合）。
  `README.md` に「将来追加」と明記して拡張点を見せる。
- 04 番で `:initial_spins` 拡張（案A: Symbol / SVector / Matrix）の全モードをデモ
  → API の幅が一目で分かる。
- 05 番は `ferh_4x4x4` を使う（Fe と Rh の自然な 2 副格子構造）。Fe / Rh それぞれの
  原子インデックス集合は `system.xml` の `<Positions>` から構築する例として書く。
  AFM 反転の有無で `|M_Fe + M_Rh|` と `|M_Fe - M_Rh|` の振る舞いが分かれる、という
  教材ポイントも併記。
- Pluto / Jupyter notebook 形式は **採用しない**（依存・実行環境の壁を増やさない）。
  プレーン `.jl` のみ。将来 Literate.jl で `.jl` → docs ページ化することは視野。

**運用ルール**:
- 各サンプルは冒頭に docstring 風コメントを置き「この例で何が示されるか」「期待出力」
  「想定実行時間」を記述。
- CI で `julia --project examples/01_quickstart.jl` を smoke test として走らせる
  （短時間で済むサンプルに限る）。長時間例（02 cooling run 等）は CI から除外。
- 将来 update を増やすときは `06_hmc_demo.jl` のような番号で追加し、README に追記。
