# Requirements — General supercell (任意 3×3 整数行列 M)

開始: 2026-06-20。

> 関連 spec: [design.md](design.md) / [tasklist.md](tasklist.md)

## 目的

現在の Simple 実装は `jphi.xml` の「base cell」(= Magesty 訓練スーパーセル) を
分割不能なタイルとして扱い、`repeat::NTuple{3,Int}` による**対角整数倍**しか
作れない。訓練セルが primitive の 2×2×2 なら、MC は 2×2×2 / 4×4×4 / … しか
組めず、3×3×3 や任意サイズ・斜め (非対角) セルは生成できない。

スピン相互作用の基本単位を「base cell + 対角 repeat」から
「**primitive cell + 一般 supercell 行列 `M` (3×3 整数)**」へ一般化し、
Sunny.jl と同等の任意 commensurate スーパーセル (非対角 `M` を含む) を
Simple 実装で生成できるようにする。

鍵となる事実:
- `jphi.xml` は既に訓練セル内の並進対称性 (`SystemData.map_sym`, `n_trans`) を
  持つ。これは「訓練セル = primitive × n_trans」を意味し、**primitive セルを
  既存 XML から抽出できる** (Magesty 側の変更は不要)。Magesty
  `src/SunnyExport.jl` の `_sunny_primitive` が同型の抽出を行っている。
  - `n_trans = base_n` の結晶 (bcc 等の Bravais) は `n_prim = 1`。
  - basis を持つ結晶 (FeGe / FeRh 等) は `n_prim = base_n / n_trans > 1`。
- エネルギー計算 (`src/simple/energy.jl`)・観測量 (`mc.jl`)・CG (`cg.jl`) は
  `ClusterInstance` の atom index しか参照しないので**不変**。改修は
  instance 生成と型/コンストラクタ配線に限定される。

## 不変条件 (絶対に守る)

- **物理規約は一切変更しない**: スピンレイアウト `3 × n_atoms`、温度の単位
  (eV)、per-atom 観測量、比熱・感受率の式、実テッサー `Zlm`、`Φᵥ` 定義 —
  すべて据え置き。変更するのは instance の **atom 付番生成のみ**。
- **`M = diag` (旧 `repeat` 相当) では既存挙動を bit-exact に保つ**。
  - `repeat=(n1,n2,n3)` 指定時、新パイプラインは旧 `_generate_instances` と
    **同一の instance 集合・同一の atom 付番**を生成する。
  - 既存テスト・parity テスト・`init_spins`・PT serialization・ユーザの
    `extra_measure` 副格子計算が**無改変**で動く。
- **incommensurate は対象外**: 周期 MC では原理的に primitive と整合な
  (commensurate な) 周期セルしか作れない。`M` は整数行列 (`det(M) ≠ 0`) に限る。
- **非対角 `M` のみ純 primitive 付番**: `super_index(cell_id, subl) =
  subl + n_prim*(cell_id-1)`。これは新機能であり既存挙動とは無関係。
- **Optimized 実装は本 spec のスコープ外**: `src/JPhiMagestyCarlo.jl` /
  `src/template_energy.jl` は対角のまま据え置く。Simple での検証後に別 spec。

## スコープ

### 含む

- `src/simple/supercell.jl` (新規): 整数線形代数 (3×3 の det / adjugate /
  Hermite 正規形 / modular wrap)、primitive 抽出 (`PrimitiveCell` /
  `extract_primitive`)、クラスタの primitive テンプレート化
  (`ClusterTemplate`)、一般 `M` タイリング (`_generate_instances_matrix`)。
- `src/simple/types.jl`: `SpinClusterHamiltonian` を `supercell_matrix` 対応に
  改修。旧 `_supercell_atom_index` / `_generate_instances` は新パイプライン
  経由に置き換える (または M=diag 互換層として保持)。
- 新 param `:supercell_matrix::AbstractMatrix{<:Integer}` (3×3)。
  `Simple.SCEMC` は `:repeat` (対角ショートカット) と `:supercell_matrix`
  (一般 `M`) の2つのみを受け付け、`:repeat` は `M=diag` の特殊ケースとして
  内部で包含する。両方指定は `ArgumentError`。
  - 注: optimized 側 `JPhiSpinMC` が持つ `:supercell` (対角 alias) は本 spec の
    スコープ外。`SCEMC` には `:supercell` を追加しない。
- `src/simple/mc.jl`: `SCEMC` の param 解決・フィールド・`init!`・
  `register_evaluables` を新付番/互換層に追従。
- `src/simple/spin_proposal.jl`: `init_spins` / base-tiling を付番互換および
  一般 `M` に追従。
- 整合性チェックとエラー: `det(M) == 0`、非整数、全 SALC drop 等。
- テスト (`test/simple/test_simple_supercell.jl`):
  - 整数線形代数の単体テスト (HNF の `M=H*U`、wrap が代表 offset に落ちる)。
  - primitive 抽出の往復一致 (bcc / fege / ferh fixture)。
  - **M=diag が旧 `repeat` と instance 集合・atom 付番で bit 等価**。
  - 非対角 `M`: 並進不変性、`total_energy ∝ |det(M)|`、対角等価 `M'` 一致。
- ドキュメント: `SpinClusterHamiltonian` / `SCEMC` docstring、
  `docs/terminology.md` の supercell 定義に一般 `M` の節を追加。

### 含まない

- Optimized 実装 (`src/JPhiMagestyCarlo.jl` / `template_energy.jl`) の一般化。
- `jphi.xml` 側の編集・出力。読み取り時の再構成のみ。
- 観測量 (`measure!` / `register_evaluables`) の定義・式の変更。
- パフォーマンス最適化 (Simple は元々リファレンス実装。正しさ優先)。
- incommensurate / 非整数スーパーセル。

## 完了基準

- `make test` (~2分) と `make test-slow` (~7分) が、既存テスト・parity を
  含めて全 pass (M=diag 互換層で付番不変)。
- 新規テストで非対角 `M` の並進不変性・スケーリング・対角等価が確認できる。
- `Simple.SCEMC` で `params[:supercell_matrix]` が効く。`:repeat` も従来通り。
- 既存 JET 静的解析が新コードを通過する。
- `code-reviewer` / `numerical-reviewer` レビュー pass。
