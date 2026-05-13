# Requirements — `src/simple/` Reference Implementation

開始: 2026-05-12。

> 関連 spec: [design.md](design.md) / [tasklist.md](tasklist.md)

## 目的

`SpinClusterMC.jl` の SCE (Spin-Cluster Expansion) を扱う 2 つ目の実装を、
**読みやすさ・拡張容易性・教材性**を最優先に書き直す。既存の `JPhiMagestyCarlo.jl`
(~1600行) は最適化機構が密に絡んで構造が追いにくいため、それと**並列に**
`src/simple/` を置き、CI で常に数値整合性を確認する。

- **主目的 (B)**: スピン更新アルゴリズム拡張の足場
  (CMC, HMC, overrelaxation, heatbath, Wolff, replica-exchange variants など)。
- **副目的 (C)**: Magesty.jl の用語・構造との対応を取り、コードを Magesty docs と
  並べて読める教材とする。

## 不変条件 (絶対に守る)

スピンクラスター展開の強みは「ハミルトニアンの形を事前に決めず、`l_max` と `N-body`
だけ変えれば任意のスピンハミルトニアンを表現できる」点。simple 版でもこの拡張性は
犠牲にしない。

- 任意の `l_max` を受け取れること。
- 任意の N-body (1体, 2体, 3体, ...) を受け取れること。
- ハミルトニアンの「形」は `jphi.xml` から駆動され、コード側に固定しない。
- → テンソル縮約のループは N-body に対して可変長で書く必要がある
  (現実装の `_tensor_contract_instance` と同じ性質を保つ)。
  l_max 固定 / N-body 固定の特殊化路線は不可。

## スコープ

### 含む (初版 v1)

- jphi.xml の独立 parser
- 型: `SpinClusterHamiltonian`, `ClusterInstance`, `CGTable`
- エネルギー API: `total_energy / local_energy / delta_local_energy / gradient`
- 外場 (`ExternalTerm` 抽象 + Zeeman 例)
- Spin 提案: `:initial_spins` の Symbol/Tuple/SVector/Matrix 拡張, geodesic proposal,
  周期的 renormalization
- MC 型 `SpinClusterMC.Simple.SCEMC` + Carlo glue (init!/sweep!/measure!/register_evaluables)
- Metropolis single-spin flip update のみ
- ユーザー観測量 callback (`params[:extra_measure]`, `params[:extra_evaluables]`)
- ready-to-run サンプル `examples/01_*.jl` … `examples/05_*.jl`
- `benchmark/simple/` (BenchmarkTools-based)

### 含まない (初版 v1 では切る、後付け可能な field 配置は維持)

- Parallel Tempering (PT)
- MPI 分散ビルド
- グローバル辞書キャッシュ
- Template fast path / Instance precompute / zlm cache
- HDF5 checkpoint 拡張 (Carlo デフォルトに任せる)
- HMC, overrelaxation, heatbath, Wolff, CMC update
  (`updates/` ディレクトリに後続実装が並ぶ構造のみ用意)

## 制約

- **物理規約** (`CLAUDE.md`):
  - 温度 `T` は eV。`kB` は呼び出し側で変換。
  - スピン行列レイアウトは `3 × n_atoms`。
  - `:Energy` 観測量は per atom (`E / n_atoms`)。
  - 球面調和関数は実 (tesseral) Zlm、NOT 複素 Ylm。
- **観測量の規約 (既存 `JPhiSpinMC` と揃える)**:
  - 平均スピンベクトル `m = sum(spins) / n_atoms`、その大きさ `|m| = norm(m)`。
  - `measure!` で記録する 4 観測量:
    - `:Magnetization`: `|m|`
    - `:AbsMagnetization`: `|m|` (PT/post-processing 互換のため同値を別キーで)
    - `:Magnetization2`: `|m|²`
    - `:Magnetization4`: `|m|⁴`
  - `:Energy` / `:Energy2` も per-atom (`E / n_atoms`, `(E / n_atoms)²`) で記録。
  - `register_evaluables` で派生量を出す:
    - `:SpecificHeat = n_atoms · (⟨E²⟩ - ⟨E⟩²) / T²` (`:Energy`, `:Energy2` 入力)
    - `:BinderRatio = ⟨m²⟩² / ⟨m⁴⟩`
    - `:Susceptibility = n_atoms · ⟨m²⟩ / T`
  - 副格子磁化等の系依存観測量は `params[:extra_measure]` / `params[:extra_evaluables]`
    callback でユーザー側が追加 (design.md `measure! の拡張ポリシー` 節)。
- **依存**: 親パッケージの依存 (Carlo, MPI, HDF5, EzXML, StaticArrays) を共有。
  Zlm 計算は SpheriCart を直接使用。CG 計算は `Magesty.AngularMomentumCoupling`
  に依存 (tesseral 位相補償付き Racah CG)。それ以外の Magesty 機能には依存しない。
- **言語**: ソース・docstring・コミット・PR は英語 (`feedback_comments_in_english.md`)。
  `docs/` 配下のみ日本語可。

## 完了基準

### 機能完了

- [x] `using SpinClusterMC.Simple` で公開 API がロードできる (M1, 2026-05-12)
- [x] `examples/01_*.jl` … `examples/05_*.jl` が `julia --project=. examples/0N_*.jl`
      でエラーなく走る (M8, 2026-05-12; CI smoke via `make examples-smoke`)
- [x] 既存 `JPhiSpinMC` と新 `Simple.SCEMC` が `bcc_2x2x2`, `fege_2x2x2`,
      `ferh_4x4x4` 全てで `make test` / `make test-slow` を通過
      (M10, 2026-05-13)

### 数値整合性 (vs `JPhiMagestyCarlo`)

| 比較対象 | 規約 | 状態 |
|---|---|---|
| `total_energy` | `rtol = 1e-8` | ✅ bcc/fege/ferh parity 通過 |
| `local_energy` | `rtol = 1e-10` | (間接) optimized に公開 `local_energy` がないため `delta_local_energy` 経由で間接検証 |
| `delta_local_energy` (ΔE) | `rtol = 1e-7` | ✅ bcc/fege/ferh parity 通過 |
| Zlm 出力 (`compute(sph, S)`) | `atol = 0` (bit-exact) | ✅ 両実装が同じ SpheriCart `SphericalHarmonics` を共有するため bit-exact |
| `:Magnetization` 系列 (同 seed, 同 sweeps) | `rtol = 1e-8` | 未実施 (ΔE が parity していれば同 RNG 軌道は一致するはずだが、明示的 trajectory 比較テストは未追加) |
| `:SpecificHeat` / `:BinderRatio` / `:Susceptibility` (同 seed, 同 sweeps) | `rtol = 1e-7` | 同上 |

これらは既存テストの規約 (test/runtests.jl:144, 165, 194, 255) と整合させる。
bit-exact なエネルギー一致は要求しない (両実装で和の集約順序が異なるため)。

trajectory parity の明示テストは future-work メモ ([design_notes.md](../../design_notes.md))
に積む。ΔE parity (`rtol=1e-7`) が通っているので、同 seed の Metropolis 軌跡は
acceptance 判定段階まで一致しうるが、丸め誤差による発散点の確認は別タスク。

### 教材性チェック

- [x] `src/simple/` 配下だけ読めば実装が理解できる (XML parser・型を含む独立した実装)
- [x] 数式を実装する関数 (`cluster_energy`, `local_energy`, ...) の docstring に
      LaTeX 数式 + Magesty docs / SCE 論文への参照 (M4)
- [x] `examples/` README に「30 秒で動かす」「30 分かけて読む順序」併記 (M8)

### ベンチマーク

- [x] `benchmark/simple/` が `benchmark/optimized/` と同じ fixture
      (`bcc_2x2x2`, `fege_2x2x2`, `ferh_4x4x4`) で走り、比率
      `simple / optimized` を出力する (M9, `bench_compare.jl`)
- [ ] 性能比 `simple / optimized` が ~ 10〜30× に収まる
      → **未達**。`bench_compare.jl` の実測では fege `total_energy` で alloc 比
      1.77e+06× / 時間比 10³ 程度 (Simple が SH cache を毎回再生成するため)。
      ボトルネックは特定済み (`design_notes.md` "SphericalHarmonics の使い回し")
      なので、構造由来の遅さの分布が解析可能というゆるい基準は満たす。
      数値の改善は別 spec で追跡する。

## 参考文献

- SCE 論文: <https://arxiv.org/abs/2512.04458>
- Magesty.jl technical notes: <https://Tomonori-Tanaka.github.io/Magesty.jl/technical_notes/>
- Spin HMC: <https://arxiv.org/abs/1902.02116>
- Constrained MC (Asselin et al.): <https://arxiv.org/abs/1006.3507>
- スピン配列レイアウト調査: [../../spin_layout_survey.md](../../spin_layout_survey.md)
