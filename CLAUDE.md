# CLAUDE.md

## プロジェクトの目的
スピンモデル・有限温度シミュレーションのための科学計算ソフトウェアを開発するリポジトリ。
スタイルのリファクタリングよりも、数値的な正確さ・再現性・物理的な整合性を優先すること。

## 基本ルール
- 数値規約を黙って変更しない。
- アルゴリズムを編集する前に、関連する方程式と現在の符号・単位規約を確認すること。
- 数値結果を変える可能性のある変更には、必ず以下を伴うこと：
  1. 結果が変わる理由の簡潔な説明
  2. リグレッションまたは検証テスト
  3. ユーザー向けの場合はdocs/examplesの更新
- git 操作（push / commit / merge / rebase / reset / checkout など変更を伴うもの）は
  必ずユーザーの確認を取ってから実行する。`.claude/hooks/confirm-git.py` が
  PreToolUse で該当コマンドを検出し確認プロンプトを出す（read-only な status / diff /
  log 等は対象外）。

## 実装規約
- 隠れたグローバル状態を避ける。
- エクスポートされるAPIには明示的な型アノテーションとdocstringを使用する。
- パフォーマンス改善の場合は、変更前後でベンチマークを取ること。

## コードスタイル
- **`src/simple/`, `test/simple/`, `test/parity/`** (および以降の simple-impl 関連の
  新規ディレクトリ) は [SciML Style](https://github.com/SciML/SciMLStyle) に準拠する。
  各ディレクトリに `.JuliaFormatter.toml` (`style = "sciml"`) を置いてあるので、
  実装変更後は `julia --project=. -e 'using JuliaFormatter; format("src/simple")'`
  などで整形する。
- 既存の `src/JPhiMagestyCarlo.jl` / `src/xml_io.jl` / `src/template_energy.jl` /
  `src/spin_utils.jl` は今のところ別スタイルなので一括フォーマットしない。
  将来的にリポジトリ全体を統一する場合は別途決める。

## 言語・用語規約

- ソース・コメント・docstring・コミットメッセージ・PR は英語
  (`feedback_comments_in_english.md`)。`docs/` 配下と会話のみ日本語可。
- **コミットメッセージは [Conventional Commits](https://www.conventionalcommits.org/) に準拠する**。
  形式: `<type>(<scope>): <subject>` (scope は省略可)。
  - 使用する type: `feat` (新機能) / `fix` (バグ修正) / `docs` / `test` / `refactor` /
    `perf` (性能改善) / `chore` (ビルド・補助ツール等) / `style` (整形のみ)。
  - subject は命令形・小文字始まり・末尾ピリオドなし (例: `add SCEMC type`)。
  - 破壊的変更は body に `BREAKING CHANGE: ...` を付ける。
  - **例外**: spec のマイルストーン完了コミット (`M7:` / `M8:` 等の prefix) は
    Conventional Commits の対象外として許容する。spec 外の通常コミットは必ず準拠する。
- **アメリカ英語** で統一する: `normalize` / `normalization` / `behavior` /
  `color` / `center` / `optimize` / `organize` / `favorite` / `canceled` 等。
  British 綴り (`normalise` / `behaviour` 等) を新たに書かない。
- 技術用語のハイフン規則: **"Monte Carlo"** (no hyphen) を採用。"Monte-Carlo" は使わない。
- 例外: **外部 API の literal は元の綴りを保つ**。
  - SpheriCart の `SphericalHarmonics(L; normalisation=:L2)` の kwarg 名 (`normalisation=`)。
  - SpheriCart 内部ファイル名 `normalisations.jl` への参照。
  - その他 upstream パッケージのシンボル名。
  説明文の散文では American spelling を使い、コードリテラルだけ original を保つ。

## テスト

テストは2層に分かれている：
- **通常テスト**（`make test`）: ~2分。Simple/JPhiSpinMC 両実装 + bcc/fege parity + JET 静的解析を含む。日常的な開発で使用する。
- **重い検証テスト**（`make test-slow`）: ~7分。上記に加えて `ferh_4x4x4`（128原子、839 936 cluster instances）での simple↔optimized parity を実行する。アルゴリズム変更時に実行する。

## 物理規約

これらを誤解すると無言でバグが埋まるので、アルゴリズムを触る前に確認すること。

- **温度の単位**: 計算経路 (energy math, Metropolis 採択 `exp(-ΔE/T)`, `register_evaluables` の `T²` など) では `T` は **eV**。`kB` は計算経路には現れない。
  - **API 境界では Kelvin で受け取って変換してよい**。例: `Simple.SCEMC` は `params[:T]::Real` を Kelvin で受け、constructor で `BOLTZMANN_EV_PER_KELVIN ≈ 8.6173e-5 eV/K` を掛けて内部 `mc.T` (eV) に変換する。境界より内側はすべて eV。
  - 既存 `JPhiSpinMC` (optimized) は `params[:T]` を eV で受け取る古い API。Simple とは異なる規約なので注意。
- **スピン行列のレイアウト**: `spins` は `3 × n_atoms`（行 = x,y,z、列 = 原子）。転置すると全計算が壊れる。`spins[:, i]` は **大きさ 1 の古典スピン方向ベクトル**。物理的な磁気モーメント `μ_i = m_i · S_i` は別概念で、`m_i` は `MomentModel` (`UniformMoment` / `PerSiteMoment` 等) が保持する。
- **`:Energy` / `:Magnetization` 観測量は per atom**: `measure!` が記録するのは:
  - `:Energy = E_total / n_atoms` (eV/atom)
  - `:Magnetization = |m|`, `:Magnetization2 = |m|²`, `:Magnetization4 = |m|⁴` ただし `m = (1/n_atoms) Σ_i S_i`。
    - `S_i` は単位ベクトル (古典スピンの**方向**) なので、`|m| ∈ [0, 1]` の **無次元秩序変数**。
    - **磁気モーメントの大きさ `m_i` は含まれない**。副格子で異なるモーメント (Fe vs Rh など) は反映されないし、Bohr magneton 単位の物理磁化 `M_phys = (1/n) Σ_i m_i S_i` とは別物。
    - 実際の `M_phys` や副格子磁化を欲しい場合は `params[:extra_measure]` callback で計算する。

  比熱 (`n · σ²(E) / T²`), Binder 比 (`⟨m²⟩² / ⟨m⁴⟩`), 感受率 (`n · ⟨m²⟩ / T`) の式はいずれもこの per-atom 無次元秩序変数を前提にしている。SI 単位の磁気感受率ではない。
- **球面調和関数は実数（テッサー型）**: `zlm_cache` は複素 `Ylm` ではなく実 `Zlm`。キャッシュの列数は `(l_max+1)²`（`sum_{l=0}^{L}(2l+1) = (L+1)²` による全(l,m)ペアの合計）。
- **`Φᵥ` の定義はMagesty.jl側**: SALCの構成・CG係数の規約はこのリポジトリに書かれていない。変更前は必ず[Magesty.jl technical notes](https://Tomonori-Tanaka.github.io/Magesty.jl/technical_notes/)を参照すること。

## 設計メモ

- 進行中の開発単位は [`docs/specs/[YYMMDD]-[slug]/`](docs/specs/) に置く
  (requirements.md / design.md / tasklist.md の 3 ファイル構成)。
- 横断的なメモ・保留中のアイデア・optimized版の future-work は
  [`docs/design_notes.md`](docs/design_notes.md)。
- 格子・セル・クラスターの用語定義は [`docs/terminology.md`](docs/terminology.md)。

### spec フォルダの運用ルール

**中規模以上の開発を始める前に、必ず先に spec フォルダを作って合意を取る。**

判定基準 (どれかに当てはまったら spec を作る):
- 数日以上かかる
- 設計判断が複数ある
- 後から「なぜこう作った?」と訊かれそう
- 既存の挙動が変わる中規模以上の変更 (新機能追加 / 中規模リファクタリング)

spec を作らなくてよいもの:
- バグ修正 (テスト追加で完結)
- ドキュメント・コメント修正
- 1 ファイル内の小規模 refactor
- 既存テストで担保される動作の小修正

手順 (Claude 側で実施):
1. `docs/specs/[YYMMDD]-[slug]/` を作る (`YYMMDD` = `date +%y%m%d`、`slug` は英語の kebab-case)。
2. 同フォルダに以下 3 ファイルを置き、user と相談しながら埋める:
   - `requirements.md` — 目的・不変条件・スコープ・完了基準
   - `design.md` — モジュール構成・API・型・規約
   - `tasklist.md` — マイルストーン (粗い粒度。日々の細かい作業は TaskCreate)
3. spec の合意ができてから実装に着手。
4. 完了後もフォルダは残す (削除しない、履歴として参照)。

参考例: [`docs/specs/260512-simple-impl/`](docs/specs/260512-simple-impl/)。

## 設計判断

### j0（定数エネルギー項）
`jphi.xml` の `ReferenceEnergy`（`j0`）は読み込まない。
理由：本パッケージはMCサンプリング専用であり、`ΔE` のみが重要なため定数項は不要。
絶対エネルギーが必要な呼び出し側は XML から直接 `ReferenceEnergy` を読むこと。

## 連動箇所（一方を変えたら全箇所を確認）

### タイリングロジック
タイル座標→原子インデックス変換のコアは `_foreach_translated_instance` に集約されている。
`_build_cluster_instances` がこのヘルパーを使う。`build_local_energy_template` は
ti=tj=tk=0 限定の `_foreach_base_instance` を使う（並進はオンザフライ）。

`coupled_cluster_energy`（リファレンスpath）は独立した実装を持つ。タイリングロジックを
変更する場合は `_foreach_translated_instance` と `coupled_cluster_energy` の2箇所を同期すること。

### エネルギーカーネルの2パス整合性
`:tensor_template`（デフォルト）と `:tensor`（リファレンス）は別コードパスだが数値結果は
一致しなければならない。変更が必要な箇所：`init!` 内の `mc.energy` 初期化（両ブランチ）、
`sweep!` 内のΔE計算。片方だけ変えると、カーネル切り替え時にサイレントに結果が変わる。

### Observablesのper atom規約
`measure!` は `:Energy = mc.energy / n_atoms`（per atom）で記録する。
`register_evaluables` の比熱・感受率の式はこれを前提にしている（例: `n * (⟨E²⟩ - ⟨E⟩²) / T²`）。
`measure!` の `/n` を変えると比熱・感受率が壊れる。

## Claudeの作業方針

### 確認不要（自由にやってよい）
- バグ修正（最小限の変更 + テスト追加）
- テストの追加・修正
- ドキュメントの誤記修正

### サブエージェントの活用
- 実装後は `test-runner` エージェントでテストを実行・診断する。
- コミット前は `code-reviewer` エージェントで変更差分をレビューする。

### 提案してから実装する
- アルゴリズムの変更（数値結果が変わる可能性があるとき）
- リファクタリング（タイリングロジックの集約など）
- パフォーマンス改善（ベンチマーク結果を先に示す）

### 実装せず、必ず確認する
- 物理規約の変更（符号・単位・規格化）
- 新しい外部依存の追加
- 観測量の定義や式の変更（`measure!` / `register_evaluables`）
- 低レベルカーネルの書き換え（`_tensor_contract_instance*` など）
