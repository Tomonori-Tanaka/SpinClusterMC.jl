# CLAUDE.md

## Project mission
This repository develops scientific computing software for electronic-structure / spin-model / finite-temperature simulations.
Prioritize numerical correctness, reproducibility, and physical consistency over stylistic refactors.

## Core rules
- Never change numerical conventions silently.
- Before editing algorithms, identify the relevant equations and current sign/unit conventions.
- Any change that can alter numerical results must be accompanied by:
  1. a brief explanation of why results may change,
  2. a regression or validation test,
  3. an update to docs/examples if user-facing.
- Do not push to github

  ## Implementation conventions
  - Avoid hidden global state.
  - Use explicit type annotations/docstrings for exported APIs.
  - For performance work, benchmark before and after.

  ## Testing

  テストは2層に分かれている：
  - **通常テスト**（`make test`）: ~10秒。日常的な開発で使用する。
  - **重い検証テスト**（`make test-slow`）: ~4分。`ferh_4x4x4`（128原子）を使った数値精度の検証。アルゴリズム変更時に実行する。

  ## Physics conventions

  これらを誤解すると無言でバグが埋まるので、アルゴリズムを触る前に確認すること。

  - **温度の単位**: `T` はeV。コード中に `kB` は現れない。Metropolis採択確率は `exp(-ΔE/T)` をそのまま使う。ケルビンから変換する場合は呼び出し側で `T_eV = kB * T_K` を行う。
  - **スピン行列のレイアウト**: `spins` は `3 × n_atoms`（行 = x,y,z、列 = 原子）。転置すると全計算が壊れる。
  - **`:Energy` 観測量は per atom**: `measure!` が記録する `:Energy` は `E / n_atoms`。比熱・感受率の式もこれを前提にしている。
  - **球面調和関数は実数（テッサー型）**: `zlm_cache` は複素 `Ylm` ではなく実 `Zlm`。キャッシュの列数は `(l_max+1)²`（`sum_{l=0}^{L}(2l+1) = (L+1)²` による全(l,m)ペアの合計）。
  - **`Φᵥ` の定義はMagesty.jl側**: SALCの構成・CG係数の規約はこのリポジトリに書かれていない。変更前は必ず[Magesty.jl technical notes](https://Tomonori-Tanaka.github.io/Magesty.jl/technical_notes/)を参照すること。

  ## Design decisions

  ### j0 (constant energy term)
  `sce_energy`、`monomial_sce_energy`、`mc.energy` はいずれも `j0` を含まない。
  理由：本パッケージはMCサンプリング専用であり、`ΔE` のみが重要なため定数項は不要。
  絶対エネルギーが必要な場合は呼び出し側で `h.j0 * prod(h.repeat)` を加算すること。

  ## Coupled code locations (change one → check all)

  ### タイリングロジック（3箇所同期必須）
  スーパーセルのタイル座標→原子インデックス変換が以下の3箇所に独立して実装されている：
  - `_build_cluster_instances` — fast path用クラスターインスタンス構築
  - `build_monomial_table` — monomial kernel用テーブル構築
  - `coupled_cluster_energy` — リファレンスpath

  どれか1つを変えたら残り2つも必ず同期すること。

  ### エネルギーカーネルの2パス整合性
  `:tensor` と `:monomial` は別コードパスだが数値結果は一致しなければならない。
  変更が必要な箇所：`init!` 内の `mc.energy` 初期化（両ブランチ）、`sweep!` 内のΔE計算。
  片方だけ変えると、カーネル切り替え時にサイレントに結果が変わる。

  ### Observablesのper atom規約
  `measure!` は `:Energy = mc.energy / n_atoms`（per atom）で記録する。
  `register_evaluables` の比熱・感受率の式はこれを前提にしている（例: `n * (⟨E²⟩ - ⟨E⟩²) / T²`）。
  `measure!` の `/n` を変えると比熱・感受率が壊れる。

  ## Safe workflow for Claude

  ### 確認不要（自由にやってよい）
  - バグ修正（最小限の変更 + テスト追加）
  - テストの追加・修正
  - ドキュメントの誤記修正

  ### 提案してから実装する
  - アルゴリズムの変更（数値結果が変わる可能性があるとき）
  - リファクタリング（タイリングロジックの集約など）
  - パフォーマンス改善（ベンチマーク結果を先に示す）

  ### 実装せず、必ず確認する
  - 物理規約の変更（符号・単位・規格化）
  - 新しい外部依存の追加
  - 観測量の定義や式の変更（`measure!` / `register_evaluables`）
  - 低レベルカーネルの書き換え（`_tensor_contract_instance*` など）