# Requirements — `jphi_threshold`: drop near-zero SALCs

開始: 2026-05-19。

> 関連 spec: [design.md](design.md) / [tasklist.md](tasklist.md)

## 目的

Magesty.jl 側でスパースモデリングによる SCE フィットを導入したため、`jphi.xml`
中の `<JPhi>` には絶対値が極めて小さい (実質ゼロの) 係数 `J_s` が含まれる。
モンテカルロ計算ではこれらを Hamiltonian から落としても結果は変わらない一方、
ループ回数 (`length(instances)` / `length(salc_list)`) は減らせるため
sweep 時間を短縮できる。

絶対値しきい値 `jphi_threshold` (eV) を導入し、`|J_s| < jphi_threshold` の SALC を
Hamiltonian 構築時に drop する。

## 不変条件 (絶対に守る)

- **デフォルト挙動は完全に既存と一致**。`jphi_threshold = 0.0` (デフォルト) で
  既存テスト・パリティテストは bit-exact に通る (`abs(J_s) < 0.0` は常に false
  なので何も filter されない)。
- **数値結果は drop した SALC を含めた場合の近似**。drop によって `ΔE` が変わる
  可能性があるため、ユーザーが明示的に `jphi_threshold > 0` を指定した場合のみ
  filter する。
- **物理規約は変更しない**: 温度の単位 (eV)、スピンレイアウト、per-atom 観測量、
  比熱・感受率の式、Φᵥ 定義 — すべて据え置き。
- **しきい値は eV の絶対値**: 相対しきい値 (`max(|J|)` 比) は今回は実装しない。
  スパースモデリング側から得られる「ノイズフロア」を直接渡せるようにするため。

## スコープ

### 含む

- `Simple.SpinClusterHamiltonian` に `jphi_threshold::Real` kw 引数を追加し、
  `_generate_instances` で `abs(J) < jphi_threshold` の SALC を skip。
- `JPhiMagestyCarlo.load_sce_hamiltonian` (optimized) に同じ kw 引数を追加し、
  `salc_list` と `jphi` を同期して filter。
- `Simple.SCEMC` / `JPhiSpinMC` の `params[:jphi_threshold]` 経由で渡せるよう plumb。
  - optimized 側のキャッシュ (`_HAM_CACHE` / `_ECACHE_CACHE` / `_DERIVED_CACHE`)
    のキーに threshold を追加 (異なる threshold が衝突しないように)。
  - `JPhiSpinMC` の MPI serialization にも `jphi_threshold` を含める
    (PT gather 後の rebuild で同じ Hamiltonian を再構成するため)。
- 構築時のログ: drop 数と最大 drop された `|J|` を `@info` で出す。
- エラー: しきい値で全 SALC が落ちる場合は `ArgumentError` を投げる。
  (silent zero-energy MC を避ける。)
- テスト:
  - `threshold = 0.0` で既存と完全一致 (Simple / optimized 両方)。
  - `threshold` を `J` 範囲内に設定 → `length(instances)` が減ること、
    `total_energy` が threshold = 0 の値とは異なること。
  - `threshold` を全 `|J|` の max より大きくする → `ArgumentError`。

### 含まない

- 相対しきい値 / 自動しきい値推定。
- 観測量 (`measure!` / `register_evaluables`) の変更。
- パフォーマンスベンチマークの自動化 (手動測定はする)。
- `jphi.xml` 側の編集 / 出力。あくまで読み取り時の filter。

## 完了基準

- 既存テスト (`make test`) と parity テスト (`make test-slow` も含む) が
  デフォルト (`threshold = 0`) で全 pass。
- 新規テストで `threshold > 0` の filter 挙動が確認できる。
- `Simple.SCEMC` / `JPhiSpinMC` のどちらでも `params[:jphi_threshold]` が効く。
- `code-reviewer` レビュー pass。
