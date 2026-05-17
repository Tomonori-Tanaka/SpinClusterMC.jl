# Tasklist — Simple Zlm Workspace

開始: 2026-05-13。

> 関連 spec: [requirements.md](requirements.md) / [design.md](design.md)

この tasklist は**マイルストーン単位**で粒度を粗く保つ。日々の細かい作業は
Claude Code 内蔵の TaskCreate で管理し、ここには反映しない。完了したマイル
ストーンは `- [x]` でチェックし、完了日を併記する。

## テストファイルの配置規約

- `test/simple/` — Workspace の Simple 版固有テスト
- `test/parity/` — 既存 simple vs optimized 数値一致テスト (本 spec では無修正で
  全 pass を維持することが完了基準。新規 parity テストは追加しない)
- `test/runtests.jl` の既存構造をそのまま使う (slow gate は `ferh_4x4x4` のみ)

## ベースライン参照

- `benchmark/results/260513-cf25754-darwin-arm64-M10-baseline.md` (現状の Simple
  実装のベンチ。本 spec の "after" はこれと並べて比較する)

## マイルストーン

### W1. `SpinClusterWorkspace` 型 + Layer 2 化 (pure buffer)
- [ ] `src/simple/workspace.jl` を新規作成。ファイル先頭 10〜20 行に design 全体
      の rationale を英語で記載 (なぜ Workspace 引数か / なぜ Hamiltonian に
      持たせないか / Layer 1/2/3 の関係 / Optimized 側との対応)
- [ ] `SpinClusterWorkspace` 構造体 + コンストラクタ `SpinClusterWorkspace(h)`
- [ ] `_compute_zlm_all!` (in-place 版) を `src/simple/energy.jl` に追加。既存の
      `_compute_zlm_all` は in-place 版を呼ぶラッパとして残す (Layer 1 互換のため)
- [ ] Layer 2 API 4 関数:
  - [ ] `total_energy(h, spins, ws)`
  - [ ] `local_energy(h, spins, i, ws)`
  - [ ] `delta_local_energy(h, spins, i, S_new, ws)`
  - [ ] `gradient(h, spins, i, ws)`
- [ ] Layer 1 API 4 関数を Layer 2 のラッパに変更
  (`f(h, spins, ...) = f(h, spins, ..., SpinClusterWorkspace(h))`)
- [ ] Layer 1 の各 docstring に "bufferless overload is not used by the Monte
      Carlo hot path; educational / testing reference" の `# Notes` 節を追記
- [ ] Layer 2 の各 docstring に `# Preconditions` / `# Postconditions` / `# Mirrors`
- [ ] `src/simple/Simple.jl` で `workspace.jl` を include、`SpinClusterWorkspace`
      を export
- [ ] テスト: `test/simple/test_simple_workspace.jl` (構築 / shape / 型のスモーク)
- [ ] テスト: `test/simple/test_simple_workspace_layer2.jl` (Layer 1 vs Layer 2 で
      `total_energy` / `local_energy` / `delta_local_energy` / `gradient` が
      bit-exact 一致 — 同じ算術を同じ順序で実行しているので `==` 比較)
- [ ] 既存 `make test` / `make test-slow` で全 pass

### W2. `_instance_energy_with_column` 内部用変種
- [ ] `_instance_energy_with_column(inst, zlm, swap_col, swap_col_data, cg_table)`
      を `src/simple/energy.jl` に追加 (internal、export しない)
- [ ] 数値検証: 任意の `inst` と `zlm`, `i` で
      `_instance_energy(inst, zlm, cg) == _instance_energy_with_column(inst, zlm, i, zlm[:, i], cg)`
      が **bit-exact** で成立 (テストで全 instance を走査)
- [ ] テスト: `test/simple/test_simple_instance_with_column.jl` (上記 bit-exact 検証
      + `swap_col` が `atoms[k]` に含まれない場合は `_instance_energy` と同値)

### W3. Layer 3 (stateful cache API)
- [ ] `sync_workspace!(ws, h, spins)`
- [ ] `delta_local_energy_cached!(h, spins, i, S_new, ws)` (`_instance_energy_with_column`
      を使い、E_old / E_new を 1 つの zlm から取り出す)
- [ ] `commit_workspace_column!(ws, i, S_new)` (現状 no-op)
- [ ] `restore_workspace_column!(ws, i)`
- [ ] 各関数の docstring に `# Preconditions` / `# Postconditions` / `# Mirrors`
- [ ] テスト: `test/simple/test_simple_workspace_cached.jl`
  - [ ] `sync_workspace!` 後の `delta_local_energy_cached!` と
        Layer 1 `delta_local_energy` の値が `rtol = 1e-14` で一致 (複数の seed と原子)
  - [ ] accept 経路: `delta_local_energy_cached!` 後に
        `commit_workspace_column!` + `spins[:, i] = S_new` で workspace 不変条件
        (`ws.zlm[:, a] == _zlm(spins[:, a])`) が全列で保たれる
  - [ ] reject 経路: `delta_local_energy_cached!` 後に
        `restore_workspace_column!` で `ws.zlm` が完全に元に戻る (bit-exact)

### W4. `SCEMC` 統合
- [ ] `SCEMC` 構造体に `ws::SpinClusterWorkspace` field を追加 (コンストラクタで生成)
- [ ] `init!`: `sync_workspace!(mc.ws, mc.h, mc.spins)` を `_full_energy` の前に追加
- [ ] `_full_energy`: Layer 2 (`total_energy(h, spins, ws)`) を使うよう変更
- [ ] `metropolis_sweep!`: `delta_local_energy_cached!` + accept/reject 分岐で
      `commit_workspace_column!` / `restore_workspace_column!`
- [ ] `_renorm_and_drift_check!`: spin の renormalize 直後に
      `sync_workspace!` を呼ぶ
- [ ] `make test`: 既存 parity (`bcc_2x2x2`, `fege_2x2x2`) が無修正で全 pass
- [ ] `make test-slow`: `ferh_4x4x4` parity が無修正で全 pass
- [ ] 同 seed の旧実装 (W1 完了時点) と最終 `mc.energy` が `rtol = 1e-12` で一致
      (新規テスト `test/simple/test_simple_workspace_sweep.jl`)

### W5. ドキュメント仕上げ
- [ ] `docs/src/api.md` の Simple Module セクションに以下を追記:
  - [ ] `SpinClusterWorkspace`
  - [ ] Layer 2 オーバーロード (`total_energy(h, spins, ws)` 等)
  - [ ] Layer 3 API (`sync_workspace!` / `delta_local_energy_cached!` /
        `commit_workspace_column!` / `restore_workspace_column!`)
- [ ] `docs/design_notes.md` の "SphericalHarmonics の使い回し" 項目を「完了
      (2026-05-XX)」に更新し、本 spec へのリンクを追加
- [ ] `docs/specs/260512-simple-impl/requirements.md` の未達基準
      (性能比 10〜30×) の状態を「本 spec で改善 → 別 spec で残りの inner-loop
      alloc を扱う」に更新
- [ ] CLAUDE.md の "連動箇所" 節を確認し、Workspace 追加で増えた連動箇所
      (`mc.spins` 直書き → `sync_workspace!`) を追記

### W6. ベンチマーク + 完了 commit
- [ ] `benchmark/simple/bench_sweep.jl` (新規 or 既存に追記): `SCEMC` を 1 sweep
      回す BenchmarkTools サンプルを追加 (bcc / fege)
- [ ] `benchmark/simple/bench_compare.jl` を `SpinClusterWorkspace` 経由に更新
      し、M10 baseline と並べて時間 / alloc を比較
- [ ] `benchmark/results/<YYMMDD>-<sha>-darwin-arm64-workspace-after.md` を
      `git add -f` でコミット (M10 baseline と同じ命名規約)
- [ ] サマリ: alloc/call 削減比 (期待: bcc sweep で >100×、fege sweep で同程度)
      を after ファイルの headline に明記
- [ ] commit メッセージは Conventional Commits の `perf:` または `feat:` (新規
      API 追加が主なら `feat:`)

## Out of scope (別 spec 推奨)

- `_instance_energy` の inner-loop allocation 除去 (`Iterators.product` の tuple
  alloc 等)。fege `total_energy` の 84M / 1.76 GiB の主因はこちら。
- `gradient_all(h, spins, ws)` 系の vector-valued API (HMC / Langevin 用)。
- Workspace の thread-local 化 / 並列対応。
- 列レイアウト (`zlm[k, atom]` ↔ `zlm[atom, k]`) の最適化。Optimized 側と異なる
  ことの整合は別問題として残す。

## 完了条件 (本 spec 終了の判断)

すべての W1〜W6 が `- [x]` になり、かつ:

- `make test` / `make test-slow` で全 pass
- `benchmark/results/*-workspace-after.md` がコミット済み
- `docs/src/api.md` の Simple Module セクションが Workspace を含む
- このフォルダ (`docs/specs/260513-simple-zlm-workspace/`) は保持 (履歴として残す)
