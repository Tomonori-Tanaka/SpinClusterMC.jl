# Requirements — Simple Zlm Workspace

開始: 2026-05-13。

> 関連 spec: [design.md](design.md) / [tasklist.md](tasklist.md)
> 前 spec: [`260512-simple-impl/`](../260512-simple-impl/) (完了)
> ベンチマーク前値: [`benchmark/results/260513-cf25754-darwin-arm64-M10-baseline.md`](../../../benchmark/results/260513-cf25754-darwin-arm64-M10-baseline.md)

## 目的

`src/simple/` の MC hot path (`metropolis_sweep!` 経由の `delta_local_energy`)
で観測される 2 つの非効率を解消する。

1. **(A) `n_atoms²` 浪費**: 1 trial で 1 原子しか動かないのに、`Zlm` キャッシュを
   全 `n_atoms` 列まるごと再計算している。`delta_local_energy` 内の
   `_compute_zlm_all` 呼び出しが原因。
2. **(B) `Matrix{Float64}(K, n_atoms)` / `SphericalHarmonics(max_l)` の毎回 alloc**:
   関数呼び出しごとに作って捨てている。

並行して、将来の HMC / Langevin 系 (全 spin 同時更新) で `gradient` を hot path に
する余地を残すため、**Workspace 引数を渡せる API レイヤ** を導入する。

## 不変条件 (絶対に守る)

- 公開 API (`total_energy`, `local_energy`, `delta_local_energy`, `gradient`) の
  既存 signature を維持する (Workspace なし版はデフォルトでバッファを内部生成)。
- 公開 API の**数値結果は変更しない**。順序由来の丸め以外の差を出さない。
  - `total_energy`: `rtol = 1e-12` で旧実装と一致 (Workspace は同じ算術を順序固定で実行する)
  - `delta_local_energy`: `rtol = 1e-12` で旧実装と一致 (E_new と E_old の取得順を変えるだけ)
- `SpinClusterHamiltonian` は純データのまま (mutable field を増やさない)。Workspace
  はホストする側 (`SCEMC` ないし呼び出し側) が保持する。
- 既存 parity テスト (`test/parity/test_parity_bcc.jl`, `test_parity_fege.jl`,
  `test_parity_ferh.jl`) は無修正で全通過。
- 既存 examples (`examples/01_*.jl` … `examples/05_*.jl`) は無修正で動作。

## スコープ

### 含む (本 spec)

- `SpinClusterWorkspace` 型の追加 (新規 `src/simple/workspace.jl`)
- Pure-buffer 引数版 API:
  - `total_energy(h, spins, ws::SpinClusterWorkspace)`
  - `local_energy(h, spins, i, ws)`
  - `delta_local_energy(h, spins, i, S_new, ws)`
  - `gradient(h, spins, i, ws)`
  既存無引数版はこれらを内部で呼ぶ薄いラッパに整理する。
- Stateful cache API (precondition / postcondition を明示):
  - `sync_workspace!(ws, h, spins)`
  - `delta_local_energy_cached!(h, spins, i, S_new, ws)`
  - `commit_workspace_column!(ws, i, S_new)`
- `SCEMC` への組み込み:
  - `mc.ws::SpinClusterWorkspace` field 追加
  - `init!` で `sync_workspace!`
  - `metropolis_sweep!` を `delta_local_energy_cached!` + `commit_workspace_column!` に切り替え
  - `_renorm_and_drift_check!` の後で `sync_workspace!` を再実行
- `_instance_energy` の内部用変種を 1 つ追加 (指定列だけ別 vector で読む):
  `_instance_energy_with_column(inst, zlm, col_idx, col_data, cg_table)` 等。
  これは `delta_local_energy_cached!` で `E_old` と `E_new` を 1 つの zlm から
  取り出すために必要。
- 充実したドキュメント (本 spec の "ドキュメント方針" 節参照)。
- ベンチマーク後値の取得とリポジトリへのコミット
  (`benchmark/results/<YYMMDD>-<sha>-darwin-arm64-workspace-after.md`)。

### 含まない (別 spec)

- `_instance_energy` の inner-loop alloc 除去 (現状 fege total_energy が
  84M allocs / 1.76 GiB の主因。これは数値経路の書き換えになるので別 spec で扱う)。
- HMC / Langevin / overrelaxation 等の新規 update スキーム (`gradient_all` API
  を含めて別 spec)。
- `total_energy` を hot path にする最適化 (init/drift check 以外で叩かない前提)。

## 制約

- **物理規約** (`CLAUDE.md`):
  - 温度 `T` は eV (Simple 内部規約)。Workspace は温度に依存しない。
  - スピン行列レイアウトは `3 × n_atoms`。
  - Zlm は実 (tesseral)。SpheriCart の `SphericalHarmonics(L; normalisation=:L2)` を使う。
- **API 規約**:
  - 既存無引数版は **後方互換維持**。signature を変えない。
  - Workspace 引数は最終引数に置き、デフォルトを内部生成にすることで `f(h, spins, ws)`
    と `f(h, spins)` の両方が共存する形にする。
- **依存**: 既存依存のみで完結 (SpheriCart, StaticArrays)。新規依存を追加しない。
- **言語**: ソース・docstring・コミット・PR は英語 (American spelling)。
- **コードスタイル**: SciML Style (`src/simple/.JuliaFormatter.toml` に従う)。

## 完了基準

### 機能完了

- [ ] `SpinClusterWorkspace(h)` で workspace が作れる
- [ ] Workspace 引数版 API 4 種が全テストで使える
- [ ] `sync_workspace!` / `delta_local_energy_cached!` / `commit_workspace_column!`
      が使え、`SCEMC.metropolis_sweep!` がそれ経由で動作する
- [ ] `make test` / `make test-slow` で全テスト pass (新規 parity テスト含む)
- [ ] 既存 examples が無修正で動作

### 数値整合性

| 比較対象 | 規約 | 状態 |
|---|---|---|
| `total_energy(h, spins)` 旧 vs `total_energy(h, spins, ws)` | `rtol = 0` (bit-exact) | 未 |
| `delta_local_energy(h, spins, i, S_new)` 旧 vs `delta_local_energy_cached!` | `rtol = 1e-14` | 未 |
| `SCEMC.metropolis_sweep!` の最終 `mc.energy` 旧 vs 新 (同 seed) | `rtol = 1e-12` | 未 |
| 既存 parity テスト (`bcc / fege / ferh`) | 既存規約 (`rtol = 1e-7` / `1e-8`) | 維持 |

bit-exact を要求するのは「同じ算術を同じ順序で実行している」と保証できるケースのみ。
Metropolis 軌跡の最終エネルギーは acceptance 判定の浮動小数差で発散しうるため
`rtol = 1e-12` で逃がす。

### パフォーマンス (vs M10 baseline)

| 指標 | M10 baseline | 目標 |
|---|---|---|
| `SCEMC.metropolis_sweep!` (bcc, 16 atoms) alloc/call | 未計測 (要前値追加) | 既存 baseline / 100 以下 |
| `total_energy` (fege, 64 atoms) alloc/call | 84 M | 改善は副次目的 (Workspace pure-buffer 経路では Zlm 行列 alloc のみ消える。inner-loop alloc は別 spec) |
| `delta_local_energy` 1 call alloc (fege) | 計測 | 0 か数十程度 |

**主目標は MC hot path の alloc 削減**。total_energy の絶対 alloc 数は本 spec では
追求しない (別 spec のスコープ)。

### 教材性

- [ ] `SpinClusterWorkspace` の docstring に **不変条件 / サイズ式 / 寿命** を箇条書きで明記
- [ ] `sync_workspace!` / `delta_local_energy_cached!` / `commit_workspace_column!`
      の docstring に `# Preconditions:` `# Postconditions:` 形式の契約を明記
- [ ] 各 cached 関数に対応する `JPhiMagestyCarlo` 側の関数名を `# Mirrors:` 行で参照
- [ ] **Workspace 引数なし版** (`total_energy(h, spins)` / `local_energy(h, spins, i)` /
      `delta_local_energy(h, spins, i, S_new)` / `gradient(h, spins, i)`) の
      docstring 冒頭に以下の趣旨を英語で明記:
      > This bufferless overload is not used in the Monte Carlo hot path; it is
      > kept as an educational / testing reference. `SCEMC.sweep!` always goes
      > through the workspace-aware overload (or `delta_local_energy_cached!`
      > for single-spin proposals).
- [ ] `src/simple/workspace.jl` (新規モジュール) 先頭に design 全体の WHY を 10〜20 行で説明
- [ ] `docs/src/api.md` の Simple Module セクションに Workspace 関連を追記

### ベンチマーク

- [ ] `benchmark/simple/bench_sweep.jl` で sweep 中の alloc/call を計測
- [ ] M10 baseline と比較した after ファイルを
      `benchmark/results/<YYMMDD>-<sha>-darwin-arm64-workspace-after.md` でコミット
- [ ] `docs/design_notes.md` の "SphericalHarmonics の使い回し" 項目を「完了」に更新

## ドキュメント方針 (本 spec 固有)

CLAUDE.md は通常「デフォルトはコメント書かない」だが、本 spec は以下の理由で
**ドキュメントを厚めに書く例外**として扱う:

- **不変条件 (precondition) が読んだだけでは見えない**: 「`ws.zlm` は `spins` と
  同期している」という契約は型情報からは伝わらない。
- **Pure-buffer / Stateful / Bufferless の 3 セマンティクスが並存する**: ユーザーが
  選択する必要があるため、API の意図を明示しないと誤用が出る。
- **reference 版 (bufferless) と hot-path 版 (workspace / cached) の役割分担**:
  「同じ関数名で signature が違うものがある」ときに、どちらが MC 本体経路かが
  docstring からすぐ読み取れるべき。
- **Optimized 側 (`_update_atom_zlm_cache!` 等) との対応関係**: Simple のリファレンス
  性を保つには「同じことを別実装でやっている」と明示する必要がある。

具体的な書き方は [design.md](design.md) の "ドキュメント規約" 節で詳述。

## 参考

- M10 baseline 数値: [`benchmark/results/260513-cf25754-darwin-arm64-M10-baseline.md`](../../../benchmark/results/260513-cf25754-darwin-arm64-M10-baseline.md)
- Optimized 側の対応実装:
  - `JPhiMagestyCarlo._build_zlm_cache` (= `sync_workspace!`)
  - `JPhiMagestyCarlo._update_atom_zlm_cache!` (= `commit_workspace_column!`)
- 設計判断の背景 (会話ログ): 案 A/B/C 比較の上で C を選択。理由は (1) 将来 HMC で
  `gradient_all` 系 API が必要になる、(2) `SpinClusterHamiltonian` を mutable に
  したくない、(3) Workspace を明示引数にすることで「副作用がある」ことが型で見える。
