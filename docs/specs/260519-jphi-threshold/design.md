# Design — `jphi_threshold`

> 関連 spec: [requirements.md](requirements.md) / [tasklist.md](tasklist.md)

## API

### Simple 実装

```julia
SpinClusterHamiltonian(xml_path::AbstractString;
                       repeat::NTuple{3,Int} = (1, 1, 1),
                       jphi_threshold::Real  = 0.0) -> SpinClusterHamiltonian
```

- `jphi_threshold` は **eV の非負実数**。`thr < 0` で `ArgumentError`。
- `Simple.SCEMC` 構築時:
  - `params[:jphi_threshold]` (デフォルト `0.0`) を読み取り、
    `SpinClusterHamiltonian(...; jphi_threshold = ...)` に渡す。
  - `Carlo.register_evaluables` でも同じ値を読んで Hamiltonian を再構築する
    (こちらが使うのは `n_atoms` のみだが、API は揃える)。

### Optimized 実装

```julia
load_sce_hamiltonian(xml_path::AbstractString;
                     repeat::NTuple{3,Int} = (1, 1, 1),
                     jphi_threshold::Real  = 0.0) -> SCEHamiltonian
```

- `SCEHamiltonian.salc_list` と `.jphi` は同じ index で対応しているため、
  filter は両方に同じ keep-mask を適用する (`keepat!` または `findall` + 再代入)。
- `JPhiSpinMC` 構築:
  - `params[:jphi_threshold]` を読み、`load_sce_hamiltonian` に渡す。
  - フィールドとして `jphi_threshold::Float64` を追加。
  - `_HAM_CACHE`, `_ECACHE_CACHE`, `_DERIVED_CACHE` のキーに threshold を追加:
    - `Dict{Tuple{String, NTuple{3,Int}, Float64}, ...}`
  - `Carlo.register_evaluables` も同じキーで lookup する (現状の
    `(xml_path, repeat)` キーに threshold を追加)。
  - `Serialization.serialize` / `deserialize` で `jphi_threshold` も書き出す
    (`xml_path`, `repeat` の直後に追加するのが自然)。

## Filter 規約

```julia
keep(s) = abs(jphi[s]) ≥ jphi_threshold
```

- 厳密に **≥** で keep、`<` で drop。`threshold = 0.0` のとき
  `abs(J) ≥ 0` は常に true なので一切落とさない (既存挙動)。
- 浮動小数点での境界比較は `abs(J)` と `threshold` の生値で行う
  (相対 epsilon を入れない — ユーザーが指定した数値を素直に解釈する)。
- `J = 0.0` (厳密ゼロ) の扱い:
  - `threshold = 0.0` ではデフォルトで keep される (`abs(0) ≥ 0` が true)。
    これは bit-exact 不変条件を守るための仕様。
  - ユーザーが「厳密ゼロを除きたい」場合は `threshold = eps()` または
    `nextfloat(0.0)` を渡す (docstring に明記)。
- **threshold = 0.0 の短絡 (bit-exact 保証)**: filter / log / empty-check の
  ブロック全体を `if thr > 0` で囲み、`threshold = 0.0` のときは 1 行も
  追加処理を走らせない。これにより `_generate_instances` (Simple) /
  `salc_list` 再代入 (optimized) のコードパスが既存と完全に同一になり、
  数値的にも順序的にも bit-exact が保証される。

## エラー処理

- 全 SALC が drop される場合:
  ```
  throw(ArgumentError(
      "jphi_threshold=$thr eV filters out all $n_total SALCs " *
      "(max |J|=$max_abs eV); Hamiltonian would be empty"))
  ```
- Simple 側で `_generate_instances` の結果が空になる場合も同様に
  `ArgumentError` を投げる (理論上は SALC を 1 個以上残しても
  filter 順序の都合で空になることはないが、防御的に確認する)。

## ログ

- 構築時、`n_dropped > 0` のとき 1 行だけ:
  ```
  @debug "Dropped $n_dropped / $n_total SALCs below jphi_threshold=$thr eV " *
         "(max dropped |J|=$max_dropped eV)"
  ```
- レベルは `@debug` (テスト中に Carlo が walker × 温度数だけ Hamiltonian を
  構築するため、`@info` だと数十行流れる)。ユーザーが確認したい場合は
  `JULIA_DEBUG=SpinClusterMC` で出せる。
- `n_dropped == 0` のときは黙る (`threshold = 0` でログが出続けるのを避ける)。

## キャッシュ整合性 (optimized 側)

現状 (シンボル名で位置指定 — 行番号は陳腐化するため避ける):

```julia
const _HAM_CACHE     = Dict{Tuple{String,NTuple{3,Int}}, SCEHamiltonian}()
const _ECACHE_CACHE  = Dict{Tuple{String,NTuple{3,Int}}, LocalEnergyCache}()
const _DERIVED_CACHE = Dict{Tuple{String,NTuple{3,Int},Tuple}, DerivedInstanceCache}()
```

→ 変更:

```julia
const _HAM_CACHE     = Dict{Tuple{String,NTuple{3,Int},Float64}, SCEHamiltonian}()
const _ECACHE_CACHE  = Dict{Tuple{String,NTuple{3,Int},Float64}, LocalEnergyCache}()
const _DERIVED_CACHE = Dict{Tuple{String,NTuple{3,Int},Float64,Tuple}, DerivedInstanceCache}()
```

`threshold` 付きタプルで揃えるべき lookup 箇所:

- `_mpi_build_ham_and_cache(xml_path, rep, thr)` (`:tensor` パス)
- `_get_or_build_derived(xml_path, rep, thr, ...)` — 現状は `:tensor_template`
  パスでのみ呼ばれる。直接 `load_sce_hamiltonian` を呼ぶ `:tensor` パスと
  キーが揃うように更新する。
- `JPhiSpinMC` constructor 内の `get!(_HAM_CACHE, (xml, rep))` も
  `get!(_HAM_CACHE, (xml, rep, thr))` に変更。
- `Carlo.register_evaluables` 内の `n_atoms` lookup (現状 `_HAM_CACHE` を
  キー `(xml_path, rep)` で参照している箇所) も同様に threshold を含める。
- `Serialization.deserialize` 内の `_mpi_build_ham_and_cache(xml_path, repeat)`
  も threshold 付きで呼ぶ (deserialize 側で `jphi_threshold` を読み取り済み
  なので渡せる)。

両パス (`:tensor` / `:tensor_template`) は filter 後の `salc_list` を
共有するため、片方だけ filter する状況は発生しない。

## テスト計画

`test/simple/` と `test/parity/` に 1 ファイルずつ追加。fixture は既存の
`test/bcc/jphi.xml` 等を使う。

1. `threshold = 0.0` (default) と XML から得た `length(jphi)` × instance 数が
   既存と一致 — bit-exact。
2. `threshold = 0.5 * minimum(abs, jphi)` 付近を選び、
   - drop 数 > 0
   - `total_energy` の値が threshold = 0 のときと differ する
   - エネルギーは drop された SALC の寄与分だけずれる (定量チェックは省略可)。
3. `threshold > maximum(abs, jphi)` → `ArgumentError`。
4. Simple/optimized 双方で同じ `threshold` を与えたとき、両者の
   `total_energy` が parity 範囲内で一致。

## 連動箇所 (CLAUDE.md 準拠で確認すべき箇所)

- `_foreach_translated_instance` / `coupled_cluster_energy` — タイリングロジック
  自体は触らない。filter は SALC レベルでのみ行う。
- `:tensor_template` / `:tensor` の 2 パス整合性 — どちらも filter 後の
  `salc_list` / `instances` を共有するので、片方だけ filter する状況は発生しない。
- `measure!` の per-atom 規約 — エネルギー量は変わるが規約は変わらない。
- physical 規約 (温度・スピン・球面調和) — 一切変更しない。

## 互換性

- 既存ユーザーは `params[:jphi_threshold]` を指定しない限り影響なし。
- 既存の保存済み HDF5 checkpoint は `spins` + `energy` のみで threshold 非依存
  なのでそのまま読める。
- MPI Serialization のレイアウトは変わる (フィールド 1 個追加) が、同一プロセス
  / 同一バージョン内でのやり取りなのでバージョン互換性問題は生じない。
  実装時、`Serialization.serialize` / `deserialize` の本体直上に
  「serialize レイアウトは順序依存。同一バージョン内のみで有効。フィールド
  追加/削除/順序変更は両関数を同時に更新すること」のコメントを残す。
