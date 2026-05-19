# Tasklist — `jphi_threshold`

> 関連 spec: [requirements.md](requirements.md) / [design.md](design.md)

各マイルストーンの完了時に `- [x]` + 完了日を付ける (途中状態は触らない、
`CLAUDE.md` の運用ルール準拠)。日々の細かい作業は `TaskCreate` で管理。

## M1: Simple 実装 + テスト

- [ ] `src/simple/types.jl`: `SpinClusterHamiltonian` に `jphi_threshold` kw 追加、
      `_generate_instances` で filter、log + empty チェック。
      **`threshold == 0.0` のときは filter/log/check ブロック全体を短絡**
      (bit-exact 保証)。`SpinClusterHamiltonian` の docstring に
      `jphi_threshold` 引数説明 + `J = 0.0` を厳密に drop したい場合の
      `eps()` 注記を追加。
- [ ] `src/simple/mc.jl`: `SCEMC` 構築で `params[:jphi_threshold]` を読み、
      `register_evaluables` でも同様に。SCEMC docstring の optional params
      表に追記。
- [ ] `test/simple/test_jphi_threshold.jl`: filter 挙動 / エラー / default 一致。
- [ ] `make test` (Simple 部分) で pass。

## M2: Optimized 実装 + テスト

- [ ] `src/JPhiMagestyCarlo.jl`: `load_sce_hamiltonian` に `jphi_threshold` kw、
      `salc_list` / `jphi` の同期 filter (`threshold == 0.0` で短絡)。
- [ ] キャッシュ Dict 型の更新: `_HAM_CACHE` / `_ECACHE_CACHE` /
      `_DERIVED_CACHE` のキーに `Float64` (threshold) を追加。
- [ ] `_mpi_build_ham_and_cache` を `(xml_path, rep, thr)` シグネチャに変更。
- [ ] `_get_or_build_derived` を `(xml_path, rep, thr, ...)` シグネチャに変更
      (現状 `:tensor_template` パスのみ呼ぶが、両パスのキャッシュキーを揃える)。
- [ ] `JPhiSpinMC` constructor 内の `get!(_HAM_CACHE, (xml, rep))` を
      threshold 付きキーに更新 (`:tensor_template` パス)。
- [ ] `register_evaluables` 内の `n_atoms` lookup を threshold 付きキーに更新。
- [ ] `JPhiSpinMC` に `jphi_threshold::Float64` フィールド追加 +
      `Serialization.serialize` / `deserialize` に追加 (deserialize 側で
      `_mpi_build_ham_and_cache(xml, rep, thr)` を呼ぶ)。
- [ ] serialize/deserialize 直上に「順序依存・同一バージョン内のみ」コメント。
- [ ] `test/optimized/test_jphi_threshold.jl` (or 既存 test dir に追加)。

## M3: Parity + slow テスト

- [ ] `test/parity/test_jphi_threshold_parity.jl`: 同じ threshold で
      Simple と optimized の `total_energy` が parity 範囲内で一致。
- [ ] 境界値テスト: `threshold` を fixture の `jphi[s]` の正確な絶対値に
      合わせ、Simple / optimized が両方とも「≥ で keep」の判定で
      同じ keep mask を返すこと (浮動小数表現の一致を担保)。
- [ ] `make test` 全体 pass、`make test-slow` も pass。
- [ ] `code-reviewer` エージェントで差分レビュー。
