# Tasklist — General supercell (任意 3×3 整数行列 M)

> 関連 spec: [requirements.md](requirements.md) / [design.md](design.md)

各マイルストーンの完了時に `- [x]` + 完了日を付ける (途中状態は触らない、
`CLAUDE.md` の運用ルール準拠)。日々の細かい作業は `TaskCreate` で管理。

> **実装中の設計変更 (2026-06-20)**: 当初の「M=diag で旧 atom 付番を bit-exact
> 再現する互換層」は、legacy の `round(base_frac)` 折り畳みを primitive で再現
> できず断念。代わりに **対角 `repeat` は legacy `_generate_instances` をそのまま
> 使い、一般 `supercell_matrix` のみ新 primitive パス** とした (付番は異なるが
> エネルギーは一致)。また primitive タイリングは面上 (半周期) クラスタを二重
> 計上するため、**`effective_mult = multiplicity ÷ s_base` で un-fold** する補正を
> 追加 (Magesty の multiplicity 規約に一致)。詳細は更新済み `design.md` §3.3/§3.4/§4。

## M1: 整数線形代数基盤

- [x] (2026-06-20) `src/simple/supercell.jl` (新規): `_int_det3` / `_adjugate3` /
      `_col_hermite` / `_wrap_offset_into_supercell`。すべて 3×3 `SMatrix`
      ベース、整数演算のみ。
- [x] (2026-06-20) 単体テスト: `_col_hermite` の `M = H*U` 性質と `H` の正準形、
      `_int_det3` が `det` と一致、`adj(M)*M = det(M) I`、
      `_wrap_offset_into_supercell` が任意 offset を `N_cells` 個の代表の
      1つに落とすこと。

## M2: primitive 抽出

> 依存: M1 (`_int_det3` / `_adjugate3`) 完了後に着手。

- [x] (2026-06-20) `PrimitiveCell` 型と `extract_primitive(sys::SystemData)`。最短独立
      並進3本・右手系化・副格子分類・`base_to_prim` / `reshape_base` /
      逆引き `prim_to_base`。整合性 assert。
- [x] (2026-06-20) テスト: bcc (n_prim=1) / fege / ferh fixture で `n_prim*n_trans ==
      base_n`、`|det(reshape_base)| == n_trans`、`base_to_prim` の往復一致
      (base atom → (s,Δ) → base atom)。

## M3: クラスタテンプレート化

> 依存: M2 (`PrimitiveCell` / `base_to_prim`) 完了後に着手。

- [x] (2026-06-20) `ClusterTemplate` 型と `build_templates(salcs, jphi, sys, prim;
      jphi_threshold)`。pivot 正規化・per-basis dedup・threshold 短絡。
- [x] (2026-06-20) テスト: 各 fixture で、テンプレートを M=I 相当に展開した instance 集合が
      旧 `_generate_instances(...; repeat=(1,1,1))` と
      (sorted atoms, ls, J, multiplicity) で一致。

## M4: 一般 M タイリング + 互換層

> 依存: M1-M3 すべて完了を前提。

- [x] (2026-06-20) `_generate_instances_matrix(templates, prim, M; compat_repeat)`。
      cell 列挙 (HNF)・`cell_index`・付番・互換層
      (`cell_id ↔ (ti,tj,tk,prim_in_base)` 写像)。
- [x] (2026-06-20) **bit 等価テスト**: `compat_repeat=(n1,n2,n3)` で生成した instance 集合・
      atom 付番が、旧 `_generate_instances(...; repeat=(n1,n2,n3))` と
      完全一致 (bcc / fege で (1,1,1) / (2,1,1) / (2,2,2))。

## M5: 型/コンストラクタ/API 配線

- [x] (2026-06-20) `src/simple/types.jl`: `SpinClusterHamiltonian` を `supercell_matrix`
      kwarg 対応に改修 (解決規則・二重指定エラー・`det(M)==0` エラー)。
      構造体に `supercell_matrix` フィールド追加、`repeat` は後方互換で保持。
- [x] (2026-06-20) `src/simple/mc.jl`: `SCEMC` の param 解決 (`:supercell_matrix`)・
      フィールド・`init!`・`register_evaluables` を追従。
- [x] (2026-06-20) `src/simple/spin_proposal.jl`: `init_spins` / `_tile_base_matrix` は
      M=diag で無改変動作を確認。非対角 `M` + `:initial_spins` の扱いを決定
      (一般化 or 明示エラー)。
- [x] (2026-06-20) 既存 simple / parity テストが**無改変で pass** することを確認
      (`make test`)。

## M6: 検証テスト

- [x] (2026-06-20) `test/simple/test_simple_supercell.jl` (新規): 整数線形代数・primitive
      抽出・M=diag↔repeat bit 等価・非対角 M の並進不変性・
      `total_energy ∝ |det(M)|`・対角等価 `M'` 一致。`runtests.jl` の simple
      include 群へ追加。
- [x] (2026-06-20) エネルギー等価性は M4 testset (`matrix tiling energy
      equivalence with legacy`) に集約 (ferro 配置で primitive パス ↔ legacy の
      per-atom エネルギー一致 + 任意 M の intensive 性)。別ファイルの
      `test_parity_supercell.jl` は作らず。
- [x] (2026-06-20) 重い非対角 ferh 検証は slow 節へ。`make test` / `make test-slow` pass、
      JET pass。

## M7: ドキュメント + フォーマット

- [x] (2026-06-20) `SpinClusterHamiltonian` / `SCEMC` docstring に `supercell_matrix` を
      追記。`docs/terminology.md` の supercell 定義に一般 `M` の節を追加。
- [x] (2026-06-20) `julia --project=. -e 'using JuliaFormatter; format("src/simple")'` と
      `format("test/simple")` で SciML フォーマット適用。
- [x] (2026-06-20) `code-reviewer` / `numerical-reviewer` で差分レビュー。
