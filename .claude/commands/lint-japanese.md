---
description: src/ test/ benchmark/ の .jl と直近のコミットメッセージから日本語混入を検出する
---

`src/`, `test/`, `benchmark/` 配下の `.jl` ファイルと、直近の git コミットメッセージを
スキャンして、日本語文字 (kanji / hiragana / katakana) が混入している箇所を報告してください。

ポリシー: `feedback_comments_in_english.md` により、ソース・コミット・PR はすべて英語。

使うコマンド (perl は macOS/Linux 両対応):

```bash
find src test benchmark -name '*.jl' -print0 2>/dev/null \
  | xargs -0 perl -CSD -ne 'print "$ARGV:$.: $_" if /[\x{3040}-\x{30FF}\x{4E00}-\x{9FFF}]/'
```

```bash
git log -1 --format=%B \
  | perl -CSD -ne 'print "commit msg line $.: $_" if /[\x{3040}-\x{30FF}\x{4E00}-\x{9FFF}]/'
```

報告フォーマット:
- 違反があれば `path:line: snippet` の箇条書きで一覧
- 違反がなければ **"clean"** とだけ返す (他の文を付けない)

`docs/` 配下は対象外 (日本語 OK)。
