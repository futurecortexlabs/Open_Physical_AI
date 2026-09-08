# GitHub・note公開前チェック

## GitHub

- [ ] `git status --short --ignored`で`runs/`とUSDがGit対象外になっている。
- [ ] 個人の絶対パス、認証情報、ローカルログを含んでいない。
- [ ] `python -m unittest discover -s 04_rl_control/tests -v`が成功する。
- [ ] Markdownの相対リンクがすべて解決する。
- [ ] 代表モデル2件と動画のSHA-256を確認する。
- [ ] USDと映像内アセットの公開条件を権利者の規約で確認する。
- [ ] 21.88%の全範囲評価と97.66%の固定近距離評価を混同しない。
- [ ] シミュレーション結果を実機の安全保証として説明しない。

## note

1. [記事原稿](../04_rl_control/docs/note_article.md)を本文として使用します。
2. 動画は[near_best.mp4](../04_rl_control/results/near_best.mp4)を掲載できます。
3. 記事内の相対リンクはGitHub公開後のURLまたはnoteの添付素材へ置き換えます。
4. 97.66%が固定近距離、21.88%が全方向5〜30cmである説明を残します。
5. 投稿プレビューで表、コードブロック、動画を確認します。

commit、push、noteへの投稿は別操作です。
