# Open Physical AI

NVIDIA Isaac Simでロボット制御を段階的に学ぶ、Pythonサンプル集です。各サンプルは教育・シミュレーション用途であり、実機の安全を保証するものではありません。

## シリーズ

| No. | テーマ | 内容 |
|---:|---|---|
| 01 | [かんたんピッキング](01_simple_picking/README.md) | 既知座標とIKによるピック＆プレース |
| 02 | [カメラピッキング](02_camera_picking/README.md) | RGB-Dと色検出による位置推定 |
| 03 | [複数ワーク](03_multi_picking/README.md) | 複数物体の検出と順次搬送 |
| 04 | [PPOロボット制御](04_rl_control/README.md) | 指示位置へ運ぶPPO-only方策 |

## #04の現在地

#04は「右20cm」「左10cm」「奥15cm」などの指示を目標座標へ変換し、1つのPPO方策で接近・把持・持ち上げ・搬送・開放・退避を行います。動作ラベル、模倣学習、途中の人操作は使いません。関節指令への変換には低レベルIKと固定した手先姿勢を使用します。

最大開始難易度、全方向5〜30cm、未使用4seed・1,024試行、完了後2秒維持という評価で、最終成功率は**224/1,024（21.88%）**でした。目標の90%には未達です。一方、以前の固定近距離条件では250/256（97.66%）を確認していますが、全範囲の成績ではありません。

PPOは環境との試行錯誤を重ねて方策を改善するため、適切なカリキュラムで追加学習するほど成功率の向上を目指せます。ただし、更新ごとに必ず上昇するわけではなく、一時的な低下や停滞もあるため、未使用seedによる独立評価で確認します。

- [#04の実行方法](04_rl_control/README.md)
- [最終学習レポート](04_rl_control/docs/training_report.md)
- [note記事原稿](04_rl_control/docs/note_article.md)
- [結果と動画](04_rl_control/results/README.md)

## 必要環境

- NVIDIA Isaac Sim 6.0.1
- CUDA対応PyTorch
- Windows 11で動作確認
- 検証PC：RTX 5070 Ti 16GB、Core Ultra 7 265K、RAM 64GB

ロボットUSDは第三者アセットを含むためGitには収録しません。[assets/README.md](assets/README.md)に従って各自で配置してください。保存済みモデルは学習時と同一ハッシュのUSDを要求します。

## 構成

```text
Open_Physical_AI/
├── 01_simple_picking/
├── 02_camera_picking/
├── 03_multi_picking/
├── 04_rl_control/
│   ├── ppo/                  # 環境・方策・学習・評価
│   ├── models/               # 公開用チェックポイント
│   ├── results/              # 最終評価と代表動画
│   ├── docs/                 # 技術資料と記事原稿
│   ├── tests/
│   ├── ppo_pick_place.py
│   ├── run_ppo_command.py
│   └── train_ppo_until_target.py
├── assets/
└── docs/publishing.md
```

コードは[MIT License](LICENSE)です。ロボットモデル、商標、映像内アセットにはそれぞれの権利者の条件が別途適用されます。
