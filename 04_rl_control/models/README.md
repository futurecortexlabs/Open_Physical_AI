# 公開モデル

| ファイル | SHA-256 | 用途 |
|---|---|---|
| `ppo_multigoal_final.pt` | `9d5646780f2865b0e535c9b410d07b477cb4c09b856271e74a8cb6e21e171049` | 指示位置対応、全方向5〜30cm。最終評価21.88% |
| `ppo_near_best.pt` | `c04a7b1d79f364c679af773c104562b4c9206d71e8c46aa2db4e1a5dd6d5de43` | 旧・固定近距離。限定評価97.66% |

`ppo_multigoal_final.pt`は`run_ppo_command.py`で日本語指示またはXY目標を渡せます。`ppo_near_best.pt`はgoal-conditionedモデルではないため、指示位置の実行には使用できません。

どちらも学習時のUSD SHA-256 `b84294802d13626d46c43ab8519eb620c47241a4f5b5601a704e28d7fa0cf7b9`を記録しています。モデルとロボット構成が一致しない場合は実行を拒否します。

PyTorch形式にはデシリアライズ上の注意があります。このリポジトリなど信頼できる配布元から取得し、`weights_only=True`で読み込んでください。
