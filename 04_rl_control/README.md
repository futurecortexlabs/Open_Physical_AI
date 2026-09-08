# #04 指示位置対応のPPOロボット制御

1つのgoal-conditioned PPO方策で、赤いキューブを指定位置へ運ぶIsaac Sim実験です。対応する指示は「右20cm」「左10cm」「奥15cm」「右15cm 奥10cm」およびロボット座標系のXY座標です。

## 結果

| モデル | 評価条件 | 全工程＋2秒維持 |
|---|---|---:|
| `models/ppo_multigoal_final.pt` | 最大開始難易度、全方向5〜30cm、16指示 | **224/1,024（21.88%）** |
| `models/ppo_near_best.pt` | 旧・固定近距離条件 | **250/256（97.66%）** |

最新の指示位置対応モデルは90%未達です。近距離97.66%は別条件であり、最新モデルの成功率として扱いません。詳細は[最終学習レポート](docs/training_report.md)と[結果一覧](results/README.md)にあります。

## 学習される動作

PPOの行動は手先XYZの増分とグリッパー指令の4次元です。観測には関節状態、手先、物体、目標との差、直前行動などを含みます。PPOが全工程で同じ方策を使い続け、工程ごとのスクリプト切替は行いません。

低レベル側には以下を残しています。

- 手先XYZから関節目標へ変換するIK
- 手先の下向き姿勢維持
- 関節制限と重力補償
- 人が設計した報酬、成功条件、カリキュラム

教師データ、BC、DAgger、人による途中操作は使用していません。

## 準備

1. Isaac Sim 6.0.1とCUDA対応PyTorch環境を用意します。
2. [共通USDの説明](../assets/README.md)に従い、`assets/crx20ia_l+Robotiq_2F_85_edit+rsd455.usd`を配置します。
3. リポジトリルートを作業ディレクトリにします。

公開モデルは学習時のUSD SHA-256 `b84294802d13626d46c43ab8519eb620c47241a4f5b5601a704e28d7fa0cf7b9`と一致する場合だけ実行できます。異なるロボット構成では再学習が必要です。

以下ではIsaac Simを利用できるPythonを`$PpoPython`とします。

```powershell
$PpoPython = 'C:\IsaacLab\env_isaaclab\Scripts\python.exe'
```

## 指示して動かす

```powershell
& $PpoPython .\04_rl_control\run_ppo_command.py `
  --checkpoint .\04_rl_control\models\ppo_multigoal_final.pt `
  --instruction '右20cm' `
  --record .\04_rl_control\results\my_right20cm.mp4
```

複合方向または絶対座標も指定できます。

```powershell
& $PpoPython .\04_rl_control\run_ppo_command.py --checkpoint .\04_rl_control\models\ppo_multigoal_final.pt --instruction '左15cm 手前10cm'
& $PpoPython .\04_rl_control\run_ppo_command.py --checkpoint .\04_rl_control\models\ppo_multigoal_final.pt --goal-xy 0.70 0.05
```

1試行の終了コードは、全工程と2秒維持に成功すれば0、課題失敗は1、実行エラーは2です。指示は目標観測を変えるだけで、モデル重みや成功条件は変えません。

## 16指示で評価する

```powershell
& $PpoPython .\04_rl_control\validate_ppo.py `
  --checkpoint .\04_rl_control\models\ppo_multigoal_final.pt `
  --eval-task full --min-success-rate 0.9 `
  --seeds 8001 8002 8003 8004 --num-envs 256 `
  --verify-steps 60 --output-dir .\04_rl_control\runs\public_validation
```

16種類を各64回、合計1,024試行で評価します。合格には全体・各seed・各指示がすべて90%以上であることを要求します。

## 継続学習する

```powershell
& $PpoPython .\04_rl_control\train_ppo_until_target.py `
  --checkpoint .\04_rl_control\models\ppo_multigoal_final.pt `
  --output-dir .\04_rl_control\runs\continuation `
  --rounds 12 --train-seconds 1800 --num-envs 256 `
  --target-rate 0.9 --first-eval-seed 8101 --confirm-candidate
```

RTX 5070 Ti環境の直近実測では、学習6時間と区間ごとの評価を合わせ約7時間29分でした。`runs/`は生成物でGit対象外です。上限終了は90%達成を意味しません。

PPOは試行経験を増やしながら報酬の高い行動を学ぶため、適切な条件で学習を重ねるほど成功率の向上を目指せます。ただし成功率は単調増加を保証されず、長時間学習だけで90%になるとは限りません。各区間の独立評価を見て、停滞や性能低下があればカリキュラムやハイパーパラメータを見直します。

## 主なファイル

| ファイル | 役割 |
|---|---|
| `ppo_pick_place.py` | 学習・評価のエントリーポイント |
| `run_ppo_command.py` | 日本語指示またはXY座標で実行 |
| `train_ppo_until_target.py` | 時間制限付き継続学習と独立評価 |
| `validate_ppo.py` | 複数seedの合格判定 |
| `summarize_ppo_stages.py` | 工程別・指示別集計 |
| `compare_ppo_evaluations.py` | 同条件評価の比較 |
| `ppo/` | PPO、環境、IK、目標、記録処理 |
| `tests/` | シミュレータ不要の単体テスト |

## テスト

```powershell
python -m unittest discover -s .\04_rl_control\tests -v
```

## 制約

- 全方向5〜30cmでは21.88%で、実用段階ではありません。
- 30cmの各指示は最終評価でほぼ成功していません。
- 把持判定の一部は接触力ではなく位置と指関節角の代理条件です。
- シミュレーションの成功は実機の安全性・再現性を保証しません。
- PyTorchチェックポイントは信頼できる配布元のものだけを読み込んでください。
