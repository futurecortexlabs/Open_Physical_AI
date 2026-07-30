# ==============================================================================
# CRX20iA/L + Robotiq 2F-85  複数ワークの順次ピッキング（note記事用 #03）
# Isaac Sim 6.0.1 対応
# ==============================================================================
#
# ■ 概要
# #02 は「赤いキューブ1個」をカメラで見つけて拾いました。#03 では **複数個** を
# 一度に見つけ、**1つずつ順番に**（＝順次）ピック＆プレースします。
#
# ワークは自作キューブをやめ、**Isaac Sim の標準アセット**（YCB の赤いスポンジ
# ブロック）を参照します。色検出のしきい値は #02 のまま変えていません。
#
# 増えたのは大きく2つだけ。
#   A. 検出が「重心1点」→「ブロブ（連結成分）ごとの重心リスト」になった
#      赤マスクを numpy だけでラベル分けして、ワーク1個ずつに切り分けます。
#   B. 状態機械の外側に「タスクキュー」ができた
#      検出した順序（既定はロボットに近い順）に並べ、1個終わったら次の1個へ。
#
# ロボットを動かす土台（JacobianIK / 8フェーズの状態機械）は #01・#02 と同一です。
#
# ■ 構成（上から順に読める並び）
#   1. 数学ユーティリティ     … クォータニオン→回転行列, 歪対称行列
#   2. 設定（dataclass）      … ロボット / シーン / カメラ / IK / 制御
#   3. フェーズ定義（Enum）   … 1ワークあたり8段階のピック＆プレース手順
#   4. JacobianIK             … 減衰最小二乗ヤコビアンIKソルバ（#01 と同一）
#   5. label_blobs            … 赤マスクを連結成分に分けるラベル伝播（numpy のみ）
#   6. WorkDetector           … RGB-Dカメラで「複数個」の世界座標を推定
#   7. MultiPickPlace         … タスクキュー＋フェーズ状態機械
#   8. main                   … ステージ読み込みとメインループ
#
# ■ 実行方法（Windows PowerShell）※リポジトリのルートから実行
#   C:\isaacsim\python.bat ".\03_multi_picking\multi_picking.py"
# ■ 実行方法（Linux）
#   /home/dev/isaacsim/python.sh "./03_multi_picking/multi_picking.py"
# ==============================================================================

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from isaacsim import SimulationApp

# SimulationApp は他の isaacsim モジュールを import する前に生成する必要がある。
simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.usd
import warp as wp
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Usd, UsdGeom, UsdPhysics

# ログは自前のハンドラで stdout へ出す。Kit（Isaac Sim 本体）が先に root ロガーへ
# ハンドラを付けてしまうため、logging.basicConfig は何もせず（既存ハンドラがあると
# 無視される仕様）、しかも Kit 側は INFO を表示しないので log.info が消える。
# 自分のロガーに直接ハンドラを付け、propagate を切って Kit へ流さないようにする。
log = logging.getLogger("multi_picking")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[MULTI_PICKING] %(message)s"))
log.addHandler(_handler)
log.propagate = False

# 手先を常に向けたい方向（真下）。
_DOWN = np.array([0.0, 0.0, -1.0])


# ==============================================================================
# 1. 数学ユーティリティ
# ==============================================================================
def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """クォータニオン (w, x, y, z) を 3x3 回転行列へ変換する。"""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def skew(v: np.ndarray) -> np.ndarray:
    """ベクトル v の歪対称行列（外積を行列積で表すための行列）。"""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ==============================================================================
# 2. 設定（dataclass）
# ==============================================================================
@dataclass(frozen=True)
class RobotConfig:
    """ロボットと USD 上のパス・関節に関する設定。"""

    usd_path: Path
    robot_path: str = "/World/crx20ia_l"
    gripper_base_link: str = "base_link"  # IKで狙うリンク（グリッパー根元）
    grip_frame_path: str = (
        "/World/crx20ia_l/J6_link/flange/ee_link/Robotiq_2F_85_edit"
        "/Robotiq_2F_85/base_link/grip_frame"
    )
    gripper_joint: str = "finger_joint"
    gripper_open: float = 0.0
    gripper_close: float = 0.55
    arm_dof: int = 6


@dataclass(frozen=True)
class SceneConfig:
    """ワーク（Isaac Sim 標準アセット）の初期配置と、並べ直す先（スロット）の設定。

    ワークは ``work_positions`` にバラバラに置き、``slot_*`` で作る一列の
    スロットへ順番に並べ替えます（ループ時は逆向きに戻す）。

    #01・#02 は ``UsdGeom.Cube`` で赤いキューブを自作していましたが、#03 では
    **Isaac Sim の標準アセット**（YCB の赤いスポンジブロック）を参照します。
    実測サイズは 77.9 × 51.2 × 52.6 mm で、短辺 51.2 mm を 2F-85（開き 85 mm）
    でつかみます。上面は赤（RGB 約 130,55,52）なので、#02 からの色検出は
    しきい値を変えずにそのまま使えます。
    """

    # Isaac Sim アセットサーバー上の相対パス（実行時にダウンロードされる）
    work_asset: str = "/Isaac/Props/YCB/Axis_Aligned/061_foam_brick.usd"
    work_height: float = 0.0526  # ワークの高さ [m]（持ち上げ高さの基準に使う）
    work_mass: float = 0.05  # アセットに質量が付いていないので自分で与える [kg]

    # 散らばったワークの初期位置（ロボットには“教えない”＝カメラで探す）
    # Z はアセット原点が中心にあるので「高さの半分」＝接地する高さ。
    work_positions: list[np.ndarray] = field(
        default_factory=lambda: [
            np.array([0.68, -0.16, 0.0263]),
            np.array([0.74, 0.02, 0.0263]),
            np.array([0.66, 0.18, 0.0263]),
        ]
    )
    # 並べ直す先（origin から pitch ずつずらして work と同数のスロットを作る）
    slot_origin: np.ndarray = field(
        default_factory=lambda: np.array([0.50, 0.06, 0.0263])
    )
    slot_pitch: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.14, 0.0])
    )

    above_height: float = 0.30  # ワークの上空へ逃がす高さ

    @property
    def slot_positions(self) -> list[np.ndarray]:
        """ワークと同数のスロット座標を一列に並べて返す。"""
        return [
            self.slot_origin + i * self.slot_pitch
            for i in range(len(self.work_positions))
        ]


@dataclass(frozen=True)
class CameraConfig:
    """物体検出に使うトップダウン RGB-D カメラの設定。

    #02 と同じ自前カメラですが、ワークとスロットの両方が画角に入るよう
    少し高く・広く見下ろします。
    """

    prim_path: str = "/World/PickCam"
    # ワークとスロットをまとめて見下ろせる位置。ここから真下(-Z)を見る。
    position: tuple = (0.62, 0.09, 1.45)
    resolution: tuple = (1280, 720)  # (幅, 高さ)

    # レンズ（ピンホールモデルの内部パラメータに使う）。USD カメラの既定値。
    focal_length: float = 24.0          # 焦点距離 [mm]
    horizontal_aperture: float = 20.955  # 横センサー幅 [mm]

    # 赤検出のしきい値（RGB は 0..255 を想定）。
    # 照明で色が薄くなっても効くよう「R が G・B より突出しているか」で判定する。
    red_min: float = 90.0        # R がこれ以上（暗すぎる赤を除外）
    red_dominance: float = 55.0  # R が G と B の大きい方より、これ以上大きいこと
    min_pixels: int = 20         # これ未満のブロブはノイズとして捨てる
    max_blobs: int = 8           # 大きい順にこの数まで採用する（暴走防止）

    debug: bool = True  # 検出の途中経過（画素数・重心・世界座標）をログ出力


@dataclass(frozen=True)
class IKConfig:
    """減衰最小二乗ヤコビアンIKのゲイン・制限。"""

    damping: float = 0.08
    step_scale: float = 0.5
    max_joint_delta: float = 0.05  # 1ステップの関節変化上限 [rad]
    ori_gain: float = 0.6
    max_pos_err: float = 0.06  # 位置誤差のクリップ [m]
    limit_margin: float = 0.02  # 関節可動域の内側マージン [rad]


@dataclass(frozen=True)
class ControlConfig:
    """フェーズ制御とタスク順序のしきい値。"""

    ee_threshold: float = 0.02  # 目標到達とみなす距離 [m]
    setpoint_speed: float = 0.008  # 目標点をなめらかに動かす速度
    warmup_frames: int = 90  # 開始時に初期姿勢を保持するフレーム数
    observe_frames: int = 60  # サイクル頭で、ホーム姿勢に戻して視界を確保する時間
    min_phase_frames: int = 40  # フェーズ切替を許すまでの最小フレーム数
    # 検出した複数ワークをどの順で処理するか
    #   "near" … ロボット基準に近い順（既定）
    #   "far"  … 遠い順
    #   "y"    … Y座標の昇順（手前から奥へ、など見た目で分かりやすい順）
    #   それ以外 … 検出された順（ブロブの大きい順）
    pick_order: str = "near"
    loop: bool = True  # 全ワーク完了後、ワーク⇄スロットを入れ替えて繰り返す


@dataclass(frozen=True)
class Config:
    """全設定の集約。"""

    robot: RobotConfig
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    control: ControlConfig = field(default_factory=ControlConfig)


def build_default_config() -> Config:
    """リポジトリ共有の assets/（1つ上の階層）から USD を探して設定を作る。"""
    repo_root = Path(__file__).resolve().parent.parent
    usd_path = repo_root / "assets" / "crx20ia_l+Robotiq_2F_85_edit+rsd455.usd"
    return Config(robot=RobotConfig(usd_path=usd_path))


# ==============================================================================
# 3. フェーズ定義（Enum）※1ワークあたりの手順。#02 と同一
# ==============================================================================
class Phase(IntEnum):
    """ピック＆プレースの手順。値はそのまま進行順を表す。"""

    APPROACH_ABOVE = 0  # 上空へ移動
    DESCEND = 1  # 下降して接近
    GRASP = 2  # 把持（その場で閉じる）
    LIFT = 3  # 持ち上げ
    TRANSPORT = 4  # 置き先へ搬送
    PLACE_DOWN = 5  # 下降して設置（ワークは保持したまま）
    RELEASE = 6  # その場でグリッパーを開いて離す
    RETREAT = 7  # 真上へ退避（ワークを弾かないよう離れる）


@dataclass(frozen=True)
class PhaseSpec:
    """各フェーズの表示名・グリッパー開閉・タイムアウト。"""

    label: str
    close_gripper: bool
    timeout: int


PHASE_SPECS: dict[Phase, PhaseSpec] = {
    Phase.APPROACH_ABOVE: PhaseSpec("approach above", False, 700),
    Phase.DESCEND: PhaseSpec("descend", False, 360),
    Phase.GRASP: PhaseSpec("grasp", True, 200),
    Phase.LIFT: PhaseSpec("lift", True, 300),
    Phase.TRANSPORT: PhaseSpec("transport", True, 500),
    Phase.PLACE_DOWN: PhaseSpec("place down", True, 360),  # 保持したまま下降
    Phase.RELEASE: PhaseSpec("release", False, 200),  # その場で開く
    Phase.RETREAT: PhaseSpec("retreat up", False, 300),  # 開いたまま上へ逃げる
}


# ==============================================================================
# 4. JacobianIK（減衰最小二乗ヤコビアンIK）※ #01 と同一
# ==============================================================================
class JacobianIK:
    """減衰最小二乗（DLS）法によるヤコビアンIKソルバ。

    ヤコビアンが得られる ``ik_link`` と、実際に狙う手先 ``grip_frame`` の
    位置ずれを skew 補正し、位置3 + 姿勢3 の6自由度タスクを1ステップ解く。
    姿勢は手先の Z 軸を常に真下へ向けるよう拘束する。
    """

    def __init__(
        self,
        articulation: Articulation,
        grip_frame: XformPrim,
        ik_link: XformPrim,
        jac_index: int,
        dof_lower: np.ndarray,
        dof_upper: np.ndarray,
        cfg: IKConfig,
        arm_dof: int,
    ) -> None:
        self._art = articulation
        self._grip = grip_frame
        self._link = ik_link
        self._jac_index = jac_index
        self._lower = dof_lower
        self._upper = dof_upper
        self._cfg = cfg
        self._arm_dof = arm_dof

    def solve(self, target: np.ndarray) -> None:
        """手先を ``target`` へ近づける1ステップ分の関節目標を送る。

        大まかな流れ:
          1. 現在の手先位置・姿勢を取得し、目標との誤差（位置3+姿勢3）を作る
          2. ヤコビアン J（関節速度→手先速度の対応表）を取得する
          3. 減衰最小二乗で「誤差を減らす関節変化 dq」を逆算する
          4. dq を安全のためクリップし、関節目標として送る
        """
        arm = self._arm_dof
        cfg = self._cfg

        grip_pos_wp, grip_ori_wp = self._grip.get_world_poses()
        grip_pos = grip_pos_wp.numpy()[0].astype(float)
        grip_rot = quat_to_rot(grip_ori_wp.numpy()[0])
        link_pos = self._link.get_world_poses()[0].numpy()[0].astype(float)

        # 位置誤差（大きすぎる場合はクリップして暴れを防ぐ）
        pos_err = target - grip_pos
        dist = float(np.linalg.norm(pos_err))
        if dist > cfg.max_pos_err:
            pos_err = pos_err / dist * cfg.max_pos_err

        # 姿勢誤差: 手先Z軸を真下(-Z)へ向ける
        ori_err = np.cross(grip_rot[:, 2], _DOWN)

        # ヤコビアンは ik_link 基準で得られる。実際に狙う grip_frame は
        # そこから r だけズレているので、r 分だけ基準点を移す（skew補正）。
        # 剛体では v_grip = v_link + ω × r。外積 ω×r を -skew(r)·ω と書けるので、
        # 角速度行 jac_arm[3:6] を使って位置行を補正する。
        jac = self._art.get_jacobian_matrices().numpy()[0, self._jac_index]
        jac_arm = jac[:, :arm]
        r = grip_pos - link_pos
        jac_grip_pos = jac_arm[:3, :] - skew(r) @ jac_arm[3:6, :]

        # 位置3行 + 姿勢3行を縦に積んで 6×armDOF のタスクヤコビアンにする
        task_jac = np.vstack([jac_grip_pos, jac_arm[3:6, :]])
        task_err = np.concatenate([pos_err, cfg.ori_gain * ori_err])

        # 減衰最小二乗解 dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
        # λ²I（damping）を足すことで特異点近傍でも発散せず安定して解ける。
        damping = np.eye(6) * (cfg.damping ** 2)
        try:
            dq = task_jac.T @ np.linalg.solve(task_jac @ task_jac.T + damping, task_err)
        except np.linalg.LinAlgError:
            log.warning("IK: no solution (near singularity)")
            return

        # いきなり大きく動かさないよう、全体を縮小し（step_scale）、
        # さらに1関節あたりの最大変化量（max_joint_delta）で頭打ちにする。
        dq *= cfg.step_scale
        peak = float(np.abs(dq).max())
        if peak > cfg.max_joint_delta:
            dq *= cfg.max_joint_delta / peak

        current = self._art.get_dof_positions().numpy().flatten()[:arm]
        arm_targets = np.clip(
            current + dq,
            self._lower[:arm] + cfg.limit_margin,
            self._upper[:arm] - cfg.limit_margin,
        )
        self._art.set_dof_position_targets(
            wp.array(arm_targets.tolist(), dtype=wp.float32),
            dof_indices=list(range(arm)),
        )


# ==============================================================================
# 5. label_blobs（赤マスクを「ワーク1個ずつ」に切り分ける）
# ==============================================================================
def label_blobs(mask: np.ndarray, max_iter: int = 1024) -> np.ndarray:
    """マスクを連結成分（ブロブ）ごとの整数ラベルに分ける。背景は -1。

    OpenCV も scipy も使わず numpy だけでやる。考え方はとても素直で、

      1. マスク内の全画素に「自分だけのユニークな番号」を配る
      2. 4近傍の最大値を取り込む、を変化がなくなるまで繰り返す

    つながっている画素は、その塊で最大の番号にだんだん染まっていき、
    最終的に「1つの塊 = 1つのラベル」になる（ラベル伝播 / flood fill）。
    番号は1回の反復で1画素ずつ広がるので、反復回数はブロブの直径ぶん。
    """
    labels = np.full(mask.shape, -1, dtype=np.int64)
    labels[mask] = np.arange(int(mask.sum()))

    height, width = mask.shape
    for _ in range(max_iter):
        # 外周を -1 で囲ってから4方向へずらす（境界の場合分けを消すため）
        padded = np.full((height + 2, width + 2), -1, dtype=np.int64)
        padded[1:-1, 1:-1] = labels
        neighbor_max = np.maximum.reduce([
            padded[:-2, 1:-1],  # 上
            padded[2:, 1:-1],   # 下
            padded[1:-1, :-2],  # 左
            padded[1:-1, 2:],   # 右
        ])
        updated = np.where(mask, np.maximum(labels, neighbor_max), -1)
        if np.array_equal(updated, labels):
            return labels  # 変化なし = 伝播が収束した
        labels = updated

    log.warning("label_blobs: not converged within %d iterations", max_iter)
    return labels


# ==============================================================================
# 6. WorkDetector（RGB-Dカメラ＋色検出で「複数個」の世界座標を推定）
# ==============================================================================
class WorkDetector:
    """トップダウン RGB-D カメラで赤いワークを **すべて** 検出する。

    #02 の CubeDetector との違いは1点だけ。赤マスクの重心を1つ求める代わりに、
    マスクを連結成分に分けて **ブロブごとに** 重心と深度を求める。
    逆投影（ピクセル+深度 → 世界座標）は #02 と同じ自前ピンホールモデル。
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.camera = None
        self._xform: XformPrim | None = None  # カメラの実際の世界姿勢を読む
        self._fx = 0.0  # ピクセル単位の焦点距離（正方ピクセルなので fx=fy）
        self._cx = 0.0  # 主点（画像中心）
        self._cy = 0.0

    def setup(self) -> None:
        """カメラ Prim を生成する（ステージ構築時に1度だけ呼ぶ）。"""
        try:
            from isaacsim.sensors.camera import Camera
        except ImportError as e:  # 環境によってモジュール名が異なる場合
            raise ImportError(
                "Failed to import the camera module. Adjust "
                "WorkDetector.setup / initialize to match your Isaac Sim "
                "camera API (see README)."
            ) from e

        # 位置はワークスペースの真上に置く。向きは後で USD 側で真下に固定する
        # （Isaac の Camera は orientation=単位 だと「+X を見る」規約なので、
        #  ここでは指定せず _force_look_down で Prim 姿勢を直接上書きする）。
        self.camera = Camera(
            prim_path=self.cfg.prim_path,
            position=np.array(self.cfg.position, dtype=float),
            resolution=self.cfg.resolution,
        )

    def _force_look_down(self) -> None:
        """カメラ Prim の姿勢を USD で「単位＝真下向き」に上書きする。

        USD カメラはローカル -Z を視線とするので、単位姿勢なら世界の -Z
        （真下）を向く。逆投影側は実際の Prim 姿勢を読むので整合する。
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.cfg.prim_path)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in self.cfg.position]))
        # Isaac が作る orient は double 精度なので合わせる（float だと精度不一致で例外）
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(1.0, 0.0, 0.0, 0.0)  # 単位 = 真下向き
        )

    def initialize(self) -> None:
        """再生開始後にカメラを初期化し、深度出力と内部パラメータを整える。"""
        cfg = self.cfg
        self.camera.initialize()
        # 深度（画像面までの距離＝光軸方向の垂直距離）をフレームに追加する
        self.camera.add_distance_to_image_plane_to_frame()
        # カメラを真下向きに固定する（Isaac の向き規約を上書き）
        self._force_look_down()

        width, height = cfg.resolution
        # 縦横のピクセルを正方にそろえる（縦アパーチャを解像度比で設定）。
        # これをやらないと Y 方向だけ倍率がずれて逆投影が狂う。
        try:
            self.camera.set_focal_length(cfg.focal_length)
            self.camera.set_horizontal_aperture(cfg.horizontal_aperture)
            self.camera.set_vertical_aperture(cfg.horizontal_aperture * height / width)
        except Exception as e:  # API 差異があっても既定値で続行
            log.warning("Failed to set some camera intrinsics: %s", e)

        # ピクセル単位の焦点距離と主点（ピンホールモデルの内部パラメータ）
        self._fx = cfg.focal_length * width / cfg.horizontal_aperture
        self._cx = width / 2.0
        self._cy = height / 2.0
        # カメラの実際の世界姿勢を読むための XformPrim
        self._xform = XformPrim(paths=cfg.prim_path)

        if cfg.debug:
            pos_wp, ori_wp = self._xform.get_world_poses()
            log.info(
                "camera pose: pos=%s quat(wxyz)=%s fx=%.1f",
                np.round(pos_wp.numpy()[0], 3),
                np.round(ori_wp.numpy()[0], 3),
                self._fx,
            )

    def _unproject(self, u: float, v: float, d: float) -> np.ndarray:
        """(u, v, 深度) を世界座標へ逆投影する（自前ピンホールモデル）。

        カメラ座標での光線方向 × 深度でカメラ座標の点を作る。
          x_c = (u - cx)/fx * d      … 右方向(+X)
          y_c = -(v - cy)/fx * d     … 上方向(+Y)。画像vは下向きなので符号反転
          z_c = -d                   … 前方は -Z（USDカメラ規約）
        それをカメラの世界姿勢 R,t で世界座標へ移す。
        """
        x_c = (u - self._cx) / self._fx * d
        y_c = -(v - self._cy) / self._fx * d
        z_c = -d

        cam_pos_wp, cam_ori_wp = self._xform.get_world_poses()
        cam_pos = cam_pos_wp.numpy()[0].astype(float)
        cam_rot = quat_to_rot(cam_ori_wp.numpy()[0])
        return (cam_pos + cam_rot @ np.array([x_c, y_c, z_c])).astype(float)

    def detect(self) -> list[np.ndarray]:
        """現在のフレームから赤いワーク全部の世界座標リストを返す。

        1個も見つからなければ空リスト。並び順はブロブの大きい順。
        """
        cfg = self.cfg
        rgba = np.asarray(self.camera.get_rgba())
        frame = self.camera.get_current_frame()
        if rgba.size == 0 or frame is None:
            if cfg.debug:
                log.info("detect: image not ready yet (rgba.size=%d)", rgba.size)
            return []
        depth = frame.get("distance_to_image_plane")
        if depth is None:
            if cfg.debug:
                log.info("detect: depth not ready yet")
            return []
        depth = np.asarray(depth)

        # --- 1) 赤いピクセルを抜き出す（#02 と同じしきい値処理）-------------
        rgb = rgba[..., :3].astype(float)
        if rgb.max() <= 1.0:  # 0..1 で返る環境向けに 0..255 へスケール
            rgb = rgb * 255.0
        red = rgb[..., 0]
        green = rgb[..., 1]
        blue = rgb[..., 2]
        # 「赤の優位性」= R が G・B の大きい方をどれだけ上回るか。
        # 明るさ（照明・トーンマップ）に左右されにくい。
        redness = red - np.maximum(green, blue)
        mask = (red > cfg.red_min) & (redness > cfg.red_dominance)

        ys, xs = np.nonzero(mask)
        if cfg.debug:
            log.info(
                "detect diag: shape=%s redness_max=%.0f red_px=%d",
                rgba.shape, float(redness.max()), int(xs.size),
            )
        if xs.size < cfg.min_pixels:
            return []

        # --- 2) マスクを連結成分（ブロブ）に分ける -------------------------
        # 全画面でラベル伝播すると無駄なので、赤画素を囲む矩形だけに切り出す。
        # 切り出しても連結関係は変わらない（矩形の外は全部背景だから）。
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub_mask = mask[y0:y1, x0:x1]
        labels = label_blobs(sub_mask)

        # --- 3) ブロブごとに重心 (u,v) と深度 d を求める -------------------
        blobs: list[tuple[int, float, float, float]] = []  # (画素数, u, v, d)
        ids, counts = np.unique(labels[sub_mask], return_counts=True)
        for blob_id, count in zip(ids.tolist(), counts.tolist()):
            if count < cfg.min_pixels:  # 小さすぎる塊はノイズとして捨てる
                continue
            bys, bxs = np.nonzero(labels == blob_id)
            # 中央値は外れ値に強い（縁のにじみ・ハイライトに引っぱられない）
            u = float(np.median(bxs)) + x0
            v = float(np.median(bys)) + y0
            d = float(np.median(depth[bys + y0, bxs + x0]))
            if not np.isfinite(d) or d <= 0.0:
                continue
            blobs.append((int(count), u, v, d))

        # 大きいブロブほど信頼できるので、大きい順に max_blobs 個だけ使う
        blobs.sort(key=lambda b: -b[0])
        blobs = blobs[: cfg.max_blobs]

        # --- 4) 逆投影して世界座標にする ----------------------------------
        found = [self._unproject(u, v, d) for _, u, v, d in blobs]
        if cfg.debug:
            for (count, u, v, d), world in zip(blobs, found):
                log.info(
                    "  blob: px=%d u=%.1f v=%.1f d=%.3f -> world=(%.3f, %.3f, %.3f)",
                    count, u, v, d, world[0], world[1], world[2],
                )
        return found


# ==============================================================================
# 7. MultiPickPlace（タスクキュー＋フェーズ状態機械）
# ==============================================================================
class MultiPickPlace:
    """検出した複数ワークを1つずつ順番にピック＆プレースする状態機械。

    #02 との違いは「フェーズ状態機械の外側にタスクキューがある」こと。

        サイクル頭: ホーム姿勢へ戻す → 全ワークを検出 → 順序を決めてキューへ
        ワーク i  : 8フェーズ（上空→下降→把持→持上→搬送→設置→解放→退避）
        ワーク完了: work_idx を1つ進めて、キューが空になるまで繰り返す
        全部完了  : ワーク側とスロット側を入れ替えて次サイクル（loop=True のとき）
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        # シーン要素（setup_scene / initialize で確定）
        self.articulation: Articulation | None = None
        self.grip_frame: XformPrim | None = None
        self.works: list[XformPrim] = []
        self.ik: JacobianIK | None = None
        self.detector = WorkDetector(cfg.camera)
        self.finger_idx = 0
        self.initial_pos: np.ndarray | None = None

        # 2つの配置（バラバラのワーク側 / 一列のスロット側）を毎サイクル入れ替える
        self._layouts = [
            [p.copy() for p in cfg.scene.work_positions],
            [p.copy() for p in cfg.scene.slot_positions],
        ]
        self._src_idx = 0  # 今ワークが置かれている側（検出失敗時のフォールバック用）

        # タスクキュー（カメラで検出したピック位置を処理順に並べたもの）
        self.queue: list[np.ndarray] = []
        self.work_idx = 0

        # 状態
        self.phase = Phase.APPROACH_ABOVE
        self.step = 0
        self.cycle = 0
        self.warmup = cfg.control.warmup_frames
        self.need_detect = True  # 各サイクル頭でカメラ検出するフラグ
        self.observe_left = cfg.control.observe_frames
        self._home_from: np.ndarray | None = None  # ホーム復帰の補間開始姿勢
        self.setpoint: np.ndarray | None = None
        self.lift_target: np.ndarray | None = None
        self.retreat_target: np.ndarray | None = None
        self.done = False

    @property
    def place_points(self) -> list[np.ndarray]:
        """今サイクルの置き先（ワーク側の反対）。"""
        return self._layouts[1 - self._src_idx]

    # --- シーン構築 -----------------------------------------------------------
    def setup_scene(self) -> None:
        set_camera_view(
            eye=[1.8, 1.2, 1.3],
            target=[0.6, 0.0, 0.3],
            camera_prim_path="/OmniverseKit_Persp",
        )
        GroundPlane("/World/GroundPlane", positions=[[0.0, 0.0, 0.0]])
        self._create_works()
        self.detector.setup()  # 検出用カメラを生成
        self.articulation = Articulation(self.cfg.robot.robot_path)
        self.grip_frame = XformPrim(paths=self.cfg.robot.grip_frame_path)

    def _create_works(self) -> None:
        """ワーク位置に Isaac Sim 標準アセットを必要な数だけ参照して置く。

        アセット（YCB の赤いスポンジブロック）は見た目だけの USD で、
        **物理が何も付いていない**（RigidBody も Collider も無い）。そのため
        参照した後に自分で
          - メッシュへ CollisionAPI ＋ 凸包近似（convexHull）
          - ルートへ RigidBodyAPI ＋ 質量
        を与える。ここを付け忘れると、つかんでも手をすり抜けて落ちない。
        """
        scene = self.cfg.scene
        stage = omni.usd.get_context().get_stage()

        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError(
                "Isaac Sim asset server is unreachable. "
                "Standard assets are downloaded at runtime, so an internet "
                "connection (or a local asset pack) is required. "
                "See 03_multi_picking/README.md."
            )
        url = assets_root + scene.work_asset

        for i, position in enumerate(scene.work_positions):
            path = f"/World/Work_{i}"
            stage_utils.add_reference_to_stage(url, path)
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"Failed to reference the work asset: {url}")

            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))

            # 参照したアセット内のメッシュすべてに衝突形状を付ける。
            # 凸包（convexHull）は箱形のワークにぴったりで、計算も軽い。
            for descendant in Usd.PrimRange(prim):
                if descendant.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(descendant)
                    UsdPhysics.MeshCollisionAPI.Apply(descendant).CreateApproximationAttr(
                        UsdPhysics.Tokens.convexHull
                    )
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(scene.work_mass)

            self.works.append(XformPrim(paths=path))

    # --- 再生開始後の初期化 ---------------------------------------------------
    def initialize(self) -> None:
        robot = self.cfg.robot
        dof_names = list(self.articulation.dof_names)
        link_names = list(self.articulation.link_names)
        log.info("DOFs: %s", dof_names)
        log.info("Links: %s", link_names)

        self.finger_idx = dof_names.index(robot.gripper_joint)

        # ヤコビアン配列はベースリンクを含まないため、リンク番号から1引いた
        # ものが対応するヤコビアンの行インデックスになる。
        link_index = link_names.index(robot.gripper_base_link)
        jac_index = link_index - 1
        ik_link = XformPrim(paths=self.articulation.link_paths[0][link_index])

        self.initial_pos = (
            self.articulation.get_dof_positions().numpy().flatten().astype(float)
        )
        self.articulation.set_dof_position_targets(self.initial_pos.tolist())

        limits = self.articulation.get_dof_limits()
        dof_lower = limits[0].numpy().flatten().astype(float)
        dof_upper = limits[1].numpy().flatten().astype(float)

        self.ik = JacobianIK(
            articulation=self.articulation,
            grip_frame=self.grip_frame,
            ik_link=ik_link,
            jac_index=jac_index,
            dof_lower=dof_lower,
            dof_upper=dof_upper,
            cfg=self.cfg.ik,
            arm_dof=robot.arm_dof,
        )

        self.detector.initialize()

    # --- タスクキューの構築（カメラ検出 → 処理順の決定） ----------------------
    def _sort_works(self, works: list[np.ndarray]) -> list[np.ndarray]:
        """検出したワークを ``pick_order`` に従って処理順に並べる。"""
        order = self.cfg.control.pick_order
        if order == "near":  # ロボット基準（原点）に近い順
            return sorted(works, key=lambda p: float(np.hypot(p[0], p[1])))
        if order == "far":
            return sorted(works, key=lambda p: -float(np.hypot(p[0], p[1])))
        if order == "y":
            return sorted(works, key=lambda p: float(p[1]))
        return list(works)  # 検出順（ブロブの大きい順）のまま

    def _build_queue(self) -> None:
        """カメラで全ワークを検出し、処理順に並べてキューに積む。

        検出できなければ既知の配置へフォールバックして停止しない（#02 と同じ方針）。
        Z は接地高さが分かっている前提で既知値を使い、XY のみカメラから得る。
        """
        sources = self._layouts[self._src_idx]
        nominal_z = float(sources[0][2])
        try:
            found = self.detector.detect()
        except Exception as e:  # カメラ API 差異などで落ちても停止させない
            log.warning("exception during detection: %s", e)
            found = []

        # 置き先の数を超えては運べないので、そこで頭打ちにする
        capacity = len(self.place_points)
        if found:
            picks = [np.array([p[0], p[1], nominal_z]) for p in found][:capacity]
            log.info("detected %d work(s) (expected %d)", len(found), len(sources))
            if len(found) > capacity:
                log.warning(
                    "more works than place points -> handling only %d", capacity
                )
        else:
            picks = [p.copy() for p in sources]
            log.warning("no work detected -> fallback to known layout")

        self.queue = self._sort_works(picks)
        self.work_idx = 0
        for i, p in enumerate(self.queue):
            log.info("  queue[%d]: x=%.3f y=%.3f", i, p[0], p[1])

    # --- 目標位置 -------------------------------------------------------------
    def _work_pos(self) -> np.ndarray:
        """今処理しているワークのピック位置（カメラ検出値）。"""
        return self.queue[self.work_idx]

    def _place_pos(self) -> np.ndarray:
        """今処理しているワークの置き先（i番目のワーク → i番目のスロット）。"""
        return self.place_points[self.work_idx]

    def _target(self) -> np.ndarray:
        work = self._work_pos()
        place = self._place_pos()
        above = self.cfg.scene.above_height
        grasp_z = work[2] + 0.005
        place_z = place[2] + 0.012

        if self.phase == Phase.APPROACH_ABOVE:
            return np.array([work[0], work[1], work[2] + above])
        if self.phase in (Phase.DESCEND, Phase.GRASP):
            return np.array([work[0], work[1], grasp_z])
        if self.phase == Phase.LIFT:  # 開始時の手先XYで真上へ
            return self.lift_target
        if self.phase == Phase.TRANSPORT:
            return np.array([place[0], place[1], place[2] + above])
        if self.phase == Phase.RETREAT:  # 離した位置から真上へ退避
            return self.retreat_target
        # PLACE_DOWN / RELEASE: 置き場所へ下降し、その位置で開く
        return np.array([place[0], place[1], place_z])

    # --- グリッパー -----------------------------------------------------------
    def _set_gripper(self, target: float) -> None:
        self.articulation.set_dof_position_targets(
            wp.array([target], dtype=wp.float32), dof_indices=[self.finger_idx]
        )

    def _finger(self) -> float:
        return float(
            self.articulation.get_dof_positions().numpy().flatten()[self.finger_idx]
        )

    # --- メインステップ -------------------------------------------------------
    def forward(self) -> None:
        if self.done:
            return

        # 開始直後は初期姿勢を保持して安定させる
        if self.warmup > 0:
            self.articulation.set_dof_position_targets(self.initial_pos.tolist())
            self.warmup -= 1
            return

        # 各サイクルの頭で「観測ステージ」: ホーム姿勢へ“なめらかに”戻して
        # 腕をどけ、カメラ視界を確保してから全ワークを検出する。
        if self.need_detect:
            # 現在姿勢からホーム姿勢へ、observe_frames かけて線形に補間する。
            # 一気に目標を飛ばすとワークを弾くので、少しずつ戻す。
            if self._home_from is None:
                self._home_from = (
                    self.articulation.get_dof_positions().numpy().flatten().astype(float)
                )
            total = self.cfg.control.observe_frames
            alpha = min(1.0, (total - self.observe_left + 1) / total)
            blended = self._home_from + alpha * (self.initial_pos - self._home_from)
            self.articulation.set_dof_position_targets(blended.tolist())

            self.observe_left -= 1
            if self.observe_left <= 0:
                self._build_queue()
                self.need_detect = False
                self.observe_left = self.cfg.control.observe_frames
                self._home_from = None
            return

        spec = PHASE_SPECS[self.phase]

        # フェーズ開始時の初期化
        if self.step == 0:
            log.info(
                "work %d/%d  phase %d: %s",
                self.work_idx + 1, len(self.queue), int(self.phase), spec.label,
            )
            self.setpoint = self.grip_frame.get_world_poses()[0].numpy()[0].astype(float)
            if self.phase == Phase.LIFT:  # 持ち上げ目標を今の手先XYで固定
                gp = self.setpoint
                lift_z = self.cfg.scene.work_height / 2 + self.cfg.scene.above_height
                self.lift_target = np.array([gp[0], gp[1], lift_z])
            if self.phase == Phase.RETREAT:  # 退避目標を今の手先XYの真上に固定
                gp = self.setpoint
                retreat_z = self.cfg.scene.work_height / 2 + self.cfg.scene.above_height
                self.retreat_target = np.array([gp[0], gp[1], retreat_z])

        # グリッパー制御
        robot = self.cfg.robot
        self._set_gripper(robot.gripper_close if spec.close_gripper else robot.gripper_open)

        # 把持／解放フェーズ以外はアームを動かす（この2つはその場でグリッパーだけ動かす）
        if self.phase not in (Phase.GRASP, Phase.RELEASE):
            # 最終目標へ一気に向かわず、追従点 setpoint を毎フレーム少しずつ
            # (speed 分だけ) 前進させることで、なめらかな軌道にする。
            target = self._target()
            to = target - self.setpoint
            dist = float(np.linalg.norm(to))
            speed = self.cfg.control.setpoint_speed
            if dist > speed:
                self.setpoint = self.setpoint + to / dist * speed
            else:
                self.setpoint = target.copy()
            # 少し進めた追従点へIKを1ステップだけ解く
            self.ik.solve(self.setpoint)

        self.step += 1

        # フェーズ遷移判定
        if self._converged() or self.step >= spec.timeout:
            self._advance()

    def _advance(self) -> None:
        self.step = 0

        # 途中フェーズなら次へ進むだけ
        if self.phase + 1 < len(Phase):
            self.phase = Phase(self.phase + 1)
            return

        # 最終フェーズ（RETREAT）完了 = 1ワーク分の処理が終了
        log.info("work %d/%d placed", self.work_idx + 1, len(self.queue))
        self.work_idx += 1

        # キューに次のワークが残っていれば、検出はやり直さず先頭フェーズへ戻る
        # （＝これが「順次ピッキング」の本体）
        if self.work_idx < len(self.queue):
            self.phase = Phase.APPROACH_ABOVE
            return

        # キューが空 = 1サイクル終了
        self.cycle += 1
        if not self.cfg.control.loop:
            self.done = True
            log.info("all works placed! (%d cycle(s))", self.cycle)
            return

        # ループ: ワーク側とスロット側を入れ替え、次サイクル頭で再び全検出する
        # （^= 1 は 0↔1 を交互に切り替えるトグル）
        self._src_idx ^= 1
        self.phase = Phase.APPROACH_ABOVE
        self.work_idx = 0  # キューは次の観測ステージで積み直す
        self.need_detect = True
        log.info("cycle %d done -> starting next cycle", self.cycle)

    def _converged(self) -> bool:
        # 入った直後の1フレームだけの誤判定を防ぐため、最低フレーム数は待つ
        if self.step < self.cfg.control.min_phase_frames:
            return False
        if self.phase == Phase.GRASP:  # 把持: 指が閉じきったか
            return self._finger() >= self.cfg.robot.gripper_close - 0.02
        if self.phase == Phase.RELEASE:  # 解放: 指が開ききったか
            return self._finger() <= self.cfg.robot.gripper_open + 0.02
        grip_pos = self.grip_frame.get_world_poses()[0].numpy()[0].astype(float)
        dist = float(np.linalg.norm(grip_pos - self._target()))
        return dist < self.cfg.control.ee_threshold


# ==============================================================================
# 8. main
# ==============================================================================
def _wait_for_robot(robot_path: str, max_frames: int = 300) -> None:
    """ロボットの Prim が構成されるまで数フレーム待つ。"""
    for _ in range(max_frames):
        simulation_app.update()
        stage = omni.usd.get_context().get_stage()
        if stage and stage.GetPrimAtPath(robot_path).IsValid():
            return


def main() -> None:
    cfg = build_default_config()
    if not cfg.robot.usd_path.exists():
        raise FileNotFoundError(
            f"USD not found: {cfg.robot.usd_path}\n"
            "The USD is not bundled in this repository. "
            "See assets/README.md to prepare it."
        )

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")

    opened, stage = stage_utils.open_stage(str(cfg.robot.usd_path))
    if not opened or stage is None:
        raise RuntimeError(f"Failed to open USD: {cfg.robot.usd_path}")
    stage.Load()

    _wait_for_robot(cfg.robot.robot_path)

    scene = MultiPickPlace(cfg)
    scene.setup_scene()
    simulation_app.update()

    app_utils.play()
    simulation_app.update()
    scene.initialize()

    # カメラのレンダリングが安定するよう数フレーム回す
    for _ in range(30):
        simulation_app.update()

    while simulation_app.is_running():
        simulation_app.update()
        if app_utils.is_playing() and SimulationManager.is_simulating():
            scene.forward()
            if scene.done:
                break

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
