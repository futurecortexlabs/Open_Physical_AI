# ==============================================================================
# CRX20iA/L + Robotiq 2F-85  カメラで物体検出ピッキング（note記事用 #02）
# Isaac Sim 6.0.1 対応
# ==============================================================================
#
# ■ 概要
# #01「かんたんピッキング」は、キューブの位置を座標で“教えて”いました。
# 今回はそれをやめ、**トップダウンの RGB-D カメラで赤いキューブを見つけて**、
# その世界座標を推定してからつかみに行きます。
#
# 検出は色（赤）のしきい値処理だけ。機械学習も OpenCV も使いません。
#   1. カメラで RGB 画像と深度画像を撮る
#   2. 赤いピクセルを numpy で抜き出し、重心（画像上の位置 u,v）を求める
#   3. 重心の深度 d を読み、カメラの投影を使って (u,v,d) を世界座標へ逆投影する
#   4. その XY をキューブ位置として、#01 と同じ IK＋状態機械でピック＆プレース
#
# ロボットを動かす土台（JacobianIK / 状態機械）は #01 と同一です。
# 変わったのは「キューブの位置をどこから得るか」だけ。
#
# ■ 実行方法（Windows PowerShell）※リポジトリのルートから実行
#   C:\isaacsim\python.bat ".\02_camera_picking\camera_picking.py"
# ■ 実行方法（Linux）
#   /home/dev/isaacsim/python.sh "./02_camera_picking/camera_picking.py"
# ==============================================================================

from __future__ import annotations

import logging
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
from pxr import Gf, UsdGeom, UsdPhysics

logging.basicConfig(
    level=logging.INFO,
    format="[CAMERA_PICKING] %(message)s",
)
log = logging.getLogger("camera_picking")

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
    """キューブと置き場所に関する設定。"""

    cube_size: float = 0.06
    cube_position: np.ndarray = field(
        default_factory=lambda: np.array([0.72, 0.0, 0.03])
    )  # キューブの初期位置（ロボットには“教えない”＝カメラで探す）
    place_position: np.ndarray = field(
        default_factory=lambda: np.array([0.50, 0.32, 0.03])
    )  # 置く場所
    above_height: float = 0.30  # ワークの上空へ逃がす高さ


@dataclass(frozen=True)
class CameraConfig:
    """物体検出に使うトップダウン RGB-D カメラの設定。

    USD 側のカメラに依存しないよう、ワークスペースの真上に自前のカメラを
    1台作って見下ろします（作り方は #01 でキューブを自作したのと同じ発想）。
    """

    prim_path: str = "/World/PickCam"
    # ワークスペース中心の真上。ここから真下(-Z)を見下ろす。
    position: tuple = (0.61, 0.16, 1.30)
    resolution: tuple = (1280, 720)  # (幅, 高さ)

    # レンズ（ピンホールモデルの内部パラメータに使う）。USD カメラの既定値。
    focal_length: float = 24.0          # 焦点距離 [mm]
    horizontal_aperture: float = 20.955  # 横センサー幅 [mm]

    # 赤検出のしきい値（RGB は 0..255 を想定）。
    # 照明で色が薄くなっても効くよう「R が G・B より突出しているか」で判定する。
    red_min: float = 90.0        # R がこれ以上（暗すぎる赤を除外）
    red_dominance: float = 55.0  # R が G と B の大きい方より、これ以上大きいこと
    min_pixels: int = 20         # これ未満なら「検出なし」とみなす

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
    """フェーズ制御のしきい値。"""

    ee_threshold: float = 0.02  # 目標到達とみなす距離 [m]
    setpoint_speed: float = 0.008  # 目標点をなめらかに動かす速度
    warmup_frames: int = 90  # 開始時に初期姿勢を保持するフレーム数
    observe_frames: int = 60  # 各サイクル頭で、ホーム姿勢に戻して視界を確保する時間
    min_phase_frames: int = 40  # フェーズ切替を許すまでの最小フレーム数
    loop: bool = True  # 完了後に拾い先/置き先を入れ替えて繰り返す


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
# 3. フェーズ定義（Enum）
# ==============================================================================
class Phase(IntEnum):
    """ピック＆プレースの手順。値はそのまま進行順を表す。"""

    APPROACH_ABOVE = 0  # 上空へ移動
    DESCEND = 1  # 下降して接近
    GRASP = 2  # 把持（その場で閉じる）
    LIFT = 3  # 持ち上げ
    TRANSPORT = 4  # 置き先へ搬送
    PLACE_DOWN = 5  # 下降して設置（キューブは保持したまま）
    RELEASE = 6  # その場でグリッパーを開いて離す
    RETREAT = 7  # 真上へ退避（キューブを弾かないよう離れる）


@dataclass(frozen=True)
class PhaseSpec:
    """各フェーズの表示名・グリッパー開閉・タイムアウト。"""

    label: str
    close_gripper: bool
    timeout: int


PHASE_SPECS: dict[Phase, PhaseSpec] = {
    Phase.APPROACH_ABOVE: PhaseSpec("上空へ移動", False, 700),
    Phase.DESCEND: PhaseSpec("下降して接近", False, 360),
    Phase.GRASP: PhaseSpec("把持", True, 200),
    Phase.LIFT: PhaseSpec("持ち上げ", True, 300),
    Phase.TRANSPORT: PhaseSpec("置き先へ搬送", True, 500),
    Phase.PLACE_DOWN: PhaseSpec("下降して設置", True, 360),  # 保持したまま下降
    Phase.RELEASE: PhaseSpec("グリッパーを離す", False, 200),  # その場で開く
    Phase.RETREAT: PhaseSpec("真上へ退避", False, 300),  # 開いたまま上へ逃げる
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
            log.warning("IK: 特異点により解が求まりませんでした")
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
# 5. CubeDetector（RGB-Dカメラ＋色検出でキューブの世界座標を推定）
# ==============================================================================
class CubeDetector:
    """トップダウン RGB-D カメラで赤いキューブを検出し、世界座標を返す。

    逆投影（ピクセル+深度 → 世界座標）は、外部ヘルパに頼らず自前の
    ピンホールモデルで計算する。座標規約を自分で完全に握れるので、
    ズレたときにどこを直せばよいか分かりやすい。
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
                "カメラモジュールを import できませんでした。"
                "お使いの Isaac Sim のカメラ API に合わせて "
                "CubeDetector.setup / initialize を調整してください（README 参照）。"
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
            log.warning("カメラ内部パラメータの設定に一部失敗: %s", e)

        # ピクセル単位の焦点距離と主点（ピンホールモデルの内部パラメータ）
        self._fx = cfg.focal_length * width / cfg.horizontal_aperture
        self._cx = width / 2.0
        self._cy = height / 2.0
        # カメラの実際の世界姿勢を読むための XformPrim
        self._xform = XformPrim(paths=cfg.prim_path)

        if cfg.debug:
            pos_wp, ori_wp = self._xform.get_world_poses()
            log.warning(
                "カメラ姿勢: pos=%s quat(wxyz)=%s fx=%.1f",
                np.round(pos_wp.numpy()[0], 3),
                np.round(ori_wp.numpy()[0], 3),
                self._fx,
            )

    def detect(self) -> np.ndarray | None:
        """現在のフレームから赤いキューブの世界座標 (x,y,z) を推定して返す。

        見つからなければ None。
        """
        cfg = self.cfg
        rgba = np.asarray(self.camera.get_rgba())
        frame = self.camera.get_current_frame()
        if rgba.size == 0 or frame is None:
            if cfg.debug:
                log.warning("検出: 画像がまだ取得できていません (rgba.size=%d)", rgba.size)
            return None
        depth = frame.get("distance_to_image_plane")
        if depth is None:
            if cfg.debug:
                log.warning("検出: 深度がまだ取得できていません")
            return None

        # --- 1) 赤いピクセルを抜き出す（RGB しきい値処理）-----------------
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
            # 画像内で最も赤い画素の実RGBと位置を出す（映っているかの確認用）
            ridx = int(np.argmax(redness))
            ry, rx = np.unravel_index(ridx, redness.shape)
            log.warning(
                "検出診断: shape=%s | 最赤画素 RGB=(%.0f,%.0f,%.0f) @(u=%d,v=%d) "
                "redness_max=%.0f | 閾値通過=%d",
                rgba.shape,
                float(red[ry, rx]), float(green[ry, rx]), float(blue[ry, rx]),
                int(rx), int(ry), float(redness.max()), int(xs.size),
            )
        if xs.size < cfg.min_pixels:
            return None

        # --- 2) 重心（画像上の位置 u,v）を求める（中央値で外れ値に強く）---
        u = float(np.median(xs))  # 横方向（列）
        v = float(np.median(ys))  # 縦方向（行）

        # --- 3) 深度を読む（マスク内の中央値）---------------------------
        depth = np.asarray(depth)
        d = float(np.median(depth[ys, xs]))
        if not np.isfinite(d) or d <= 0.0:
            return None

        # --- 4) (u,v,d) を世界座標へ逆投影する（自前ピンホール）----------
        # カメラ座標での光線方向 × 深度でカメラ座標の点を作る。
        #   x_c = (u - cx)/fx * d      … 右方向(+X)
        #   y_c = -(v - cy)/fx * d     … 上方向(+Y)。画像vは下向きなので符号反転
        #   z_c = -d                   … 前方は -Z（USBカメラ規約）
        # それをカメラの世界姿勢 R,t で世界座標へ移す。
        x_c = (u - self._cx) / self._fx * d
        y_c = -(v - self._cy) / self._fx * d
        z_c = -d

        cam_pos_wp, cam_ori_wp = self._xform.get_world_poses()
        cam_pos = cam_pos_wp.numpy()[0].astype(float)
        cam_rot = quat_to_rot(cam_ori_wp.numpy()[0])
        world = cam_pos + cam_rot @ np.array([x_c, y_c, z_c])

        if cfg.debug:
            log.warning(
                "検出: u=%.1f v=%.1f d=%.3f -> world=(%.3f, %.3f, %.3f)",
                u, v, d, world[0], world[1], world[2],
            )
        return world.astype(float)


# ==============================================================================
# 6. CameraPickPlace（フェーズ状態機械。キューブ位置はカメラ検出で得る）
# ==============================================================================
class CameraPickPlace:
    """カメラで見つけたキューブを拾って置く状態機械。

    #01 との違いは「キューブ位置をカメラ検出から得る」点だけ。
    IK・フェーズ・往復ループの仕組みは #01 と同じ。
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        # シーン要素（setup_scene / initialize で確定）
        self.articulation: Articulation | None = None
        self.grip_frame: XformPrim | None = None
        self.cube: XformPrim | None = None
        self.ik: JacobianIK | None = None
        self.detector = CubeDetector(cfg.camera)
        self.finger_idx = 0
        self.initial_pos: np.ndarray | None = None

        # 置き場所は2点を毎サイクル入れ替えて往復させる（#01 と同じ）
        self._place_points = [
            cfg.scene.place_position.copy(),
            cfg.scene.cube_position.copy(),
        ]
        self._place_idx = 0

        # カメラが検出したキューブ位置（初期値は既知座標をフォールバックに）
        self.detected_cube_pos = self._place_points[1].copy()

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
    def place_target(self) -> np.ndarray:
        return self._place_points[self._place_idx]

    # --- シーン構築 -----------------------------------------------------------
    def setup_scene(self) -> None:
        set_camera_view(
            eye=[1.8, 1.2, 1.3],
            target=[0.6, 0.0, 0.3],
            camera_prim_path="/OmniverseKit_Persp",
        )
        GroundPlane("/World/GroundPlane", positions=[[0.0, 0.0, 0.0]])
        self._create_cube()
        self.detector.setup()  # 検出用カメラを生成
        self.articulation = Articulation(self.cfg.robot.robot_path)
        self.grip_frame = XformPrim(paths=self.cfg.robot.grip_frame_path)

    def _create_cube(self) -> None:
        scene = self.cfg.scene
        stage = omni.usd.get_context().get_stage()
        cube = UsdGeom.Cube.Define(stage, "/World/Cube_RED")
        cube.CreateSizeAttr(scene.cube_size)
        cube.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.05, 0.02)])
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in scene.cube_position]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.05)
        self.cube = XformPrim(paths="/World/Cube_RED")

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

    # --- キューブ位置（カメラ検出） -------------------------------------------
    def _detect_cube(self) -> None:
        """カメラでキューブを検出し、self.detected_cube_pos を更新する。

        検出できなければ既知の座標（前サイクルで置いた場所）へフォールバック。
        Z は接地高さが分かっている前提で既知値を使い、XY のみカメラから得る。
        """
        fallback = self._place_points[1 - self._place_idx].copy()
        nominal_z = float(fallback[2])
        try:
            pos = self.detector.detect()
        except Exception as e:  # カメラ API 差異などで落ちても停止させない
            log.warning("検出中に例外が発生: %s", e)
            pos = None

        if pos is not None:
            self.detected_cube_pos = np.array([pos[0], pos[1], nominal_z])
            log.info("キューブ検出（カメラ）: x=%.3f, y=%.3f", pos[0], pos[1])
        else:
            self.detected_cube_pos = fallback
            log.warning("キューブ未検出 → 既知座標にフォールバック: %s", fallback)

    # --- 目標位置 -------------------------------------------------------------
    def _cube_pos(self) -> np.ndarray:
        # #01 は真値を読んでいたが、#02 はカメラ検出値を使う
        return self.detected_cube_pos

    def _target(self) -> np.ndarray:
        cube = self._cube_pos()
        place = self.place_target
        above = self.cfg.scene.above_height
        grasp_z = cube[2] + 0.005
        place_z = place[2] + 0.012

        if self.phase == Phase.APPROACH_ABOVE:
            return np.array([cube[0], cube[1], cube[2] + above])
        if self.phase in (Phase.DESCEND, Phase.GRASP):
            return np.array([cube[0], cube[1], grasp_z])
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
        # 腕をどけ、カメラ視界を確保してからキューブを検出する。
        if self.need_detect:
            # 現在姿勢からホーム姿勢へ、observe_frames かけて線形に補間する。
            # 一気に目標を飛ばすとキューブを弾くので、少しずつ戻す。
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
                self._detect_cube()
                self.need_detect = False
                self.observe_left = self.cfg.control.observe_frames
                self._home_from = None
            return

        spec = PHASE_SPECS[self.phase]

        # フェーズ開始時の初期化
        if self.step == 0:
            log.info("Phase %d: %s", int(self.phase), spec.label)
            self.setpoint = self.grip_frame.get_world_poses()[0].numpy()[0].astype(float)
            if self.phase == Phase.LIFT:  # 持ち上げ目標を今の手先XYで固定
                gp = self.setpoint
                lift_z = self.cfg.scene.cube_size / 2 + self.cfg.scene.above_height
                self.lift_target = np.array([gp[0], gp[1], lift_z])
            if self.phase == Phase.RETREAT:  # 退避目標を今の手先XYの真上に固定
                gp = self.setpoint
                retreat_z = self.cfg.scene.cube_size / 2 + self.cfg.scene.above_height
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

        # 最終フェーズ（RELEASE）完了 = 1サイクル終了
        self.cycle += 1
        if not self.cfg.control.loop:
            self.done = True
            log.info("ピッキング完了！（%d サイクル）", self.cycle)
            return

        # ループ: 拾い先/置き先を入れ替え、次サイクル頭で再びカメラ検出する
        # （^= 1 は 0↔1 を交互に切り替えるトグル）
        self._place_idx ^= 1
        self.phase = Phase.APPROACH_ABOVE
        self.need_detect = True
        log.info("サイクル %d 完了 → 次サイクル開始", self.cycle)

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
# 7. main
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
            f"USD が見つかりません: {cfg.robot.usd_path}\n"
            "USD 本体はリポジトリに同梱していません。"
            "assets/README.md を参照して用意してください。"
        )

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")

    opened, stage = stage_utils.open_stage(str(cfg.robot.usd_path))
    if not opened or stage is None:
        raise RuntimeError(f"USD を開けませんでした: {cfg.robot.usd_path}")
    stage.Load()

    _wait_for_robot(cfg.robot.robot_path)

    scene = CameraPickPlace(cfg)
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
