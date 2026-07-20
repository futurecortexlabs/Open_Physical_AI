# ==============================================================================
# CRX20iA/L + Robotiq 2F-85  かんたんピッキング（note記事用）
# Isaac Sim 6.0.1 対応
# ==============================================================================
#
# ■ 概要
# 固定座標に置いた赤いキューブを1個つかんで、指定した場所へ置くだけのデモです。
# カメラも学習も使いません。キューブの位置は最初から分かっている（既知）前提。
#
# ■ 構成（上から順に読める並び）
#   1. 数学ユーティリティ   … クォータニオン→回転行列, 歪対称行列
#   2. 設定（dataclass）    … ロボット / シーン / IK / 制御 のパラメータ
#   3. フェーズ定義（Enum） … 6段階のピック＆プレース手順
#   4. JacobianIK           … 減衰最小二乗ヤコビアンIKソルバ
#   5. SimplePickPlace      … フェーズ状態機械
#   6. main                 … ステージ読み込みとメインループ
#
# ■ 実行方法（Windows PowerShell）
#   C:\isaacsim\python.bat ".\simple_picking.py"
# ■ 実行方法（Linux）
#   /home/dev/isaacsim/python.sh "./simple_picking.py"
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
    format="[SIMPLE_PICKING] %(message)s",
)
log = logging.getLogger("simple_picking")

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
    )  # つかむ場所
    place_position: np.ndarray = field(
        default_factory=lambda: np.array([0.50, 0.32, 0.03])
    )  # 置く場所
    above_height: float = 0.30  # ワークの上空へ逃がす高さ


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
    min_phase_frames: int = 40  # フェーズ切替を許すまでの最小フレーム数
    loop: bool = True  # 完了後に拾い先/置き先を入れ替えて繰り返す


@dataclass(frozen=True)
class Config:
    """全設定の集約。"""

    robot: RobotConfig
    scene: SceneConfig = field(default_factory=SceneConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    control: ControlConfig = field(default_factory=ControlConfig)


def build_default_config() -> Config:
    """このスクリプトと同じ場所の assets/ から USD を探して設定を作る。"""
    usd_path = (
        Path(__file__).parent / "assets" / "crx20ia_l+Robotiq_2F_85_edit+rsd455.usd"
    ).resolve()
    return Config(robot=RobotConfig(usd_path=usd_path))


# ==============================================================================
# 3. フェーズ定義（Enum）
# ==============================================================================
class Phase(IntEnum):
    """ピック＆プレースの6段階。値はそのまま進行順を表す。"""

    APPROACH_ABOVE = 0  # 上空へ移動
    DESCEND = 1  # 下降して接近
    GRASP = 2  # 把持（その場で閉じる）
    LIFT = 3  # 持ち上げ
    TRANSPORT = 4  # 置き先へ搬送
    PLACE_DOWN = 5  # 下降して設置（キューブは保持したまま）
    RELEASE = 6  # その場でグリッパーを開いて離す


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
}


# ==============================================================================
# 4. JacobianIK（減衰最小二乗ヤコビアンIK）
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
# 5. SimplePickPlace（フェーズ状態機械）
# ==============================================================================
class SimplePickPlace:
    """1個のキューブを拾って置く 6フェーズ状態機械。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        # シーン要素（setup_scene / initialize で確定）
        self.articulation: Articulation | None = None
        self.grip_frame: XformPrim | None = None
        self.cube: XformPrim | None = None
        self.ik: JacobianIK | None = None
        self.finger_idx = 0
        self.initial_pos: np.ndarray | None = None

        # 置き場所は2点を毎サイクル入れ替えて往復させる
        # （拾い先はキューブの実座標を読むので自動で追従する）
        self._place_points = [
            cfg.scene.place_position.copy(),
            cfg.scene.cube_position.copy(),
        ]
        self._place_idx = 0

        # 状態
        self.phase = Phase.APPROACH_ABOVE
        self.step = 0
        self.cycle = 0
        self.warmup = cfg.control.warmup_frames
        self.setpoint: np.ndarray | None = None
        self.lift_target: np.ndarray | None = None
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

    # --- 目標位置 -------------------------------------------------------------
    def _cube_pos(self) -> np.ndarray:
        return self.cube.get_world_poses()[0].numpy()[0].astype(float)

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

        spec = PHASE_SPECS[self.phase]

        # フェーズ開始時の初期化
        if self.step == 0:
            log.info("Phase %d: %s", int(self.phase), spec.label)
            self.setpoint = self.grip_frame.get_world_poses()[0].numpy()[0].astype(float)
            if self.phase == Phase.LIFT:  # 持ち上げ目標を今の手先XYで固定
                gp = self.setpoint
                lift_z = self.cfg.scene.cube_size / 2 + self.cfg.scene.above_height
                self.lift_target = np.array([gp[0], gp[1], lift_z])

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

        # ループ: 拾い先/置き先を入れ替えて先頭フェーズへ戻る
        # （^= 1 は 0↔1 を交互に切り替えるトグル）
        self._place_idx ^= 1
        self.phase = Phase.APPROACH_ABOVE
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
# 6. main
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
        raise FileNotFoundError(f"USD が見つかりません: {cfg.robot.usd_path}")

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")

    opened, stage = stage_utils.open_stage(str(cfg.robot.usd_path))
    if not opened or stage is None:
        raise RuntimeError(f"USD を開けませんでした: {cfg.robot.usd_path}")
    stage.Load()

    _wait_for_robot(cfg.robot.robot_path)

    scene = SimplePickPlace(cfg)
    scene.setup_scene()
    simulation_app.update()

    app_utils.play()
    simulation_app.update()
    scene.initialize()

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
