"""Actual simulator frames; rendering must not advance physics."""

import cv2
import numpy as np
import omni.replicator.core as rep
from pxr import UsdGeom, Gf
from isaacsim.core.rendering_manager import RenderingManager
from .goals import CAMERA_EYE, CAMERA_TARGET


class Recorder:
    def __init__(self, env, path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite video: {path}")
        self.env, self.path = env, path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.index = getattr(env.args, "record_env", 0)
        # PhysX view order must not be assumed to equal clone creation order.
        # Resolve the actual recorded robot's environment root from its link path.
        self.environment_path = str(env.robot.link_paths[self.index][0]).split("/Robot/", 1)[0]
        environment_prim = env.stage.GetPrimAtPath(self.environment_path)
        if not environment_prim.IsValid() or not self.environment_path.startswith("/World/envs/env_"):
            raise RuntimeError(f"Cannot resolve recorded environment: {self.environment_path}")
        origin = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(environment_prim).ExtractTranslation())
        marker = UsdGeom.Cylinder.Define(env.stage, "/World/PPOGoal")
        marker.CreateRadiusAttr(.055)
        marker.CreateHeightAttr(.002)
        marker.AddTranslateOp().Set(Gf.Vec3d(*(origin + env.goal[self.index].cpu().numpy() * np.array([1, 1, 0]) + np.array([0, 0, .002]))))
        marker.CreateDisplayColorAttr([Gf.Vec3f(.1, .8, .2)])
        camera = UsdGeom.Camera.Define(env.stage, "/World/PPOCamera")
        # Include the whole arm and off-target motion in failed policy trials too.
        eye = Gf.Vec3d(*(origin + np.array(CAMERA_EYE)))
        target = Gf.Vec3d(*(origin + np.array(CAMERA_TARGET)))
        matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
        camera.AddTransformOp().Set(matrix)
        camera.CreateFocalLengthAttr(24.)
        self.product = rep.create.render_product(str(camera.GetPath()), (960, 640))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([self.product])
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1 / (env.dt * env.decimation), (960, 640))
        if not self.writer.isOpened():
            raise RuntimeError(f"Video writer did not open: {path}")
        for _ in range(8):
            RenderingManager.render()

    def capture(self, label):
        RenderingManager.render()
        rgb = self.annotator.get_data()
        if rgb is None or rgb.size == 0:
            raise RuntimeError("No RGB frame from simulator")
        frame = cv2.cvtColor(np.asarray(rgb)[..., :3], cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (12, 12), (948, 67), (18, 18, 18), -1)
        cv2.putText(frame, label, (24, 45), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        self.writer.release()
        self.annotator.detach([self.product])
        self.product.destroy()
