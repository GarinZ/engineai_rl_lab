"""Record a bounded replay video from an EngineAI tracking motion NPZ.

This is the checkpoint-friendly variant of ``replay_npz.py``: it replays the
motion once in the Isaac viewport, records the virtual display with ffmpeg,
then exits.
"""

import argparse
import os
import signal
import subprocess
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a replay video from a converted motion NPZ.")
parser.add_argument("--input_file", type=str, required=True, help="Path to a local .npz motion file.")
parser.add_argument("--video_path", type=str, required=True, help="Output MP4 path.")
parser.add_argument("--robot", type=str, default="pm01", choices=["pm01", "t800"], help="Robot type to use.")
parser.add_argument("--width", type=int, default=1280, help="Recorded video width.")
parser.add_argument("--height", type=int, default=720, help="Recorded video height.")
parser.add_argument("--video_fps", type=int, default=30, help="Recorded video FPS.")
parser.add_argument("--replay_fps", type=float, default=None, help="Wall-clock replay FPS. Defaults to NPZ fps.")
parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on recorded frames.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from engineai_rl_lab.tasks.tracking.mdp.commands import MotionLoader
from engineai_rl_lab.tasks.tracking.robots.pm01 import PM01_CYLINDER_CFG
from engineai_rl_lab.tasks.tracking.robots.t800 import T800_CYLINDER_CFG

ROBOT_CFGS = {
    "pm01": PM01_CYLINDER_CFG,
    "t800": T800_CYLINDER_CFG,
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")


def _resolve_joint_indexes(robot: Articulation, motion: MotionLoader, scene: InteractiveScene):
    num_motion_joints = motion.joint_pos.shape[-1]
    num_sim_joints = robot.data.default_joint_pos.shape[-1]
    motion_joint_names = getattr(motion, "joint_names", None)

    if motion_joint_names is not None:
        robot_joint_indexes = robot.find_joints(motion_joint_names, preserve_order=True)[0]
        num_robot_joints = len(robot_joint_indexes)
        print(f"[INFO]: Replaying {num_robot_joints} named joints from motion file.")
    elif num_motion_joints == num_sim_joints:
        robot_joint_indexes = slice(None)
        num_robot_joints = num_sim_joints
        print(f"[INFO]: Replaying {num_robot_joints} joints in native simulation order.")
    else:
        robot_joint_indexes = robot.find_joints(scene.cfg.robot.joint_sdk_names, preserve_order=True)[0]
        num_robot_joints = len(robot_joint_indexes)
        print(f"[INFO]: Replaying {num_robot_joints} configured SDK joints.")

    if num_motion_joints < num_robot_joints:
        raise RuntimeError(
            f"Motion has {num_motion_joints} joint columns, but robot '{args_cli.robot}' expects "
            f"{num_robot_joints} joints."
        )
    return robot_joint_indexes, num_robot_joints


def _start_ffmpeg():
    display = os.environ.get("DISPLAY")
    if not display:
        raise RuntimeError("DISPLAY is not set. Run this script inside Xvfb or a GUI session.")

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.video_path)), exist_ok=True)
    log_path = os.path.splitext(os.path.abspath(args_cli.video_path))[0] + ".ffmpeg.log"
    log_file = open(log_path, "w", encoding="utf-8")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "x11grab",
        "-draw_mouse",
        "0",
        "-video_size",
        f"{args_cli.width}x{args_cli.height}",
        "-framerate",
        str(args_cli.video_fps),
        "-i",
        f"{display}.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        args_cli.video_path,
    ]
    print("[INFO]: Starting ffmpeg screen capture")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file, log_path


def _hide_floating_ui():
    try:
        import omni.ui as ui

        for name in ("Simulation Settings",):
            window = ui.Workspace.get_window(name)
            if window is not None:
                window.visible = False
    except Exception as exc:
        print(f"[WARN]: Could not hide floating UI windows: {exc}")


def _stop_ffmpeg(proc: subprocess.Popen, log_file, log_path: str):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5.0)
    log_file.close()
    if proc.returncode not in (0, 255):
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}. See {log_path}")


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    motion = MotionLoader(args_cli.input_file, torch.tensor([0], dtype=torch.long, device=sim.device), sim.device)
    robot_joint_indexes, num_robot_joints = _resolve_joint_indexes(robot, motion, scene)

    total_frames = int(motion.time_step_total)
    if args_cli.max_frames is not None:
        total_frames = min(total_frames, args_cli.max_frames)
    if total_frames <= 0:
        raise RuntimeError("Motion has no frames to record.")

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.video_path)), exist_ok=True)
    replay_fps = args_cli.replay_fps
    if replay_fps is None:
        replay_fps = float(np.asarray(motion.fps).reshape(-1)[0])
    replay_dt = 1.0 / replay_fps

    sim_dt = sim.get_physics_dt()
    ffmpeg_proc = None
    ffmpeg_log_file = None
    ffmpeg_log_path = ""
    try:
        # Give the viewport a few frames to warm up before starting screen capture.
        for _ in range(5):
            sim.render()
        _hide_floating_ui()

        ffmpeg_proc, ffmpeg_log_file, ffmpeg_log_path = _start_ffmpeg()
        next_frame_time = time.perf_counter()

        for frame_idx in range(total_frames):
            time_steps = torch.full((scene.num_envs,), frame_idx, dtype=torch.long, device=sim.device)

            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.body_pos_w[time_steps, 0] + scene.env_origins
            root_states[:, 3:7] = motion.body_quat_w[time_steps, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[time_steps, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[time_steps, 0]

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            joint_pos[:, robot_joint_indexes] = motion.joint_pos[time_steps][:, :num_robot_joints]
            joint_vel[:, robot_joint_indexes] = motion.joint_vel[time_steps][:, :num_robot_joints]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.write_data_to_sim()

            lookat = root_states[0, :3].detach().cpu().numpy()
            sim.set_camera_view(lookat + np.array([2.0, 2.0, 0.7]), lookat + np.array([0.0, 0.0, 0.25]))
            sim.render()
            scene.update(sim_dt)

            next_frame_time += replay_dt
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

            if frame_idx % 100 == 0:
                print(f"[INFO]: Recorded {frame_idx + 1}/{total_frames} frames")
    finally:
        if ffmpeg_proc is not None and ffmpeg_log_file is not None:
            _stop_ffmpeg(ffmpeg_proc, ffmpeg_log_file, ffmpeg_log_path)
    print(f"[INFO]: Replay video saved to {args_cli.video_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
