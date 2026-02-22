"""
Standalone camera validation script.

Run this ONCE before training to determine whether Genesis multi-env camera
rendering provides different views per environment.

Usage:
    python tools/validate_camera.py

Result determines whether Genesis multi-env camera rendering provides distinct
views when using manual camera.set_pose() to track per-env drone positions.
"""

import sys
import os
import numpy as np
import torch

# Ensure project root is on path when run from tools/ subdirectory
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import genesis as gs
from scipy.spatial.transform import Rotation as R


N_ENVS = 4
DRONE_HEIGHTS = [5.0, 7.0, 9.0, 11.0]   # Different heights -> different depth views
STRIP_WIDTH = 2.0
STRIP_LENGTH = 50.0
NUM_STRIPS = 5
IMAGE_SIZE = 64

SOIL_COLOR = (0.55, 0.36, 0.18, 1.0)
VINEYARD_COLOR = (0.18, 0.55, 0.18, 1.0)
GROUND_COLOR = (0.5, 0.5, 0.5, 1.0)


def build_validation_scene():
    """Build minimal vineyard scene for camera validation."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=list(range(N_ENVS)),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -15, 20),
            camera_lookat=(0, 0, 0),
            camera_fov=60,
        ),
        show_viewer=False,
    )

    # Ground plane
    scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, 0)),
        surface=gs.surfaces.Default(color=GROUND_COLOR),
    )

    # Alternating strips
    total_width = NUM_STRIPS * STRIP_WIDTH
    y_start = -total_width / 2 + STRIP_WIDTH / 2
    for i in range(NUM_STRIPS):
        is_soil = (i % 2 == 0)
        color = SOIL_COLOR if is_soil else VINEYARD_COLOR
        scene.add_entity(
            gs.morphs.Box(
                size=(STRIP_LENGTH, STRIP_WIDTH, 0.02),
                pos=(0, y_start + i * STRIP_WIDTH, 0.01),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=color),
        )

    # Drone at initial position (will be repositioned after build)
    drone = scene.add_entity(
        gs.morphs.Drone(
            file="assets/robots/draugas/draugas_genesis.urdf",
            pos=(0, 0, DRONE_HEIGHTS[0]),
            euler=(0, 0, 90),
            propellers_link_name=["prop0_link", "prop1_link", "prop2_link", "prop3_link"],
            propellers_spin=[1, -1, 1, -1],
        )
    )

    # Downward-facing camera attached to drone
    camera = scene.add_camera(
        res=(IMAGE_SIZE, IMAGE_SIZE),
        fov=70,
        GUI=False,
    )

    rotation = R.from_euler('zyx', [-90, -90, 0], degrees=True)
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[2, 3] = -0.1

    scene.build(
        n_envs=N_ENVS,
        env_spacing=(STRIP_LENGTH + 5, total_width + 5),
    )

    return scene, drone, camera, T, drone.get_link("base")


def set_drone_heights(drone, heights):
    """Set each drone to a different height to create diverse depth views."""
    for i, h in enumerate(heights):
        pos = np.array([[0.0, 0.0, h]])
        drone.set_pos(pos, envs_idx=np.array([i]))


def update_camera_pose(camera, drone_link, offset_T):
    """Manually update camera to follow drone, replicating move_to_attach logic."""
    import genesis.utils.geom as gu
    link_pos = drone_link.get_pos()
    link_quat = drone_link.get_quat()
    link_T = gu.trans_quat_to_T(link_pos, link_quat)
    offset_T_tensor = torch.as_tensor(offset_T, dtype=gs.tc_float, device=gs.device)
    world_T = torch.matmul(link_T, offset_T_tensor)
    camera.set_pose(transform=world_T)


def render_cameras(camera):
    """
    Attempt camera rendering using the prototype's direct render pattern.
    Falls back to start/render/stop pattern if needed.

    Returns:
        depth_array: np.ndarray of depth images
        success: bool
    """
    # Pattern 1: Direct render (prototype pattern)
    try:
        result = camera.render(depth=True, segmentation=True)
        # Result is a tuple: (rgb, depth, segmentation, ...)
        # rgb is index 0, depth is index 1
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            depth = result[1]  # depth is second element
            if depth is not None:
                return np.array(depth), True
    except Exception as e:
        print(f"  Pattern 1 (direct render) failed: {e}")

    # Pattern 2: start/render/stop pattern
    try:
        camera.start()
        result = camera.render(depth=True, segmentation=True)
        camera.stop()
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            depth = result[1]
            if depth is not None:
                return np.array(depth), True
    except Exception as e:
        print(f"  Pattern 2 (start/render/stop) failed: {e}")

    return None, False


def analyze_depth_results(depth_array, heights):
    """
    Analyze whether multi-env rendering returns distinct views.

    Returns:
        is_multi_env: bool - True if depth array has shape (N_ENVS, H, W)
        views_differ: bool - True if depth values vary significantly across envs
    """
    shape = depth_array.shape
    print(f"  Depth array shape: {shape}")

    if len(shape) == 3 and shape[0] == N_ENVS:
        is_multi_env = True
        # Check if mean depths differ (higher drone -> larger mean depth)
        mean_depths = [float(np.mean(depth_array[i])) for i in range(N_ENVS)]
        print(f"  Mean depth per env: {[f'{d:.2f}' for d in mean_depths]}")
        depth_range = max(mean_depths) - min(mean_depths)
        views_differ = depth_range > 0.5  # at least 0.5m difference in mean depth
        print(f"  Depth range across envs: {depth_range:.2f}m")
    elif len(shape) == 2:
        is_multi_env = False
        views_differ = False
        print(f"  Only single env returned (H x W shape)")
    else:
        is_multi_env = False
        views_differ = False
        print(f"  Unexpected shape: {shape}")

    return is_multi_env, views_differ


def main():
    print("=" * 60)
    print("Genesis Multi-Env Camera Validation")
    print("=" * 60)
    print(f"Testing {N_ENVS} environments at heights: {DRONE_HEIGHTS}m")
    print()

    # Initialize Genesis (standalone script — calls init itself)
    gs.init(backend=gs.gpu)

    print("Building validation scene...")
    scene, drone, camera, offset_T, base_link = build_validation_scene()

    print("Positioning drones at different heights...")
    set_drone_heights(drone, DRONE_HEIGHTS)

    print("Stepping physics once...")
    scene.step()

    print("Updating camera pose to track drone positions...")
    update_camera_pose(camera, base_link, offset_T)

    print("Rendering cameras...")
    depth_array, render_success = render_cameras(camera)

    print()
    if not render_success or depth_array is None:
        print("RESULT: FAIL")
        print("-" * 60)
        print("Camera rendering failed entirely.")
        print("Recommended action: Debug render API compatibility with this Genesis version.")
        return 1

    print("Analyzing depth results...")
    is_multi_env, views_differ = analyze_depth_results(depth_array, DRONE_HEIGHTS)

    print()
    print("=" * 60)
    if is_multi_env and views_differ:
        print("RESULT: PASS")
        print("-" * 60)
        print("Multi-env cameras work. Depth values differ across environments.")
        print("Recommended action: Multi-env camera rendering works with manual set_pose tracking.")
        print("  -> Training will use parallel camera rendering (fast path).")
    elif is_multi_env and not views_differ:
        print("RESULT: FAIL (shape OK but views identical)")
        print("-" * 60)
        print("Multi-env cameras return same view despite correct shape.")
        print("Recommended action: Use sequential rendering fallback.")
        print("  -> Render each env camera one at a time during training.")
    else:
        print("RESULT: FAIL")
        print("-" * 60)
        print("Multi-env cameras return only a single environment's view.")
        print("Recommended action: Use sequential rendering fallback.")
        print("  -> Render each env camera one at a time during training.")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
