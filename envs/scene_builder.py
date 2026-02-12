"""
Scene builder for vineyard landing environment.
Creates Genesis scene with semantic regions (soil/vineyard strips).
"""

import genesis as gs
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class VineyardSceneBuilder:
    """
    Creates Genesis scene with semantic regions:
    - Label 0: Ground plane (gray) - neutral/background
    - Label 1: Soil/dirt strips (brown) - safe landing zones
    - Label 2: Vineyard strips (green) - unsafe, penalized

    Note: gs.init() must NOT be called here. Call it once in train.py before
    constructing this builder.
    """

    SEMANTIC_LABELS = {
        'background': 0,
        'soil': 1,
        'vineyard': 2,
        'drone': 3,
    }

    # Colors for visualization
    SOIL_COLOR = (0.55, 0.36, 0.18, 1.0)      # Brown
    VINEYARD_COLOR = (0.18, 0.55, 0.18, 1.0)  # Green
    GROUND_COLOR = (0.5, 0.5, 0.5, 1.0)       # Gray

    def __init__(self, config: dict):
        self.config = config
        self.strip_width = config.get('strip_width', 2.0)
        self.strip_length = config.get('strip_length', 50.0)
        self.num_strips = config.get('num_strips', 5)
        self.image_width = config.get('image_width', 64)
        self.image_height = config.get('image_height', 64)
        self.camera_fov = config.get('camera_fov', 70)
        # seg_idx -> semantic_label mapping built after scene.build()
        self.seg_idx_to_label = {}

    def build_scene(self, n_envs: int = 1, show_viewer: bool = False):
        """
        Build Genesis scene with vineyard strips and drone.

        Args:
            n_envs: Number of parallel environments
            show_viewer: Whether to show interactive viewer

        Returns:
            scene, drone, camera, seg_idx_to_label
        """
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                substeps=2,
            ),
            vis_options=gs.options.VisOptions(
                env_separate_rigid=True,
                rendered_envs_idx=list(range(n_envs)),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -15, 20),
                camera_lookat=(0, 0, 0),
                camera_fov=60,
            ),
            show_viewer=show_viewer,
        )

        entity_to_label = {}

        # 1. Base ground plane (Label 0 - background)
        ground = scene.add_entity(
            gs.morphs.Plane(pos=(0, 0, 0)),
            surface=gs.surfaces.Default(color=self.GROUND_COLOR),
        )
        entity_to_label[ground.uid] = self.SEMANTIC_LABELS['background']

        # 2. Create alternating strips (soil=1, vineyard=2)
        total_width = self.num_strips * self.strip_width
        y_start = -total_width / 2 + self.strip_width / 2

        for i in range(self.num_strips):
            is_soil = (i % 2 == 0)
            label_name = 'soil' if is_soil else 'vineyard'
            color = self.SOIL_COLOR if is_soil else self.VINEYARD_COLOR

            strip = scene.add_entity(
                gs.morphs.Box(
                    size=(self.strip_length, self.strip_width, 0.02),
                    pos=(0, y_start + i * self.strip_width, 0.01),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=color),
            )
            entity_to_label[strip.uid] = self.SEMANTIC_LABELS[label_name]

        # 3. Add drone with propeller configuration from URDF
        drone = scene.add_entity(
            gs.morphs.Drone(
                file="assets/robots/draugas/draugas_genesis.urdf",
                pos=(0, 0, self.config.get('spawn_height_min', 8.0)),
                euler=(0, 0, 90),
                propellers_link_name=["prop0_link", "prop1_link", "prop2_link", "prop3_link"],
                propellers_spin=[1, -1, 1, -1],
            )
        )
        entity_to_label[drone.uid] = self.SEMANTIC_LABELS['drone']

        # 4. Add downward-facing camera attached to drone
        camera = scene.add_camera(
            res=(self.image_width, self.image_height),
            fov=self.camera_fov,
            GUI=False,
        )

        # Camera transform: looking down from drone
        # Rotate to face downward (-Z in world frame)
        rotation = R.from_euler('zyx', [-90, -90, 0], degrees=True)
        T = np.eye(4)
        T[:3, :3] = rotation.as_matrix()
        T[2, 3] = -0.1  # Offset slightly below drone

        camera.attach(drone.get_link("base"), T)

        # Build scene with parallel environments
        scene.build(
            n_envs=n_envs,
            env_spacing=(self.strip_length + 5, total_width + 5),
        )

        # Build seg_idx -> semantic_label mapping after scene.build()
        # segmentation_idx_dict maps seg_idx -> (entity_idx, link_idx) or entity_idx
        seg_dict = scene.segmentation_idx_dict
        idx_to_label = {uid: lbl for uid, lbl in entity_to_label.items()}
        self.seg_idx_to_label = {}

        for seg_idx, entity_info in seg_dict.items():
            if isinstance(entity_info, tuple):
                entity_idx = entity_info[0]
            else:
                entity_idx = entity_info
            if entity_idx in idx_to_label:
                self.seg_idx_to_label[seg_idx] = idx_to_label[entity_idx]

        return scene, drone, camera, self.seg_idx_to_label

    def segmentation_to_semantic(self, raw_segmentation) -> torch.Tensor:
        """
        Convert Genesis entity-based segmentation indices to semantic labels.

        Uses the seg_idx_to_label mapping built during build_scene().

        Args:
            raw_segmentation: (H, W) or (B, H, W) array/tensor with entity indices

        Returns:
            semantic_mask: torch.Tensor with labels 0/1/2
        """
        if isinstance(raw_segmentation, np.ndarray):
            raw_seg = torch.from_numpy(raw_segmentation)
        else:
            raw_seg = raw_segmentation

        semantic_mask = torch.zeros_like(raw_seg, dtype=torch.long)

        for seg_idx, label in self.seg_idx_to_label.items():
            semantic_mask[raw_seg == seg_idx] = label

        return semantic_mask
