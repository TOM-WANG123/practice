from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from mmdet3d.models import FUSIONMODELS

from .base import Base3DFusionModel
from .bevfusion import BEVFusion

__all__ = ["BEVFusionPC"]


@FUSIONMODELS.register_module()
class BEVFusionPC(BEVFusion):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(encoders, fuser, decoder, heads, **kwargs)

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        **kwargs,
    ) -> torch.Tensor:
        # B, N, C, H, W = x.size()  # torch.Size([16, 6, 3, 256, 704])

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            **kwargs,
        )

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        if "bev_backbone" in self.encoders["camera"]:
            x = self.encoders["camera"]["bev_backbone"](x)
        if "bev_neck" in self.encoders["camera"]:
            x = self.encoders["camera"]["bev_neck"](x)

        return x
