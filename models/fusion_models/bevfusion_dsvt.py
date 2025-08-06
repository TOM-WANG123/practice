from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_neck,
)
from mmdet3d.models import FUSIONMODELS

from .bevfusion import BEVFusion
from mmdet3d.models.backbones import DynamicPillarVFE3D

__all__ = ["BEVFusionDSVT"]


@FUSIONMODELS.register_module()
class BEVFusionDSVT(BEVFusion):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        encoders_lidar = encoders.pop("lidar", None)
        super().__init__(encoders, fuser, decoder, heads, **kwargs)
        # self.encoders["lidar"]["voxelize"] = DynamicPillarVFE3D

        if encoders_lidar is not None:
            voxelize_module = DynamicPillarVFE3D(**encoders_lidar["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders_lidar["backbone"]),
                    "map_to_bev": build_backbone(encoders_lidar["map_to_bev"])
                }
            )
            # self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)
            if encoders_lidar.get("bev_backbone") is not None:
                encoders_lidar["bev_backbone"] = build_backbone(encoders["lidar"]["bev_backbone"])
            # else:
            #     self.encoders["lidar"]["bev_backbone"] = None
            if encoders_lidar.get("bev_neck") is not None:
                encoders_lidar["bev_neck"] = build_neck(encoders["lidar"]["bev_neck"])
            # else:
            #     self.encoders["lidar"]["bev_neck"] = None

    def extract_lidar_features(self, x) -> torch.Tensor:
        x = torch.cat(
            [torch.cat([torch.full((pc.shape[0], 1), i, dtype=pc.dtype, device=pc.device), pc], dim=1) for i, pc in
             enumerate(x)], dim=0)
        x = self.encoders["lidar"]["voxelize"](x)
        x = self.encoders["lidar"]["backbone"](x)
        x = self.encoders["lidar"]["map_to_bev"](x)
        x = x["spatial_features"]

        if "bev_backbone" in self.encoders["lidar"]:
            x = self.encoders["lidar"]["bev_backbone"](x)
        if "bev_neck" in self.encoders["lidar"]:
            x = self.encoders["lidar"]["bev_neck"](x)

        return x
