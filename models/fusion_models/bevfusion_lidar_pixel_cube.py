from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from mmdet3d.models import FUSIONMODELS
from torch.nn import functional as F
import einops

from .base import Base3DFusionModel
from .bevfusion import BEVFusion

__all__ = ["BEVFusionLPC"]


@FUSIONMODELS.register_module()
class BEVFusionLPC(BEVFusion):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(encoders, fuser, decoder, heads, **kwargs)
        self.input_size = kwargs['input_size']

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        # feats, coords, sizes = self.voxelize(points)
        # # points -> coords
        # # -50, -50, -4 -> 0, 0, 0, 0
        # # 50, -50, -4 -> 0, 3, 0, 0
        # # -50, 50, -4 -> 0, 0, 3, 0
        # # -50, -50, 2 -> 0, 0, 0, 1

        # Step 1: 体素化点云，获得非空体素的体素坐标 coords
        feats, coords, sizes = self.voxelize(points)
        batch_size = coords[-1, 0].item() + 1

        batch_indices = coords[:, 0]
        unique_batches, counts = torch.unique(batch_indices, return_counts=True)
        counts = [c.item() for c in counts]
        max_voxels = max(counts)
        voxel_centers_padded = feats.new_zeros(batch_size, max_voxels, 3)
        for i in range(batch_size):
            mask = (batch_indices == i)
            # current_voxels = mask.sum().item()
            # assert current_voxels > 0
            # voxel_centers_padded[batch_idx, :current_voxels] = feats[mask, :3]
            voxel_centers_padded[i, :counts[i]] = feats[mask, :3]

        # vtransform = self.encoders["camera"]["vtransform"]
        # vtransform.ref_3d = voxel_centers_padded

        x = einops.rearrange(img, 'B N C H W -> B C N H W')
        # assert vtransform.top_type == 'lidar'
        camera2sensor = camera2lidar
        # reference_points_img, volume_mask = vtransform.point_sampling(camera2sensor, camera_intrinsics[..., :3, :3],
        #                                                               img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3],
        #                                                               lidar_aug_matrix)
        reference_points_img, volume_mask = self.point_sampling(camera2sensor, camera_intrinsics[..., :3, :3],
                                                                img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3],
                                                                lidar_aug_matrix, ref_3d=voxel_centers_padded)

        bs, num_points, _ = reference_points_img.shape
        valid_index_per_batch = [volume_mask[i].squeeze(-1).nonzero().squeeze(-1) for i in range(bs)]
        max_len = max([len(per_batch) for per_batch in valid_index_per_batch])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        reference_points_rebatch = reference_points_img.new_zeros([bs, max_len, 3])
        for j in range(bs):
            reference_points_rebatch[j, :len(valid_index_per_batch[j])] = reference_points_img[j, valid_index_per_batch[j]]
        reference_points_rebatch = reference_points_rebatch.view(bs, max_len, 1, 1, 3)
        reference_points_rebatch = 2 * reference_points_rebatch - 1  # torch.Size([1, 312387, 1, 1, 3])
        # need half here
        reference_points_rebatch = reference_points_rebatch.half()

        # x: b, c, n, h, w
        # reference_points_rebatch: b, max_len, 1, 1, xyz
        sampling_feats = F.grid_sample(
            x,
            reference_points_rebatch,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True)
        # bn, c, max_len, num_points_in_voxel
        sampling_feats = einops.rearrange(sampling_feats,
                                          'b c max_len 1 1 -> b max_len c',
                                          b=bs)
        feats_volume = x.new_zeros([bs, num_points, 3])
        for j in range(bs):
            feats_volume[j, valid_index_per_batch[j]] = sampling_feats[j, :len(valid_index_per_batch[j])]

        total_points = feats.shape[0]
        # need half here
        rgb_feats = feats_volume.new_zeros([total_points, 3])
        for i in range(batch_size):
            mask = (batch_indices == i)
            voxel_centers_padded[i, :counts[i]] = feats[mask, :3]
            rgb_feats[mask, :] = feats_volume[i, :counts[i]]

        feats_w_rgb = torch.cat([feats, rgb_feats], dim=-1)
        x = self.encoders["lidar"]["backbone"](feats_w_rgb, coords, batch_size, sizes=sizes)

        batch_size = x.shape[0]

        if self.decoder is not None:
            x = self.decoder["backbone"](x)
            x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for head_type, head in self.heads.items():
                if head_type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif head_type == "map":
                    losses = head(x, gt_masks_bev)
                elif head_type == "occ":
                    # occ_pred = head(x)
                    # TODO: [yz] this is so weird!
                    occ_pred = head(x, lidar_aug_matrix, lidar2ego, kwargs['occ_aug_matrix'])
                    losses = head.loss(occ_pred, kwargs['voxel_semantics'], kwargs['mask_camera'])
                else:
                    raise ValueError(f"unsupported head: {head_type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{head_type}/{name}"] = val * self.loss_scale[head_type]
                    else:
                        outputs[f"stats/{head_type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for head_type, head in self.heads.items():
                if head_type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif head_type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                elif head_type == "occ":
                    # occ_pred = head(x)
                    # TODO: [yz] this is so weird!
                    occ_pred = head(x, lidar_aug_matrix, lidar2ego, kwargs['occ_aug_matrix'])
                    occ_pred = head.get_occ(occ_pred, kwargs['voxel_semantics'])
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "occ_pred": occ_pred[k],  # already in cpu
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {head_type}")
            return outputs

    @force_fp32()
    def point_sampling(self, camera2sensor, cam2imgs, post_rots, post_trans, bda, mode='fix', ref_3d=None):
        # bda:[4,4,4], sensor2ego:[4,6,4,4], cam2imgs:[4,6,3,3], post_rots:[4,6,3,3], post_trans:[4,6,3]
        # NOTE: close tf32 here. TODO: the allow_tf32 has no effect here
        # allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        # torch.backends.cuda.matmul.allow_tf32 = False
        # torch.backends.cudnn.allow_tf32 = False

        B, N, _, _ = camera2sensor.shape
        if ref_3d.dim() == 2:
            assert ref_3d.shape[-1] == 3
            reference_points = ref_3d[None, :, :]
        elif ref_3d.dim() == 3:
            assert ref_3d.shape[-1] == 3 and ref_3d.shape[0] == B
            reference_points = ref_3d
        else:
            raise NotImplementedError

        num_points = ref_3d.shape[1]  # [1/B NP 3]
        # [(NP 3) -> (1 NP 3)] - [(B 3) -> (B 1 3)] -> (B NP 3)
        reference_points = reference_points - bda[:, :3, 3].view(B, 1, 3)
        # TODO: [yz] check torch.inverse(bda) = bda.T, use self.ref_3d @ bda replace
        # [(B 3 3) -> (B 1 3 3)] @ [(B NP 3) -> (B NP 3 1)] -> (B NP 3 1)
        # reference_points = torch.inverse(bda.view(B, 1, 3, 3)).matmul(self.ref_3d.to(bda).view(1, num_points, 3, 1))
        # register self.ref_3d as buffer
        reference_points = torch.inverse(bda[:, :3, :3].view(B, 1, 3, 3)).matmul(reference_points.unsqueeze(-1))
        # bda_inv = torch.inverse(bda.float())
        # reference_points = bda_inv.half().view(B, 1, 3, 3).matmul(self.ref_3d.to(bda).view(1, num_points, 3, 1))
        # (B NP 3 1) -> (B 1 NP 3 1) -> (B 1 NP 3)
        reference_points = reference_points.view(B, 1, num_points, 3, 1).squeeze(-1)
        # (B 1 NP 3) - (B N 1 3) -> (B N NP 3) -> (B N NP 3 1)
        reference_points = (reference_points - camera2sensor[:, :, :3, 3].view(B, N, 1, 3)).unsqueeze(-1)
        # TODO: [yz] check torch.inverse(sensor2ego_r) == sensor2ego_r.T
        # (B N 3 3)
        combine = cam2imgs.matmul(camera2sensor[:, :, :3, :3].transpose(-1, -2))
        # [(B N 3 3) -> (B N 1 3 3)] @ (B N NP 3 1) -> (B N NP 3 1) -> (B N NP 3)
        reference_points_img = combine.view(B, N, 1, 3, 3).matmul(reference_points).squeeze(-1)

        eps = 1e-5
        # (B N NP 1)
        volume_mask = (reference_points_img[..., 2:3] > eps)
        # (B N NP 2)
        reference_points_img = reference_points_img[..., 0:2] / torch.maximum(
            reference_points_img[..., 2:3], torch.ones_like(reference_points_img[..., 2:3]) * eps)

        # do post-transformation
        post_rots2 = post_rots[:, :, :2, :2]
        post_trans2 = post_trans[:, :, :2]
        # [(B N 2 2) -> (B N 1 2 2)] @ [(B N NP 2) -> (B N NP 2 1)] -> (B N NP 2 1)
        reference_points_img = post_rots2.view(B, N, 1, 2, 2).matmul(reference_points_img.unsqueeze(-1))
        # [(B N NP 2 1) -> (B N NP 2)] + [(B N 2) -> (B N 1 2)] -> (B N NP 2)
        reference_points_img = reference_points_img.squeeze(-1) + post_trans2.view(B, N, 1, 2)

        H_in, W_in = self.input_size
        reference_points_img[..., 0] /= W_in  # 1600 w
        reference_points_img[..., 1] /= H_in  # 928 h

        volume_mask = (volume_mask & (reference_points_img[..., 1:2] > 0.0)
                       & (reference_points_img[..., 1:2] < 1.0)
                       & (reference_points_img[..., 0:1] < 1.0)
                       & (reference_points_img[..., 0:1] > 0.0))
        # export onnx
        # if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        #     volume_mask = torch.nan_to_num(volume_mask)
        # else:
        #     volume_mask = volume_mask.new_tensor(
        #         np.nan_to_num(volume_mask.cpu().numpy()))
        volume_mask = torch.nan_to_num(volume_mask)

        # reference_points_img: torch.Size([1, 6, 163840, 2])
        # volume_mask: torch.Size([1, 6, 163840, 1])
        # (B N NP 2) -> (B NP N 2)
        reference_points_img = reference_points_img.permute(0, 2, 1, 3)
        # (B N NP 1) -> (B NP N 1)
        volume_mask = volume_mask.permute(0, 2, 1, 3)
        if mode == 'fix':
            idx_camera = torch.argmax(volume_mask.squeeze(-1).float(), dim=-1)
        elif mode == 'random':
            raise NotImplementedError
        else:
            raise NotImplementedError

        idx_batch = torch.arange(B, dtype=torch.long, device=reference_points_img.device)
        idx_point = torch.arange(num_points, dtype=torch.long, device=reference_points_img.device)
        idx_batch = idx_batch.view(B, 1).expand(B, num_points)
        idx_point = idx_point.view(1, num_points).expand(B, num_points)
        reference_points_img = reference_points_img[idx_batch, idx_point, idx_camera]
        # torch.Size([1, 163840, 2])
        volume_mask = volume_mask[idx_batch, idx_point, idx_camera]
        # torch.Size([1, 163840, 1])

        # TODO: Check whether the numerical precision will have an impact.
        coors_camera = idx_camera[..., None].float() / (N - 1)
        reference_points_img = torch.cat([reference_points_img, coors_camera], dim=-1)

        # torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        # torch.backends.cudnn.allow_tf32 = allow_tf32
        return reference_points_img, volume_mask
        # torch.Size([1, 163840, 3]), torch.Size([1, 163840, 1])
