from typing import Callable, Optional, Sequence, Tuple

import torch
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.utils import masked_gather
from pytorch3d.transforms import euler_angles_to_matrix
from torch import nn
import numpy as np

Transform = Callable[[torch.Tensor], torch.Tensor]


def resample_points(points: torch.Tensor, num_points: int) -> torch.Tensor:
    if points.shape[1] > num_points:
        if num_points == 1024:
            num_samples = 1200
        elif num_points == 2048:
            num_samples = 2400
        elif num_points == 4096:
            num_samples = 4800
        elif num_points == 8192:
            num_samples = 8192
        else:
            raise NotImplementedError()
        if points.shape[1] < num_samples:
            num_samples = points.shape[1]
        _, idx = sample_farthest_points(
            points[:, :, :3].float(), K=num_samples, random_start_point=True
        )
        points = masked_gather(points, idx)
        points = points[:, torch.randperm(num_samples)[:num_points]]
        return points
    else:
        raise RuntimeError("Not enough points")


class PointcloudSubsampling(nn.Module):
    def __init__(self, num_points: int, strategy: str = "fps"):
        super().__init__()
        self.num_points = num_points
        self.strategy = strategy

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        if points.shape[1] < self.num_points:
            raise RuntimeError(
                f"Too few points in pointcloud: {points.shape[1]} vs {self.num_points}"
            )
        elif points.shape[1] == self.num_points:
            return points

        if self.strategy == "resample":
            return resample_points(points, self.num_points)
        elif self.strategy == "fps":
            _, idx = sample_farthest_points(
                points[:, :, :3].float(), K=self.num_points, random_start_point=True
            )
            return masked_gather(points, idx)
        elif self.strategy == "random":
            return points[:, torch.randperm(points.shape[1])[: self.num_points]]
        else:
            raise RuntimeError(f"No such subsampling strategy {self.strategy}")


# TODO: remove this
class PointcloudCenterAndNormalize(nn.Module):
    def __init__(
        self,
        centering: bool = True,
        normalize=True,
    ):
        super().__init__()
        self.centering = centering
        self.normalize = normalize

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        if self.centering:
            points[:, :, :3] = points[:, :, :3] - torch.mean(
                points[:, :, :3], dim=-2, keepdim=True
            )
        if self.normalize:
            max_norm = torch.max(
                torch.norm(points[:, :, :3], dim=-1, keepdim=True),
                dim=-2,
                keepdim=True,
            ).values
            points[:, :, :3] = points[:, :, :3] / max_norm
        return points


class PointcloudCentering(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        points[:, :, :3] = points[:, :, :3] - torch.mean(
            points[:, :, :3], dim=-2, keepdim=True
        )
        return points


class PointcloudUnitSphere(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        max_norm = torch.max(
            torch.norm(points[:, :, :3], dim=-1, keepdim=True),
            dim=-2,
            keepdim=True,
        ).values
        points[:, :, :3] = points[:, :, :3] / max_norm
        return points


class PointcloudHeightNormalization(nn.Module):
    def __init__(
        self,
        dim: int,
        append: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.append = append

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        min_height = torch.min(points[:, :, self.dim], dim=-1).values
        heights = points[:, :, self.dim] - min_height.unsqueeze(-1)
        if self.append:
            points = torch.cat([points, heights.unsqueeze(-1)], dim=-1)
        else:
            points[:, :, self.dim] = heights
        return points


class PointcloudScaling(nn.Module):
    def __init__(
        self,
        min: float,
        max: float,
        anisotropic: bool = True,
        scale_xyz: Tuple[bool, bool, bool] = (True, True, True),
        symmetries: Tuple[int, int, int] = (0, 0, 0),  # mirror scaling, x --> -x
    ):
        super().__init__()
        self.scale_min = min
        self.scale_max = max
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.symmetries = torch.tensor(symmetries)

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        scale = (
            torch.rand(3 if self.anisotropic else 1, device=points.device)
            * (self.scale_max - self.scale_min)
            + self.scale_min
        )

        symmetries = torch.round(torch.rand(3, device=points.device)) * 2 - 1
        self.symmetries = self.symmetries.to(points.device)
        symmetries = symmetries * self.symmetries + (1 - self.symmetries)
        scale *= symmetries
        for i, s in enumerate(self.scale_xyz):
            if not s:
                scale[i] = 1
        points[:, :, :3] = points[:, :, :3] * scale
        return points


class PointcloudTranslation(nn.Module):
    def __init__(
        self,
        translation: float,
    ):
        super().__init__()
        self.translation = translation

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        translation = (
            torch.rand(3, device=points.device) * 2 * self.translation
            - self.translation
        )

        points[:, :, :3] = points[:, :, :3] + translation
        return points


class PointcloudRotation(nn.Module):
    def __init__(self, dims: Sequence[int], deg: Optional[int] = None):
        # deg: \in [0...179], eg 45 means rotation steps of 45 deg are allowed
        super().__init__()
        self.dims = dims
        self.deg = deg
        assert self.deg is None or (self.deg >= 0 and self.deg <= 180)

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        euler_angles = torch.zeros(3)
        for dim in self.dims:
            if self.deg is not None:
                possible_degs = (
                    torch.tensor(list(range(0, 360, self.deg))) / 360
                ) * 2 * torch.pi - torch.pi
                euler_angles[dim] = possible_degs[
                    torch.randint(high=len(possible_degs), size=(1,))
                ]
            else:
                euler_angles[dim] = (2 * torch.pi) * torch.rand(1) - torch.pi
        R = euler_angles_to_matrix(euler_angles, "XYZ").to(points.device)
        points[:, :, :3] = points[:, :, :3] @ R.T
        return points


class Compose:
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms

    def __call__(self, points: torch.Tensor):
        for t in self.transforms:
            points = t(points)
        return points


# augument method
def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


class PointcloudRotatePerturbation(nn.Module):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18, p=1):
        super().__init__()
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip
        self.p = p

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def forward(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(2) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, :, 0:3]
            pc_normals = points[:, :, 3:]
            points[:, :, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t().to(pc_xyz))
            points[:, :, 3:] = torch.matmul(pc_normals, rotation_matrix.t().to(pc_normals))

            return points


class PointcloudRandomCrop(nn.Module):
    def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1, min_num_points=4096, max_try_num=10):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.p = p

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points

    def forward(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.cpu().numpy()

        B, N, C = points.shape
        points = points.reshape([-1, C])

        isvalid = False
        try_num = 0
        while not isvalid:
            coord_min = np.min(points[:, :3], axis=0)
            coord_max = np.max(points[:, :3], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            new_coord_range = np.zeros(3)
            new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            new_coord_range[1] = new_coord_range[0] * ar
            new_coord_range[2] = new_coord_range[0] / ar
            # new_coord_range = np.where(new_coord_range>1, 1, new_coord_range)

            new_coord_min = np.random.uniform(0, 1 - new_coord_range)
            new_coord_max = new_coord_min + new_coord_range

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            new_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            new_indices = np.sum(new_indices, axis=1) == 3
            new_points = points[new_indices]

            # other_num = points.shape[0] - new_points.shape[0]
            # if new_points.shape[0] > 0:
            #     isvalid = True
            if new_points.shape[0] >= self.min_num_points and new_points.shape[0] < points.shape[0]:
                isvalid = True

            try_num += 1
            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()

        # other_indices = np.random.choice(np.arange(new_points.shape[0]), other_num)
        # other_points = new_points[other_indices]
        # new_points = np.concatenate([new_points, other_points], axis=0)

        # new_points[:,:3] = (new_points[:,:3] - new_coord_min) / (new_coord_max - new_coord_min) * coord_diff + coord_min
        A, C = new_points.shape
        M = A / B
        BB = int(M) * B
        new_points = new_points[:BB, :]
        new_points = new_points.reshape(B, int(M), C)
        return torch.from_numpy(new_points).float()


class PointcloudRandomCutout(nn.Module):
    def __init__(self, ratio_min=0.3, ratio_max=0.6, p=1, min_num_points=4096, max_try_num=10):
        super().__init__()
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.p = p
        self.min_num_points = min_num_points
        self.max_try_num = max_try_num

    def forward(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.cpu().numpy()

        B, N, C = points.shape
        points = points.reshape([-1, C])

        try_num = 0
        valid = False
        while not valid:
            coord_min = np.min(points[:,:3], axis=0)
            coord_max = np.max(points[:,:3], axis=0)
            coord_diff = coord_max - coord_min

            cut_ratio = np.random.uniform(self.ratio_min, self.ratio_max, 3)
            new_coord_min = np.random.uniform(0, 1-cut_ratio)
            new_coord_max= new_coord_min + cut_ratio

            new_coord_min = coord_min + new_coord_min * coord_diff
            new_coord_max = coord_min + new_coord_max * coord_diff

            cut_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            cut_indices = np.sum(cut_indices, axis=1) == 3

            # print(np.sum(cut_indices))
            # other_indices = (points[:, :3] < new_coord_min) | (points[:, :3] > new_coord_max)
            # other_indices = np.sum(other_indices, axis=1) == 3
            try_num += 1

            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()

            # cut the points, sampling later

            if points.shape[0] - np.sum(cut_indices) >= self.min_num_points and np.sum(cut_indices) > 0:
                # print (np.sum(cut_indices))
                points = points[cut_indices==False]
                valid = True
        # if np.sum(other_indices) > 0:
        #     comp_indices = np.random.choice(np.arange(np.sum(other_indices)), np.sum(cut_indices))
        #     points[cut_indices] = points[comp_indices]
        A, C = points.shape
        M = A / B
        BB = int(M) * B
        points = points[:BB, :]
        points = points.reshape(B, int(M), C)
        return torch.from_numpy(points).float().to("cuda")