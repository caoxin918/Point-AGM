# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import torch

import chamfer


# class ChamferFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, xyz1, xyz2):
#         dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
#         ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
#
#         return dist1, dist2
#
#     @staticmethod
#     def backward(ctx, grad_dist1, grad_dist2):
#         xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
#         grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
#         return grad_xyz1, grad_xyz2

class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2

# class ChamferDistanceL2(torch.nn.Module):
#     f''' Chamder Distance L2
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros
#
#     def forward(self, xyz1, xyz2):
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
#
#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         return torch.mean(dist1) + torch.mean(dist2)

class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)

# class ChamferDistanceL2_split(torch.nn.Module):
#     f''' Chamder Distance L2
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros
#
#     def forward(self, xyz1, xyz2):
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
#
#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, _, _  = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

# class ChamferDistanceL1(torch.nn.Module):
#     f''' Chamder Distance L1
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros
#
#     def forward(self, xyz1, xyz2):
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
#
#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         # import pdb
#         # pdb.set_trace()
#         dist1 = torch.sqrt(dist1)
#         dist2 = torch.sqrt(dist2)
#         return (torch.mean(dist1) + torch.mean(dist2))/2

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, _, _  = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2))/2

def dis_normalized_l2(normal1, normal2):
    # B * Gs * 3
    normal1 = torch.nn.functional.normalize(normal1, dim=2)
    normal2 = torch.nn.functional.normalize(normal2, dim=2)
    return torch.min((normal1-normal2).pow(2).sum(2), (normal1+normal2).pow(2).sum(2))
    # Note the above equation is closely related to the below absolute normal distance; specifically, (normal1-normal2)^2 = normal1^2 - normal2^2 - 2*normal1*normal2 = 2 (1 - normal1*normal2)
    # We adopt the above form since it is tested in the project construction stage. Similar results should be achieved with the below equation.
    # return 1 - torch.clamp(torch.abs((normal1 * normal2).sum(2)), min=0, max=1)


def dis_normalized_l2_strict(normal1, normal2):
    # B * Gs * 3
    normal1 = torch.nn.functional.normalize(normal1, dim=2)
    normal2 = torch.nn.functional.normalize(normal2, dim=2)
    return (normal1-normal2).pow(2).sum(2)


class ChamferDistanceL2_withnormal(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2, normal_rebuild, normal_gt, curve_rebuild=torch.rand(0), curve_gt=torch.rand(0)):
        # xyz1: B * Gs * 6
        batch_size = xyz1.size(0)
        # if batch_size == 1 and self.ignore_zeros:
        #     non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
        #     non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
        #     xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
        #     xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, idx1, idx2 = ChamferFunction.apply(xyz1, xyz2)

        # target_xyz1 = torch.gather(xyz2[:, :, :3], 1, idx1.long().unsqueeze(2).expand(xyz1[:, :, :3].size()))
        # target_xyz2 = torch.gather(xyz1[:, :, :3], 1, idx2.long().unsqueeze(2).expand(xyz2[:, :, :3].size()))
        # our_dist1 = dis_l2(xyz1[:, :, :3], target_xyz1) ## this is identical to dist1
        # our_dist12 = dis_l2(xyz2[:, :, :3], target_xyz2) ## this is identical to dist2

        target_xyz1 = torch.gather(normal_gt, 1, idx1.long().unsqueeze(2).expand(normal_rebuild.size()))
        target_xyz2 = torch.gather(normal_rebuild, 1, idx2.long().unsqueeze(2).expand(normal_gt.size()))
        normal_dist1 = dis_normalized_l2(normal_rebuild, target_xyz1)
        normal_dist2 = dis_normalized_l2(normal_gt, target_xyz2)

        # The curve estimation is not utilized.
        if curve_rebuild.size()[0] != 0:
            target_xyz1_curve = torch.gather(curve_gt, 1, idx1.long().unsqueeze(2).expand(curve_rebuild.size()))
            target_xyz2_curve = torch.gather(curve_rebuild, 1, idx2.long().unsqueeze(2).expand(curve_gt.size()))
            curve_dist1 = dis_normalized_l2_strict(curve_rebuild, target_xyz1_curve)
            curve_dist2 = dis_normalized_l2_strict(curve_gt, target_xyz2_curve)
            return torch.mean(dist1) + torch.mean(dist2), torch.mean(normal_dist1) + torch.mean(normal_dist2) + torch.mean(curve_dist1) + torch.mean(curve_dist2)
        else:
            return torch.mean(dist1) + torch.mean(dist2),  torch.mean(normal_dist1) + torch.mean(normal_dist2)

