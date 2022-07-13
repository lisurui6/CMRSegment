import math
import torch
import numpy as np
import open3d as o3d
from mayavi import mlab
from matplotlib import pyplot as plt


def o3d_volumetric_mesh(vertices):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    print("pcd.points", np.asarray(pcd.points))
    convex_mesh, __ = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)

    return pcd, convex_mesh


def lv_volumetric_tetra(num_lv, num_points):
    tetras = np.concatenate(
        [
            np.array(
                [
                    [i, i+1, i + num_points, num_points - 1],
                    [i, i+1, i + 1 + num_points, num_points - 1],
                    [i, i+1, i + num_points, 2 * num_points - 1],
                    [i, i+1, i + 1 + num_points, 2 * num_points - 1],

                    [i + num_points, i + 1 + num_points, i, num_points - 1],
                    [i + num_points, i + 1 + num_points, i + 1, num_points - 1],
                    [i + num_points, i + 1 + num_points, i, 2*num_points - 1],
                    [i + num_points, i + 1 + num_points, i+1, 2*num_points - 1],

                    [num_points - 1, 2 * num_points - 1, i, i + 1],
                    [num_points - 1, 2 * num_points - 1, i + num_points, i + 1 + num_points],
                ]
            )
            for i in range(num_points - 2)
        ], axis=0
    )
    tetras = np.concatenate(
        [
            tetras,
            np.array(
                [
                    [num_points - 2, 0, 2 * num_points - 2, num_points - 1],
                    [num_points - 2, 0, num_points, num_points - 1],
                    [num_points - 2, 0, 2 * num_points - 2, 2 * num_points - 1],
                    [num_points - 2, 0, num_points, 2 * num_points - 1],

                    [2 * num_points - 2, num_points, num_points - 2, num_points - 1],
                    [2 * num_points - 2, num_points, 0, num_points - 1],
                    [2 * num_points - 2, num_points, num_points - 2, 2 * num_points - 1],
                    [2 * num_points - 2, num_points, 0, 2 * num_points - 1],

                    [num_points - 1, 2 * num_points - 1, num_points - 2, 0],
                    [num_points - 1, 2 * num_points - 1, 2 * num_points - 2, num_points],
                ]
            )
        ], axis=0
    )
    tetras = np.concatenate([tetras + i * num_points for i in range(num_lv - 1)])
    return tetras


def lv_tetrahedron(lv_points):
    # lv_points: (B, n_lv, n_points, 3)
    print("lv_points", lv_points[0][:2, :, :])
    print()
    vertices = lv_points[0][:2, :, :].view(2 * lv_points.shape[2], 3).detach().cpu().numpy()
    pcd, mesh = o3d_volumetric_mesh(vertices)
    visualise_o3d_mesh(
        mesh=mesh,
        pcd=pcd,
        is_triangle=False,
        show_pcd_normal=False,
        show_voxel=False,
        voxel_size=0.02,
        fname="lv_myo_tetra"
    )
    print("mesh.vertices", np.array(mesh.vertices))
    print()
    print("vertices", vertices)
    print()
    tetras = np.array(mesh.tetras)  # (n_tetras, 4), values [0, 2 * n_points - 1]
    # final_tetras = [tetras]
    # for s in range(1, vertices.shape[0] - 1):
    #     # tetras = tetras + vertices.shape[2]
    #     final_tetras.append(tetras)
    # final_tetras = np.concatenate(final_tetras, axis=0)
    return torch.Tensor(tetras)[None, None, ...].to(torch.device("cuda")).repeat(lv_points.shape[0], 1, 1, 1).type(torch.int32)


def batch_sample_rv_points(
    c0x, c0y, c1z, dz1, r1, theta_c2, theta2_ratio, d_c2_c0_ratio,
    num_points_per_slice, num_slices,
    voxel_width, voxel_depth, voxel_height, batch_size
):
    """

    Args:
        c0x: (B, n_lv)
        c0y: (B, n_lv)
        c1z: (B, n_lv)
        r1: (B, n_lv)
        dz1: (B, 1)
        theta_c2: (B,)
        theta2_ratio: (B,)
        d_c2_c0_ratio: (B,)
        num_points_per_slice:
        num_slices: n_lv
        voxel_width:
        voxel_depth:
        voxel_height:
        batch_size:

    Returns:

    """

    d_max = 0.5
    d_theta_max = math.pi / (num_slices - 1) / 2

    theta2_max = torch.tensor(math.pi * 3 / 4).float().repeat(batch_size).cuda().unsqueeze(1)
    theta2_min = torch.tensor(math.pi * 1 / 6).float().repeat(batch_size).cuda().unsqueeze(1)

    theta2 = theta2_min + torch.mul(theta2_ratio.unsqueeze(1), theta2_max - theta2_min)  # (B, 1)

    z_c0 = torch.complex(real=c0x, imag=c0y)  # (B, n_rv)
    dmin = r1 * 1 / 2  # (B, n_rv)
    dmax = r1 * 3 / 2  # (B, n_rv)

    d_c2_c0 = torch.mul(d_c2_c0_ratio.unsqueeze(1), dmax - dmin) + dmin  # (B, n_rv)

    theta_c2 = theta_c2.unsqueeze(1) * math.pi * 2  # (B, 1)

    r2 = torch.sqrt(r1[:, 0] * r1[:, 0] + d_c2_c0[:, 0] * d_c2_c0[:, 0] - 2 * r1[:, 0] * d_c2_c0[:, 0] * torch.cos(theta2[:, 0]))  # (B,)
    x2 = d_c2_c0[:, 0] + r2

    x2 = torch.sqrt(1 - (c1z - c1z[:, 0].unsqueeze(1)) * (c1z - c1z[:, 0].unsqueeze(1)) / (dz1 * dz1)) * x2.unsqueeze(1)  # (B, n_rv)
    r2 = x2 - d_c2_c0  # (B, n_rv)
    theta2 = torch.arccos((r1 * r1 + d_c2_c0 * d_c2_c0 - r2 * r2) / (2 * r1 * d_c2_c0)).float()

    z_c2 = z_c0 + d_c2_c0 * torch.exp(
        torch.complex(real=torch.tensor(0).float().cuda().repeat(batch_size, num_slices), imag=theta_c2.float())
    )  # (B, n_lv)

    theta_p0 = theta_c2 - theta2  # theta_p0 = (-pi, 2pi), (B, n_rv)
    theta_p1 = theta_c2 + theta2  # theta_p1 = (0, 3pi), (B, n_rv)

    # theta_p0, theta_p1 = (0, 2pi)
    theta_p0 = torch.where(
        theta_p0 < 0,
        theta_p0 + math.pi * 2,
        theta_p0,
    )  # (B, n_rv)

    theta_p1 = torch.where(
        theta_p1 > math.pi * 2,
        theta_p1 - math.pi * 2,
        theta_p1,
    )  # (B, n_rv)

    theta_p1 = torch.where(
        theta_p1 < theta_p0,
        theta_p1 + math.pi * 2,
        theta_p1
    )  # (B, n_rv)
    n_arc_points = num_points_per_slice // 2

    theta_p0 = theta_p0.unsqueeze(2).repeat(1, 1, n_arc_points)  # (B, n_rv, n_arc_points)
    theta_p1 = theta_p1.unsqueeze(2).repeat(1, 1, n_arc_points)  # (B, n_rv, n_arc_points)

    arc_count = torch.arange(n_arc_points).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_slices, 1).cuda()  # (B, n_rv, n_arc_points)
    arc_phase = theta_p0 + torch.mul(theta_p1 - theta_p0, arc_count) / (n_arc_points - 1)  # (B, n_rv, n_arc_points)
    arc_angle = torch.exp(
        torch.complex(real=torch.tensor(0).float().repeat(batch_size, num_slices, n_arc_points).cuda(), imag=arc_phase.float())
    )  # (B, n_rv, n_arc_points)
    arc = z_c0.unsqueeze(2).repeat(1, 1, n_arc_points) + torch.mul(r1.unsqueeze(2).repeat(1, 1, n_arc_points), arc_angle)  # (B, n_lv, n_arc_points)
    arc_1 = torch.flip(arc, dims=[2])  # p1 to p0 arc

    r2 = (torch.view_as_real(z_c2) - torch.view_as_real(arc_1[..., -1])).norm(dim=2)  # (B, n_lv)
    theta_c2_p0 = torch.log(arc_1[..., -1] - z_c2).imag  # theta_c2_p0 = (-pi, pi), (B, n_lv)
    theta_c2_p1 = torch.log(arc_1[..., 0] - z_c2).imag  # theta_c2_p1 = (-pi, pi), (B, n_lv)

    theta_c2_p0 = torch.where(
        theta_c2_p0 < 0,
        theta_c2_p0 + math.pi * 2,
        theta_c2_p0,
    )

    theta_c2_p1 = torch.where(
        theta_c2_p1 < 0,
        theta_c2_p1 + math.pi * 2,
        theta_c2_p1,
    )

    theta_c2_p1 = torch.where(
        theta_c2_p0 > theta_c2_p1,
        theta_c2_p1 + math.pi * 2,
        theta_c2_p1,
    )
    theta_c2_p0 = theta_c2_p0.unsqueeze(2).repeat(1, 1, n_arc_points)  # (B, n_lv, n_arc_points)
    theta_c2_p1 = theta_c2_p1.unsqueeze(2).repeat(1, 1, n_arc_points)  # (B, n_lv, n_arc_points)

    arc_phase = theta_c2_p1 + torch.mul(theta_c2_p0 - theta_c2_p1, arc_count) / (n_arc_points - 1)  # (B, n_lv, n_arc_points)
    arc_angle = torch.exp(
        torch.complex(real=torch.tensor(0).float().repeat(batch_size, num_slices, n_arc_points).cuda(), imag=arc_phase)
    )  # (B, n_lv, n_arc_points)
    arc_2 = z_c2.unsqueeze(2).repeat(1, 1, n_arc_points) + torch.mul(r2.unsqueeze(2).repeat(1, 1, n_arc_points), arc_angle)  # (B, n_lv, n_arc_points)
    arc_1 = torch.view_as_real(arc_1)  # (B, n_lv, n_arc_points, 2)
    arc_1 = arc_1.view(arc_1.shape[0], -1, arc_1.shape[-1])  # (B, n_lv * n_arc_points, 2)
    arc_2 = torch.view_as_real(arc_2)  # (B, n_lv, n_arc_points, 2)
    arc_2 = arc_2.view(arc_2.shape[0], -1, arc_2.shape[-1])  # (B, n_lv * n_arc_points, 2)

    c0z = c1z.unsqueeze(2).repeat(1, 1, n_arc_points).unsqueeze(3)  # (B, n_lv, n_arc_points, 1)
    c0z = c0z.view(c0z.shape[0], -1, c0z.shape[-1])  # (B, n_lv * n_arc_points, 1)
    surface2_arc_in = torch.cat([arc_1, c0z], dim=2)
    surface2_arc_out = torch.cat([arc_2, c0z], dim=2)

    surface2_arc_in[..., 0] = (surface2_arc_in[..., 0] - voxel_width / 2) / voxel_width * 2
    surface2_arc_in[..., 1] = (surface2_arc_in[..., 1] - voxel_depth / 2) / voxel_depth * 2
    surface2_arc_in[..., 2] = (surface2_arc_in[..., 2] - voxel_height / 2) / voxel_height * 2

    surface2_arc_out[..., 0] = (surface2_arc_out[..., 0] - voxel_width / 2) / voxel_width * 2
    surface2_arc_out[..., 1] = (surface2_arc_out[..., 1] - voxel_depth / 2) / voxel_depth * 2
    surface2_arc_out[..., 2] = (surface2_arc_out[..., 2] - voxel_height / 2) / voxel_height * 2
    rv_points = torch.cat([surface2_arc_in, surface2_arc_out], dim=1)

    return rv_points


def batch_sample_lv_myo_points(
    c0x, c0y, c0z, c0z_end_ratio, c1z_end_ratio, r1, r0_r1_ratio, num_points_per_slice, num_lv_slices,
    voxel_width, voxel_depth, voxel_height,
):
    """

    Args:
        c0x: (B,)
        c0y: (B,)
        c0z: (B,)
        c0z_end_ratio: (B,)
        c1z_end_ratio: (B,)
        r1: (B,)
        r0_r1_ratio: (B,)
        num_points_per_slice:
        num_lv_slices:
        voxel_width:
        voxel_height:
        voxel_depth:
        batch_size:

    Returns:

    """
    d_max = 0.5
    batch_size = c0x.shape[0]
    c0x = (c0x.unsqueeze(1) / 2 + 0.5) * voxel_width  # (B, 1)
    c0y = (c0y.unsqueeze(1) / 2 + 0.5) * voxel_depth  # (B, 1)
    c0z_0 = (c0z.unsqueeze(1) / 2 + 0.5) * voxel_height  # (B, 1)
    c1z_end = c0z_0 + (voxel_height - c0z_0) * c1z_end_ratio.unsqueeze(1)
    # c1z_end = (c1z_end / 2 + 0.5) * voxel_height  # (B, 1)

    dz = (c1z_end - c0z_0) / (num_lv_slices)  # (B, 1)
    dz = dz.repeat(1, num_lv_slices - 1)  # (B, n_lv - 1)
    c1z = c0z_0 + torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dz, dim=1)], dim=1)  # (B, n_lv)

    c0z_end = c0z_end_ratio.unsqueeze(1) * (c1z_end - c0z_0) * 0.9 + c0z_0
    dz = (c0z_end - c0z_0) / (num_lv_slices)  # (B, 1)
    dz = dz.repeat(1, num_lv_slices - 1)  # (B, n_lv - 1)
    c0z = c0z_0 + torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dz, dim=1)], dim=1)  # (B, n_lv)

    c0x = c0x.repeat(1, num_lv_slices)
    c0y = c0y.repeat(1, num_lv_slices)

    r1 = r1.unsqueeze(1) * voxel_width  # (B, 1)
    r0 = (r0_r1_ratio.unsqueeze(1) * 0.4 + 0.5) * r1
    # r0 = r0_r1_ratio.unsqueeze(1) * r1  # (B, 1)
    # dr0_max = r0 / (num_lv_slices - 1)  # (B, 1)
    # dr0 = torch.mul(dr0, dr0_max)  # (B, n_lv - 1)
    # r0 = r0 - torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dr0, dim=1)], dim=1)  # (B, n_lv)

    r1 = torch.sqrt(1 - (c1z - c0z_0) * (c1z - c0z_0) / ((c1z_end - c0z_0) * (c1z_end - c0z_0))) * r1
    r0 = torch.sqrt(1 - (c0z - c0z_0) * (c0z - c0z_0) / ((c0z_end - c0z_0) * (c0z_end - c0z_0))) * r0

    c0_phase = torch.arange(num_points_per_slice).repeat(batch_size, num_lv_slices, 1).cuda()
    c0_phase = 2 * math.pi * c0_phase / num_points_per_slice  # (B, n_lv, n_points)
    c0_angle = torch.exp(
        torch.complex(
            real=torch.tensor(0).float().repeat(batch_size, num_lv_slices, num_points_per_slice).cuda(),
            imag=c0_phase,
        )
    )  # (B, n_lv, n_points)
    z_c0 = torch.complex(real=c0x, imag=c0y).unsqueeze(2)  # (B, n_lv, 1)
    lv_xy = z_c0.repeat(1, 1, num_points_per_slice) + torch.mul(r0.unsqueeze(2).repeat(1, 1, num_points_per_slice), c0_angle)  # (B, n_lv, n_points)
    lv_xy = torch.cat([lv_xy, z_c0], dim=2)
    lv_xy = torch.view_as_real(lv_xy)  # (B, n_lv, n_points, 2)

    lv_points = torch.cat([lv_xy, c0z.unsqueeze(2).repeat(1, 1, num_points_per_slice + 1).unsqueeze(3)], dim=3)
    lv_points = lv_points.view(lv_points.shape[0], -1, lv_points.shape[3])

    # myo
    z_c1 = torch.complex(real=c0x, imag=c0y).unsqueeze(2)  # (B, n_lv + n_myo, 1)
    c1_phase = torch.arange(num_points_per_slice).repeat(batch_size, num_lv_slices, 1).cuda()
    c1_phase = 2 * math.pi * c1_phase / num_points_per_slice  # (B, n_lv + n_myo, n_points)
    c1_angle = torch.exp(
        torch.complex(
            real=torch.tensor(0).float().repeat(batch_size, num_lv_slices,
                                                num_points_per_slice).cuda(),
            imag=c1_phase,
        )
    )  # (B, n_lv + n_myo, n_points)
    myo_xy = z_c1.repeat(1, 1, num_points_per_slice) + torch.mul(r1.unsqueeze(2).repeat(1, 1, num_points_per_slice), c1_angle)  # (B, n_lv + n_myo, n_points)
    myo_xy = torch.cat([myo_xy, z_c1], dim=2)
    myo_xy = torch.view_as_real(myo_xy)  # (B, n_lv, n_points, 2)
    lv_myo_points = torch.cat([myo_xy, c1z.unsqueeze(2).repeat(1, 1, num_points_per_slice + 1).unsqueeze(3)], dim=3)

    lv_myo_points = lv_myo_points.view(lv_myo_points.shape[0], -1, lv_myo_points.shape[3])

    plot = False
    if plot:
        lv_points = torch.cat([lv_points, lv_myo_points], dim=1)
        lv_pcd, lv_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_points[0].detach().cpu().numpy(),
        )
        o3d.visualization.draw_geometries([lv_pcd], point_show_normal=True)
        points = np.asarray(lv_tetra_mesh.vertices)
        o3d.visualization.draw_geometries([lv_pcd, lv_tetra_mesh], mesh_show_back_face=False)
        lv_myo_pcd, lv_myo_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_myo_points[0].detach().cpu().numpy(),
        )
        o3d.visualization.draw_geometries([lv_myo_pcd], point_show_normal=True)
        points = np.asarray(lv_myo_tetra_mesh.vertices)
        o3d.visualization.draw_geometries([lv_myo_pcd, lv_myo_tetra_mesh], mesh_show_back_face=False)

    lv_points[..., 0] = (lv_points[..., 0] - voxel_width / 2) / voxel_width * 2
    lv_points[..., 1] = (lv_points[..., 1] - voxel_depth / 2) / voxel_depth * 2
    lv_points[..., 2] = (lv_points[..., 2] - voxel_height / 2) / voxel_height * 2

    lv_myo_points[..., 0] = (lv_myo_points[..., 0] - voxel_width / 2) / voxel_width * 2
    lv_myo_points[..., 1] = (lv_myo_points[..., 1] - voxel_depth / 2) / voxel_depth * 2
    lv_myo_points[..., 2] = (lv_myo_points[..., 2] - voxel_height / 2) / voxel_height * 2
    return lv_points, lv_myo_points, c0x, c0y, c1z, r1, c1z_end - c0z_0


# batch_sample_lv_myo_points(
#     c0x=torch.from_numpy(np.array([0])).cuda(),
#     c0y=torch.from_numpy(np.array([0])).cuda(),
#     c0z=torch.from_numpy(np.array([0])).cuda(),
#     c0z_end_ratio=torch.from_numpy(np.array([0.9])).cuda(),
#     c1z_end=torch.from_numpy(np.array([0.5])).cuda(),
#     r1=torch.from_numpy(np.array([0.05])).cuda(),
#     r0_r1_ratio=torch.from_numpy(np.array([0.9])).cuda(),
#     num_points_per_slice=128,
#     num_lv_slices=128,
#     voxel_width=64,
#     voxel_depth=64,
#     voxel_height=32,
# )


def sample_lv_rv_points(
    lv_par1, lv_par2, rv_par1, num_lv_slices, voxel_width, voxel_depth, voxel_height, num_points, batch_size, epoch,
    lv_tetras=None, lv_myo_tetras=None, rv_tetras=None
):
    """
    lv_par1: (-1, 1)
        c0_x, c0_y, c0_z
    lv_par2: (0, 1)
        r1, r0/r1: 1, c1_z_end, c0z_end_ratio: 1
    rv_par1: (0, 1)
        theta_c2, theta2_ratio, d_c2_c0_ratio
    """

    d_slices = num_lv_slices - 1
    lv_points, lv_myo_points, c0x, c0y, c1z, r1, dz1 = batch_sample_lv_myo_points(
        c0x=lv_par1[:, 0],
        c0y=lv_par1[:, 1],
        c0z=lv_par1[:, 2],
        c0z_end_ratio=lv_par2[:, 3],
        c1z_end_ratio=lv_par2[:, 2],
        r1=lv_par2[:, 0],
        r0_r1_ratio=lv_par2[:, 1],
        voxel_width=voxel_width,
        voxel_depth=voxel_depth,
        voxel_height=voxel_height,
        num_points_per_slice=num_points,
        num_lv_slices=num_lv_slices,
    )

    rv_points = batch_sample_rv_points(
        c0x=c0x,
        c0y=c0y,
        c1z=c1z,
        r1=r1,
        dz1=dz1,
        theta_c2=rv_par1[:, 0],
        theta2_ratio=rv_par1[:, 1],
        d_c2_c0_ratio=rv_par1[:, 2],
        voxel_width=voxel_width,
        voxel_depth=voxel_depth,
        voxel_height=voxel_height,
        num_points_per_slice=num_points,
        num_slices=num_lv_slices,
        batch_size=batch_size,
    )
    if rv_tetras is None:
        pcd = o3d.geometry.PointCloud()
        points = torch.cat([lv_points, lv_myo_points, rv_points], dim=1)
        pcd.points = o3d.utility.Vector3dVector(points[0].detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(
            np.concatenate(
                [
                    np.tile(np.array([[255, 0, 0]]), (lv_points.shape[1], 1)),
                    np.tile(np.array([[0, 255, 0]]), (lv_myo_points.shape[1], 1)),
                    np.tile(np.array([[0, 0, 255]]), (rv_points.shape[1], 1)),
                ],
                axis=0,
            )
        )
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    if lv_tetras is None:
        # lv_tetras = lv_tetrahedron(lv_points.view(lv_points.shape[0], num_lv_slices, num_points + 1, 3))
        lv_tetras = lv_volumetric_tetra(num_lv_slices, num_points + 1)

        lv_pcd, lv_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_points[0].detach().cpu().numpy(),
        )
        lv_tetra_mesh.tetras = o3d.utility.Vector4iVector(lv_tetras.astype(np.int32))
        lv_tetra_mesh.vertices = o3d.utility.Vector3dVector(lv_points[0].detach().cpu().numpy())
        lv_pcd.points = o3d.utility.Vector3dVector(lv_points[0].detach().cpu().numpy())

        visualise_o3d_mesh(
            mesh=lv_tetra_mesh,
            pcd=lv_pcd,
            is_triangle=False,
            show_pcd_normal=False,
            show_voxel=False,
            voxel_size=0.02,
            fname="lv_tetra"
        )

    if lv_myo_tetras is None:
        lv_myo_tetras = lv_volumetric_tetra(num_lv_slices, num_points + 1)
        # lv_myo_tetras = lv_tetrahedron(lv_myo_points.view(lv_myo_points.shape[0], num_lv_slices + num_extra_lv_myo_slices, num_points + 1, 3))

        # lv_myo_pcd, lv_myo_tetra_mesh = o3d_volumetric_mesh(
        #     vertices=lv_myo_points[0].detach().cpu().numpy(),
        # )
        # lv_myo_tetra_mesh.tetras = o3d.utility.Vector4iVector(lv_myo_tetras.astype(np.int32))
        # lv_myo_tetra_mesh.vertices = o3d.utility.Vector3dVector(lv_myo_points[0].detach().cpu().numpy())
        # lv_myo_pcd.points = o3d.utility.Vector3dVector(lv_myo_points[0].detach().cpu().numpy())
        #
        # visualise_o3d_mesh(
        #     mesh=lv_myo_tetra_mesh,
        #     pcd=lv_myo_pcd,
        #     is_triangle=False,
        #     show_pcd_normal=False,
        #     show_voxel=False,
        #     voxel_size=0.02,
        #     fname="lv_myo_tetra"
        # )

    batch_lv_tetras = torch.Tensor(lv_tetras)[None, None, ...].to(torch.device("cuda")).repeat(batch_size, 1, 1, 1).type(torch.int32)
    batch_lv_myo_tetras = torch.Tensor(lv_myo_tetras)[None, None, ...].to(torch.device("cuda")).repeat(batch_size, 1, 1, 1).type(torch.int32)

    if rv_tetras is None:
        rv_tetras = rv_volumentric_tetra(rv_points.shape[1], num_points)
    batch_rv_tetras = torch.Tensor(rv_tetras)[None, None, ...].to(torch.device("cuda")).repeat(batch_size, 1, 1, 1).type(torch.int32)

    return [lv_points, batch_lv_tetras, lv_tetras], \
           [lv_myo_points, batch_lv_myo_tetras, lv_myo_tetras], \
           [rv_points, batch_rv_tetras, rv_tetras]


def visualise_o3d_mesh(
        mesh, pcd, is_triangle: bool = False, show_voxel: bool = False, voxel_size=0.01, show_pcd_normal: bool = True,
        use_mayavi: bool = False, fname: str = "mesh",
):
    o3d.visualization.draw_geometries([pcd], point_show_normal=show_pcd_normal)
    points = np.asarray(mesh.vertices)
    o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=False)

    if is_triangle:
        if use_mayavi:
            mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2], np.asarray(mesh.triangles))
            mlab.savefig(fname + ".png")
            mlab.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot_trisurf(
                points[:, 0], points[:, 1], -points[:, 2], triangles=np.asarray(mesh.triangles), cmap=plt.cm.Spectral
            )
            print("saving {}.png".format(fname))
            plt.savefig(fname + ".png")
    if show_voxel:
        if is_triangle:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
        else:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
        o3d.visualization.draw_geometries([voxel_grid])


def rv_volumentric_tetra(num_total_points, num_points_per_layer):
    """RV points are concated as [arc_in, arc_out]"""
    num_layers = int(num_total_points / num_points_per_layer)
    arc_out_offset = int(num_total_points / 2)
    num_points_per_layer = int(num_points_per_layer / 2)
    tetras = np.concatenate(
        [
            np.array([
                [i + arc_out_offset, i + 1 + arc_out_offset, i, i + num_points_per_layer],
                [i + arc_out_offset, i + 1 + arc_out_offset, i, i + num_points_per_layer + 1],
                [i + arc_out_offset, i + 1 + arc_out_offset, i, i + num_points_per_layer + arc_out_offset],
                [i + arc_out_offset, i + 1 + arc_out_offset, i, i + num_points_per_layer + arc_out_offset + 1],
                [i + arc_out_offset + 1, i, i + 1, i + num_points_per_layer],
                [i + arc_out_offset + 1, i, i + 1, i + num_points_per_layer + 1],
                [i + arc_out_offset + 1, i, i + 1, i + num_points_per_layer + arc_out_offset],
                [i + arc_out_offset + 1, i, i + 1, i + num_points_per_layer + arc_out_offset + 1],
                [i + arc_out_offset + num_points_per_layer, i + 1 + arc_out_offset + num_points_per_layer, i + num_points_per_layer, i],
                [i + arc_out_offset + num_points_per_layer, i + 1 + arc_out_offset + num_points_per_layer, i + num_points_per_layer, i + 1],
                [i + arc_out_offset + num_points_per_layer, i + 1 + arc_out_offset + num_points_per_layer, i + num_points_per_layer, i + arc_out_offset],
                [i + arc_out_offset + num_points_per_layer, i + 1 + arc_out_offset + num_points_per_layer, i + num_points_per_layer, i + arc_out_offset + 1],
                [i + arc_out_offset + 1 + num_points_per_layer, i + num_points_per_layer, i + 1 + num_points_per_layer, i],
                [i + arc_out_offset + 1 + num_points_per_layer, i + num_points_per_layer, i + 1 + num_points_per_layer, i + 1],
                [i + arc_out_offset + 1 + num_points_per_layer, i + num_points_per_layer, i + 1 + num_points_per_layer, i + arc_out_offset],
                [i + arc_out_offset + 1 + num_points_per_layer, i + num_points_per_layer, i + 1 + num_points_per_layer, i + 1 + arc_out_offset],
            ])
            for i in range(num_points_per_layer - 1)
        ], axis=0
    )
    tetras = np.concatenate([tetras + i * num_points_per_layer for i in range(num_layers - 1)])
    return tetras
