import math
import torch
import numpy as np
import open3d as o3d
import ccitk
from pathlib import Path
import nibabel as nib


def voxelize_mask(voxeliser, nodes, faces):
    P3d = torch.squeeze(nodes, dim=1)
    faces = torch.squeeze(faces, dim=1).to(nodes.device)
    mask = voxeliser(P3d, faces).unsqueeze(1)
    return mask


def visualise_o3d_mesh(
        mesh, pcd, is_triangle: bool = False, show_voxel: bool = False, voxel_size=0.01, show_pcd_normal: bool = True,
        use_mayavi: bool = False, fname: str = "mesh",
):
    o3d.visualization.draw_geometries([pcd], point_show_normal=show_pcd_normal)
    points = np.asarray(mesh.vertices)
    o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=False)


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


def o3d_volumetric_mesh(vertices):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    print("pcd.points", np.asarray(pcd.points))
    convex_mesh, __ = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)

    return pcd, convex_mesh


def torch_basis_function_array(u: torch.Tensor, i, p, knots: torch.Tensor, values=None, cuda: bool = False):
    if values is None:
        values = {}
    if p == 0:
        v_ip = torch.where(
            (u <= knots[i + 1]) & (knots[i] <= u),
            1.,
            0.,
        )
        values[(i, p)] = v_ip
        return v_ip
    if (i, p) in values:
        return values[(i, p)]
    b_i_p_1 = torch_basis_function_array(u, i, p - 1, knots, values, cuda)
    if (knots[i + p] - knots[i]) == 0.:
        part1 = torch.zeros(b_i_p_1.shape).float()

    else:
        part1 = torch.mul((u - knots[i]) / (knots[i + p] - knots[i]), b_i_p_1)
    b_i_p_1 = torch_basis_function_array(u, i + 1, p - 1, knots, values, cuda)
    if knots[i + p + 1] - knots[i + 1] == 0.:
        part2 = torch.zeros(b_i_p_1.shape).float()
    else:
        part2 = torch.mul((knots[i + p + 1] - u) / (knots[i + p + 1] - knots[i + 1]), b_i_p_1)
    if cuda:
        part1 = part1.cuda()
        part2 = part2.cuda()
    values[(i, p)] = part1 + part2
    return part1 + part2


def torch_b_spline_surface_knots(points: torch.Tensor, p: int, q: int, cuda: bool = False):
    """
    points: (n, m, 3)
    """
    diff = points[1:] - points[:-1]  # (n - 1, m, 3)
    diff = torch.sqrt(torch.sum(diff * diff, dim=-1))  # (n - 1, m)
    d = torch.sum(diff, dim=0)  # (m,)
    if cuda:
        diff = torch.cat([torch.zeros((1, d.shape[0])).float().cuda(), diff / d], dim=0)  # (n, m)
    else:
        diff = torch.cat([torch.zeros((1, d.shape[0])).float(), diff / d], dim=0)  # (n, m)

    U_hat = torch.cumsum(diff, dim=0)  # (n, m)
    u_hat = torch.mean(U_hat, dim=1)  # (n,)

    diff = points[:, 1:] - points[:, :-1]  # (n, m - 1, 3)
    diff = torch.sqrt(torch.sum(diff * diff, dim=-1))  # (n, m - 1)
    d = torch.sum(diff, dim=1).unsqueeze(1)  # (n,)
    if cuda:
        diff = torch.cat([torch.zeros((d.shape[0], 1)).float().cuda(), diff / d], dim=1)  # (n, m)
    else:
        diff = torch.cat([torch.zeros((d.shape[0], 1)).float(), diff / d], dim=1)  # (n, m)
    V_hat = torch.cumsum(diff, dim=1)  # (n, m)
    v_hat = torch.mean(V_hat, dim=0)  # (m,)

    pre_knots = torch.zeros(p + 1).float()
    post_knots = torch.ones(p + 1).float()
    if cuda:
        pre_knots = pre_knots.cuda()
        post_knots = post_knots.cuda()
    knots = torch.nn.AvgPool1d(kernel_size=p, stride=1)(u_hat[1:-1].float().unsqueeze(0).unsqueeze(0)).squeeze(
        0).squeeze(0)
    u_knots = torch.cat([pre_knots, knots, post_knots], dim=0)  # (n + p + 1)

    pre_knots = torch.zeros(q + 1).float()
    post_knots = torch.ones(q + 1).float()
    if cuda:
        pre_knots = pre_knots.cuda()
        post_knots = post_knots.cuda()
    knots = torch.nn.AvgPool1d(kernel_size=q, stride=1)(v_hat[1:-1].float().unsqueeze(0).unsqueeze(0)).squeeze(
        0).squeeze(0)
    v_knots = torch.cat([pre_knots, knots, post_knots], dim=0)  # (m + q + 1)
    return u_hat, v_hat, u_knots, v_knots


def tourch_b_spline_curve_control_points_given_points(points: torch.Tensor, u_hat: torch.Tensor, u_knots: torch.Tensor, p: int, cuda):
    n = u_hat.shape[0]
    values = {}
    basis_matrix = torch.stack([torch_basis_function_array(u_hat, i, p, u_knots, values, cuda) for i in range(n)], dim=1)
    basis_matrix[-1, -1] = 1.
    try:
        inv = torch.linalg.inv(basis_matrix)
    except Exception as e:
        print(points)
        print(u_hat)
        print(u_knots)
        print(basis_matrix.numpy())
        raise Exception
    cp = torch.matmul(inv, points)  # (n, 3)
    return cp


def torch_b_spline_surface_control_points_given_points(
        points: torch.Tensor,
        u_hat: torch.Tensor, v_hat: torch.Tensor,
        u_knots: torch.Tensor, v_knots: torch.Tensor,
        p: int, q: int, cuda,
):
    """
    points: (n, m, 3)
    u_hat: (n,)
    v_hat: (m,)
    """
    n, m, __ = points.shape
    R = []
    for l in range(m):
        r = tourch_b_spline_curve_control_points_given_points(
            points=points[:, l, :],  # (n, 3)
            u_hat=u_hat,
            u_knots=u_knots,
            p=p,
            cuda=cuda,
        )  # (n, 3)
        R.append(r)
    R = torch.stack(R, dim=1)  # (n, m, 3)
    CP = []
    for k in range(n):
        cp = tourch_b_spline_curve_control_points_given_points(
            points=R[k, :, :],  # (m, 3)
            u_hat=v_hat,
            u_knots=v_knots,
            p=q,
            cuda=cuda,
        )  # (m, 3)
        CP.append(cp)
    CP = torch.stack(CP, dim=0)  # (n, m, 3)
    return CP


def torch_b_spline_surface(
        control_points: torch.Tensor, u_knots: torch.Tensor, v_knots: torch.Tensor, p: int, q: int, n_points: int,
        cuda: bool = False
):
    """
    control_points: (n, m, 3)
    u_knots: (n, 3)
    v_knots: (m, 3)
    """
    n, m, __ = control_points.shape
    u = torch.linspace(0, 1, n_points).float()
    if cuda:
        u = u.cuda()
    u_basis_matrix = torch.stack([torch_basis_function_array(u, i, p, u_knots, {}, cuda) for i in range(n)], dim=1)  # (N, n)

    v = torch.linspace(0, 1, n_points).float()
    if cuda:
        v = v.cuda()
    v_basis_matrix = torch.stack([torch_basis_function_array(v, i, q, v_knots, {}, cuda) for i in range(m)], dim=1)  # (N, m)
    u_basis_matrix = u_basis_matrix.unsqueeze(2).unsqueeze(1)  # (N, 1, n, 1)
    u_basis_matrix = u_basis_matrix.repeat(1, n_points, 1, m)  # (N, N, n, m)

    v_basis_matrix = v_basis_matrix.unsqueeze(1).unsqueeze(0)  # (1, N, 1, m)
    v_basis_matrix = v_basis_matrix.repeat(n_points, 1, n, 1)  # (N, N, n, m)
    basis_matrix = u_basis_matrix * v_basis_matrix  # (N, N, n, m)
    basis_matrix = basis_matrix.reshape(basis_matrix.shape[0], basis_matrix.shape[1],  basis_matrix.shape[2] * basis_matrix.shape[3])
    control_points = control_points.reshape(control_points.shape[0] * control_points.shape[1], control_points.shape[2])
    surface = torch.matmul(basis_matrix, control_points)  # (N, N, 3)
    return surface


def torch_b_spline_surface_interpolating_points(points, p, q, n_points, cuda):
    u_hat, v_hat, u_knots, v_knots = torch_b_spline_surface_knots(
        points=points,
        p=p, q=q,
        cuda=cuda,
    )

    control_points = torch_b_spline_surface_control_points_given_points(
        points=points,
        u_hat=u_hat,
        v_hat=v_hat,
        u_knots=u_knots,
        v_knots=v_knots,
        p=p,
        q=q,
        cuda=cuda,
    )

    surface = torch_b_spline_surface(
        control_points=control_points,
        u_knots=u_knots,
        v_knots=v_knots,
        p=p, q=q, n_points=n_points, cuda=cuda
    )
    return surface, control_points


def sample_lv(center, radius, z_end, n_slices, n_points, cuda: bool = False, plot: bool = False):
    """

    Args:
        center: (3,)
        radius: (1,)
        z_end: (1,)
        n_slices: (1,)
    Returns:

    """
    d_max = 0.5
    xy = (center[:2] / 2 + 0.5) * img_dim  # (3,)
    z = (center[2] / 2 + 0.5) * img_height  # (3,)
    z = z.unsqueeze(0)
    center = torch.cat([xy, z], dim=0)
    radius = radius * img_dim
    z_end = z_end * img_height

    dz = (z_end - center[2]) / n_slices  # (1,)
    dz = dz.repeat(n_slices - 1)  # (n_lv - 1, )
    if cuda:
        c1z = center[2] + torch.cat([torch.zeros(1).cuda(), torch.cumsum(dz, dim=0)], dim=0)  # (n_lv, )
    else:
        c1z = center[2] + torch.cat([torch.zeros(1), torch.cumsum(dz, dim=0)], dim=0)  # (n_lv, )

    c0x = center[0].repeat(n_slices)
    c0y = center[1].repeat(n_slices)

    radius = torch.sqrt(1 - (c1z - center[2]) * (c1z - center[2]) / ((z_end - center[2]) * (z_end - center[2]))) * radius  # (n_lv, )

    c0_phase = torch.arange(n_points).repeat(n_slices, 1)
    c0_phase = 2 * math.pi * c0_phase / n_points  # (B, n_lv, n_points)
    real = torch.tensor(0).float().repeat(n_slices, n_points)
    if cuda:
        c0_phase = c0_phase.cuda()
        real = real.cuda()
    c0_angle = torch.exp(
        torch.complex(
            real=real,
            imag=c0_phase,
        )
    )  # (B, n_lv, n_points)
    z_c0 = torch.complex(real=c0x, imag=c0y).unsqueeze(1)  # (n_lv, 1)
    lv_xy = z_c0.repeat(1, n_points) + torch.mul(radius.unsqueeze(1).repeat(1, n_points), c0_angle)  # (n_lv, n_points)
    lv_xy = torch.cat([lv_xy, z_c0], dim=1)  # (n_lv, n_points + 1)
    lv_xy = torch.view_as_real(lv_xy)  # (n_lv, n_points + 1, 2)

    lv_points = torch.cat([lv_xy, c1z.unsqueeze(1).repeat(1, n_points + 1).unsqueeze(2)], dim=2)  # (n_lv, n_points + 1, 3)
    lv_points = lv_points.view(-1, lv_points.shape[2])

    if plot:
        lv_pcd, lv_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_points.detach().cpu().numpy(),
        )
        o3d.visualization.draw_geometries([lv_pcd], point_show_normal=True)
        points = np.asarray(lv_tetra_mesh.vertices)
        o3d.visualization.draw_geometries([lv_pcd, lv_tetra_mesh], mesh_show_back_face=False)

    lv_points[..., :2] = (lv_points[..., :2] - img_dim / 2) / img_dim * 2
    lv_points[..., 2] = (lv_points[..., 2] - img_height / 2) / img_height * 2
    return lv_points, c0x, c0y, c1z, radius


def sample_rv_points(
    c0x, c0y, c0z, lv_radii, z_end, theta_c2, theta2, d_c2_c0_ratio,
    n_points, n_slices, cuda: bool = False, plot: bool = False, param: bool = False, surface2_arc_out_param=None
):
    """

    Args:
        c0x: (n_slices, )
        c0y: (n_slices, )
        c0z: (n_slices, )
        lv_radii: (n_slices, )
        z_end: (1, )
        theta_c2: (1,)
        theta2: (1,)
        d_c2_c0_ratio: (1,)
        n_points:
        n_slices: n_lv
        cuda:

    Returns:

    """

    d_max = 0.5
    # d_theta_max = math.pi / (num_slices - 1) / 2
    #
    # theta2_max = torch.tensor(math.pi * 3 / 4).float().repeat(batch_size).cuda().unsqueeze(1)
    # theta2_min = torch.tensor(math.pi * 1 / 6).float().repeat(batch_size).cuda().unsqueeze(1)
    #
    # theta2 = theta2_min + torch.mul(theta2_ratio.unsqueeze(1), theta2_max - theta2_min)  # (B, 1)
    z_end = z_end * img_height
    z_c0 = torch.complex(real=c0x, imag=c0y)  # (n_rv, )
    dmin = lv_radii * 1 / 2  # (n_rv, )
    dmax = lv_radii * 3 / 2  # (n_rv, )

    d_c2_c0 = torch.mul(d_c2_c0_ratio, dmax - dmin) + dmin  # (n_rv, )

    theta_c2 = theta_c2 * math.pi * 2  # (1, )
    theta2 = theta2 * math.pi

    r2 = torch.sqrt(lv_radii[0] * lv_radii[0] + d_c2_c0[0] * d_c2_c0[0] - 2 * lv_radii[0] * d_c2_c0[0] * torch.cos(theta2[0]))  # (1,)
    x2 = d_c2_c0[0] + r2

    dz1 = z_end - c0z[0]
    x2 = torch.sqrt(1 - (c0z - c0z[0]) * (c0z - c0z[0]) / (dz1 * dz1)) * x2  # (n_rv, )
    r2 = x2 - d_c2_c0  # (n_rv, )
    theta2 = torch.arccos((lv_radii * lv_radii + d_c2_c0 * d_c2_c0 - r2 * r2) / (2 * lv_radii * d_c2_c0)).float()  # (n_rv, )

    if cuda:
        z_c2 = z_c0 + d_c2_c0 * torch.exp(
            torch.complex(real=torch.tensor(0).float().cuda().repeat(n_slices), imag=theta_c2)
        )  # (B, n_lv)
    else:
        z_c2 = z_c0 + d_c2_c0 * torch.exp(
            torch.complex(real=torch.tensor(0).float().repeat(n_slices), imag=theta_c2)
        )  # (B, n_lv)
    theta_p0 = theta_c2 - theta2  # theta_p0 = (-pi, 2pi), (n_rv, )
    theta_p1 = theta_c2 + theta2  # theta_p1 = (0, 3pi), (n_rv, )

    # theta_p0, theta_p1 = (0, 2pi)
    theta_p0 = torch.where(
        theta_p0 < 0,
        theta_p0 + math.pi * 2,
        theta_p0,
    )  # (n_rv, )

    theta_p1 = torch.where(
        theta_p1 > math.pi * 2,
        theta_p1 - math.pi * 2,
        theta_p1,
    )  # (n_rv, )

    theta_p1 = torch.where(
        theta_p1 < theta_p0,
        theta_p1 + math.pi * 2,
        theta_p1
    )  # (n_rv, )
    n_arc_points = n_points // 2

    theta_p0 = theta_p0.unsqueeze(1).repeat(1, n_arc_points)  # (n_rv, n_arc_points)
    theta_p1 = theta_p1.unsqueeze(1).repeat(1, n_arc_points)  # (n_rv, n_arc_points)

    if cuda:
        arc_count = torch.arange(n_arc_points).unsqueeze(0).repeat(n_slices, 1).cuda()  # (n_rv, n_arc_points)
    else:
        arc_count = torch.arange(n_arc_points).unsqueeze(0).repeat(n_slices, 1)  # (n_rv, n_arc_points)
    arc_phase = theta_p0 + torch.mul(theta_p1 - theta_p0, arc_count) / (n_arc_points - 1)  # (n_rv, n_arc_points)
    if cuda:
        arc_angle = torch.exp(
            torch.complex(real=torch.tensor(0).float().repeat(n_slices, n_arc_points).cuda(), imag=arc_phase.float())
        )  # (n_rv, n_arc_points)
    else:
        arc_angle = torch.exp(
            torch.complex(real=torch.tensor(0).float().repeat(n_slices, n_arc_points), imag=arc_phase.float())
        )  # (n_rv, n_arc_points)
    arc = z_c0.unsqueeze(1).repeat(1, n_arc_points) + torch.mul(lv_radii.unsqueeze(1).repeat(1, n_arc_points), arc_angle)  # (n_lv, n_arc_points)
    arc_1 = torch.flip(arc, dims=[1])  # p1 to p0 arc

    r2 = (torch.view_as_real(z_c2) - torch.view_as_real(arc_1[..., -1])).norm(dim=1)  # (n_lv, )
    c0z = c0z.unsqueeze(1).repeat(1, n_arc_points).unsqueeze(2)  # (n_lv, n_arc_points, 1)

    if surface2_arc_out_param is None:
        theta_c2_p0 = torch.log(arc_1[..., -1] - z_c2).imag  # theta_c2_p0 = (-pi, pi), (n_lv, )
        theta_c2_p1 = torch.log(arc_1[..., 0] - z_c2).imag  # theta_c2_p1 = (-pi, pi), (n_lv, )

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
        theta_c2_p0 = theta_c2_p0.unsqueeze(1).repeat(1, n_arc_points)  # (n_lv, n_arc_points)
        theta_c2_p1 = theta_c2_p1.unsqueeze(1).repeat(1, n_arc_points)  # (n_lv, n_arc_points)

        arc_phase = theta_c2_p1 + torch.mul(theta_c2_p0 - theta_c2_p1, arc_count) / (n_arc_points - 1)  # (n_lv, n_arc_points)
        if cuda:
            arc_angle = torch.exp(
                torch.complex(real=torch.tensor(0).float().repeat(n_slices, n_arc_points).cuda(), imag=arc_phase)
            )  # (B, n_lv, n_arc_points)
        else:
            arc_angle = torch.exp(
                torch.complex(real=torch.tensor(0).float().repeat(n_slices, n_arc_points), imag=arc_phase)
            )  # (B, n_lv, n_arc_points)
        arc_2 = z_c2.unsqueeze(1).repeat(1, n_arc_points) + torch.mul(r2.unsqueeze(1).repeat(1, n_arc_points), arc_angle)  # (n_lv, n_arc_points)
        arc_2 = torch.view_as_real(arc_2)  # (n_lv, n_arc_points, 2)
        arc_2 = arc_2[:, 1:-1]

        # 3D params
        # surface2_arc_out = torch.cat([arc_2, c0z[:, 1:-1]], dim=2)  # (n_lv, n_arc_points - 2, 1)
        #
        # # arc_2 = torch.cat([arc_1[:, 0].unsqueeze(1), arc_2, arc_1[:, -1].unsqueeze(1)], dim=1)  # (n_lv, n_arc_points, 2)
        # if param:
        #     print("param")
        #     np_surface2_arc_out = surface2_arc_out.detach().cpu().numpy().copy()
        #     if cuda:
        #         surface2_arc_out_param = torch.nn.Parameter(torch.from_numpy(np_surface2_arc_out / img_dim).float().cuda())
        #     else:
        #         surface2_arc_out_param = torch.nn.Parameter(torch.from_numpy(np_surface2_arc_out / img_dim).float())
        # else:
        #     surface2_arc_out_param = surface2_arc_out / img_dim

        # 2D params
        surface2_arc_out = arc_2

        # arc_2 = torch.cat([arc_1[:, 0].unsqueeze(1), arc_2, arc_1[:, -1].unsqueeze(1)], dim=1)  # (n_lv, n_arc_points, 2)
        if param:
            print("param")
            np_surface2_arc_out = surface2_arc_out.detach().cpu().numpy().copy()
            if cuda:
                surface2_arc_out_param = torch.nn.Parameter(torch.from_numpy(np_surface2_arc_out / img_dim).float().cuda())
            else:
                surface2_arc_out_param = torch.nn.Parameter(torch.from_numpy(np_surface2_arc_out / img_dim).float())
        else:
            surface2_arc_out_param = surface2_arc_out / img_dim
    arc_1 = torch.view_as_real(arc_1)  # (n_lv, n_arc_points, 2)
    arc_1_0 = torch.cat([arc_1[:, 0].unsqueeze(1), c0z[:, 0].unsqueeze(1)], dim=2)  # (n_lv, 1, 1)
    arc_1_1 = torch.cat([arc_1[:, -1].unsqueeze(1), c0z[:, -1].unsqueeze(1)], dim=2)  # (n_lv, 1, 1)
    surface2_arc_out = surface2_arc_out_param * img_dim

    # 2D params
    surface2_arc_out = torch.cat([surface2_arc_out, c0z[:, 1:-1]], dim=2)  # (n_lv, n_arc_points - 2, 1)

    surface2_arc_out = torch.cat([arc_1_0, surface2_arc_out, arc_1_1], dim=1)
    surface2_arc_out, cps = torch_b_spline_surface_interpolating_points(surface2_arc_out, 4, 4, n_points // 2, cuda)

    # surface2_arc_out = surface2_arc_out.transpose(0, 1)
    # assert 1 == 0
    arc_1 = arc_1.view(-1, arc_1.shape[-1])  # (n_lv * n_arc_points, 2)
    # arc_2 = arc_2.view(-1, arc_2.shape[-1])  # (n_lv * n_arc_points, 2)
    c0z_flat = c0z.view(-1, c0z.shape[-1])  # (n_lv * n_arc_points, 1)
    surface2_arc_in = torch.cat([arc_1, c0z_flat], dim=1)

    surface2_arc_in[..., :2] = (surface2_arc_in[..., :2] - img_dim / 2) / img_dim * 2    # (n_lv * n_arc_points, 3)
    surface2_arc_in[..., 2] = (surface2_arc_in[..., 2] - img_height / 2) / img_height * 2

    surface2_arc_out[..., :2] = (surface2_arc_out[..., :2] - img_dim / 2) / img_dim * 2
    surface2_arc_out[..., 2] = (surface2_arc_out[..., 2] - img_height / 2) / img_height * 2    # (n_lv * n_arc_points, 3)


    if plot:
        pcd = o3d.geometry.PointCloud()
        surface2_arc_out_plot = surface2_arc_out.view(-1, surface2_arc_out.shape[-1])
        points = torch.cat([surface2_arc_in, surface2_arc_out_plot], dim=0)
        pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(
            np.concatenate(
                [
                    np.tile(np.array([[255, 0, 0]]), (surface2_arc_in.shape[0], 1)),
                    np.tile(np.array([[0, 255, 0]]), (surface2_arc_out_plot.shape[0], 1)),
                ],
                axis=0,
            )
        )
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        rv_pcd, rv_tetras_mesh = o3d_volumetric_mesh(
            vertices=points.detach().cpu().numpy(),
        )
        rv_tetras = rv_volumentric_tetra(surface2_arc_in.shape[0] * 2, n_points)

        rv_tetras_mesh.tetras = o3d.utility.Vector4iVector(rv_tetras.astype(np.int32))
        rv_tetras_mesh.vertices = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        rv_pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())

        visualise_o3d_mesh(
            mesh=rv_tetras_mesh,
            pcd=rv_pcd,
            is_triangle=False,
            show_pcd_normal=False,
            show_voxel=False,
            voxel_size=0.02,
            fname="rv_tetra"
        )
    return surface2_arc_in, surface2_arc_out, surface2_arc_out_param


def display(lv_mask, rv_mask, label):
    from mayavi import mlab
    from CMRSegment.common.nn.torch.augmentation import resize_image
    opacity = 0.999
    label = torch.movedim(label, 2, -1)

    print(lv_mask.shape, label.shape)
    label = label.detach().cpu().numpy()[0]
    lv_label = np.zeros_like(label[0])
    lv_label[[label[0] == 1]] = 1
    rv_label = np.zeros_like(label[0])
    rv_label[[label[1] == 1]] = 1

    for masks, prefix in zip([[lv_mask, rv_mask]], ["init"]):
        print(prefix)
        map = masks[1][0, 0].detach().cpu().numpy()
        xx, yy, zz = np.where(map > 0.5)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(1, 1, 0),
                             scale_factor=1, transparent=True, opacity=opacity)

        map = masks[0][0, 0].detach().cpu().numpy()
        xx, yy, zz = np.where(map > 0.5)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 1, 1),
                             scale_factor=1, transparent=True, opacity=opacity)

        xx, yy, zz = np.where(label[0] >= 0)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 0, 0),
                             scale_factor=1, transparent=True, opacity=0)
        mlab.outline()

        xx, yy, zz = np.where(lv_label > 0.5)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 0, 1),
                             scale_factor=1, opacity=1)

        xx, yy, zz = np.where(rv_label > 0.5)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(1, 0, 0),
                             scale_factor=1, opacity=1)
        mlab.show()


def save_masks(lv_mask: torch.Tensor, rv_mask, affine, output_dir: Path, i):
    mask = np.zeros((lv_mask.shape[0], lv_mask.shape[1], lv_mask.shape[2]))
    mask[lv_mask.detach().cpu().numpy() == 1] = 1
    mask[rv_mask.detach().cpu().numpy() == 1] = 2
    nim2 = nib.Nifti1Image(mask, affine)
    # nim2.header['pixdim'] = nim.header['pixdim']
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/seg_{1}.nii.gz'.format(str(output_dir), str(i)))
    pass


def save_resizes(image, label, affine, output_dir: Path):
    out_label = np.zeros(image.shape)
    out_label[label[0, :, :, :] == 1] = 1
    out_label[label[1, :, :, :] == 1] = 2
    nim2 = nib.Nifti1Image(out_label, affine)
    # nim2.header['pixdim'] = nim.header['pixdim']
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/resized_label.nii.gz'.format(str(output_dir)))

    nim2 = nib.Nifti1Image(image, affine)
    # nim2.header['pixdim'] = nim.header['pixdim']
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/resized_image.nii.gz'.format(str(output_dir)))


img_dim = 128
img_height = 32
n_slices = 25
n_points = 50
n_interpolation_points = 10
cuda = True
plot = False
param = True
output_dir = Path(__file__).parent
label, __ = ccitk.read_nii_label(Path(__file__).parent.joinpath("label.nii.gz"), labels=[1, 2, 3])
img, affine = ccitk.read_nii_image(Path(__file__).parent.joinpath("image.nii.gz"))
label[0, :, :, :][label[1, :, :, :] == 1] = 1
label[1, :, :, :] = label[2, :, :, :]
label = label[:2, :, :, :]
print(label.shape, img.shape)
img = ccitk.resize_image(img, (img_dim, img_dim, img_height), order=2)
label = ccitk.resize_label(label, (img_dim, img_dim, img_height))
save_resizes(img, label, affine, output_dir)
print(label.shape, img.shape)


center = torch.from_numpy(np.array([0, 0, 0])).float() / img_dim
radius = torch.from_numpy(np.array([10])).float() / img_dim
z_end = torch.from_numpy(np.array([26])).float() / img_height

theta_c2 = torch.from_numpy(np.array([0])).float() / (math.pi * 2)
theta2 = torch.from_numpy(np.array([math.pi/3])).float() / math.pi
d_c2_c0_ratio = torch.from_numpy(np.array([0.33])).float()

if cuda:
    center = center.cuda()
    radius = radius.cuda()
    z_end = z_end.cuda()
    theta_c2 = theta_c2.cuda()
    theta2 = theta2.cuda()
    d_c2_c0_ratio = d_c2_c0_ratio.cuda()

if param:
    center = torch.nn.Parameter(center)
    radius = torch.nn.Parameter(radius)
    z_end = torch.nn.Parameter(z_end)
    theta_c2 = torch.nn.Parameter(theta_c2)
    theta2 = torch.nn.Parameter(theta2)
    d_c2_c0_ratio = torch.nn.Parameter(d_c2_c0_ratio)

lv_points, c0x, c0y, c1z, radius_out = sample_lv(
    center=center,
    radius=radius,
    z_end=z_end,
    n_slices=n_slices,
    n_points=n_points,
    plot=plot,
    cuda=cuda,
)

surface2_arc_in, surface2_arc_out, surface2_arc_out_param = sample_rv_points(
    c0x=c0x,
    c0y=c0y,
    c0z=c1z,
    lv_radii=radius_out,
    z_end=z_end,
    theta_c2=theta_c2,
    theta2=theta2,
    d_c2_c0_ratio=d_c2_c0_ratio,
    n_points=n_points,
    n_slices=n_slices,
    cuda=cuda,
    plot=plot,
    param=False,
    surface2_arc_out_param=None,
)
from rasterizor.voxelize import Voxelize

# surface2_arc_in: (n_slices * n_points, 3)
# surface2_arc_out: (n_slices, n_points, 3)
voxeliser = Voxelize(voxel_width=img_dim, voxel_depth=img_dim, voxel_height=img_height, eps=1e-4, eps_in=20)
np_rv_tetras = rv_volumentric_tetra(surface2_arc_in.shape[0]*2, n_points)
np_lv_tetras = lv_volumetric_tetra(n_slices, n_points + 1)

lv_tetras = torch.from_numpy(np_lv_tetras)
rv_tetras = torch.from_numpy(np_rv_tetras)

if cuda:
    lv_tetras = lv_tetras.cuda()
    rv_tetras = rv_tetras.cuda()

rv_points = torch.cat([surface2_arc_in, surface2_arc_out.view(-1, surface2_arc_out.shape[-1])], dim=0)
rv_points = rv_points / 2 * img_dim + img_dim / 2
rv_points[..., :2] = (rv_points[..., :2] - img_dim / 2) / img_dim * 2
rv_points[..., 2] = (rv_points[..., 2] - img_height / 2) / img_height * 2

print(rv_points.detach().cpu().numpy())
params = [center, radius, z_end, theta_c2, theta2, d_c2_c0_ratio]
# params = [arc_2]

for p in params:
    print(p.is_leaf)
optimizer = torch.optim.Adam(
    params,
    lr=1e-2,
)
loss_criterion = torch.nn.MSELoss()
label = torch.from_numpy(label).float().cuda()
pretrain_step = 50
for i in range(1000):
    print(i)
    # print(center, radius, z_end, theta_c2, theta2, d_c2_c0_ratio)
    pred_mask_lv = voxelize_mask(voxeliser, lv_points.unsqueeze(0), lv_tetras.unsqueeze(0))
    rv_points = torch.cat([surface2_arc_in, surface2_arc_out.view(-1, surface2_arc_out.shape[-1])], dim=0)
    pred_mask_rv = voxelize_mask(voxeliser, rv_points.unsqueeze(0), rv_tetras.unsqueeze(0))

    if i % 10 == 0:
        # rv_points[..., :2] = rv_points[..., :2] / 2 * img_dim + img_dim / 2
        # rv_points[..., 2] = rv_points[..., 2] / 2 * img_height + img_height / 2
        print(surface2_arc_out_param.detach().cpu().numpy() * img_dim)
        save_masks(pred_mask_lv.squeeze(0).squeeze(0), pred_mask_rv.squeeze(0).squeeze(0), affine, output_dir, i)
    loss = loss_criterion(pred_mask_lv, label[0, :, :, :]) + loss_criterion(pred_mask_rv, label[1, :, :, :])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # display(pred_mask_lv, pred_mask_rv, labe

    lv_points, c0x, c0y, c1z, radius_out = sample_lv(
        center=center,
        radius=radius,
        z_end=z_end,
        n_slices=n_slices,
        n_points=n_points,
        plot=plot,
        cuda=cuda,
    )

    if i < pretrain_step:
        surface2_arc_in, surface2_arc_out, surface2_arc_out_param = sample_rv_points(
            c0x=c0x,
            c0y=c0y,
            c0z=c1z,
            lv_radii=radius_out,
            z_end=z_end,
            theta_c2=theta_c2,
            theta2=theta2,
            d_c2_c0_ratio=d_c2_c0_ratio,
            n_points=n_points,
            n_slices=n_slices,
            cuda=cuda,
            plot=plot,
            param=False,
            surface2_arc_out_param=None,
        )
    elif i == pretrain_step:
        surface2_arc_in, surface2_arc_out, surface2_arc_out_param = sample_rv_points(
            c0x=c0x,
            c0y=c0y,
            c0z=c1z,
            lv_radii=radius_out,
            z_end=z_end,
            theta_c2=theta_c2,
            theta2=theta2,
            d_c2_c0_ratio=d_c2_c0_ratio,
            n_points=n_points,
            n_slices=n_slices,
            cuda=cuda,
            plot=plot,
            param=True,
            surface2_arc_out_param=None,
        )
        params = [center, radius, z_end, theta_c2, theta2, d_c2_c0_ratio, surface2_arc_out_param]
        optimizer = torch.optim.Adam(
            params,
            lr=1e-4,
        )
    else:
        surface2_arc_in, surface2_arc_out, surface2_arc_out_param = sample_rv_points(
            c0x=c0x,
            c0y=c0y,
            c0z=c1z,
            lv_radii=radius_out,
            z_end=z_end,
            theta_c2=theta_c2,
            theta2=theta2,
            d_c2_c0_ratio=d_c2_c0_ratio,
            n_points=n_points,
            n_slices=n_slices,
            cuda=cuda,
            plot=plot,
            param=False,
            surface2_arc_out_param=surface2_arc_out_param,
        )
    lv_tetras = torch.from_numpy(np_lv_tetras)
    rv_tetras = torch.from_numpy(np_rv_tetras)
    if cuda:
        lv_tetras = lv_tetras.cuda()
        rv_tetras = rv_tetras.cuda()
    # if i > pretrain_step:
    #     rv_tri = Delaunay(rv_xy.detach().cpu().numpy()).simplices.copy()
    #     rv_tri = triangulate_within(rv_xy.detach().cpu().numpy(), rv_tri)
    #     rv_tri = rv_tri.copy()
    #     rv_tri = torch.from_numpy(rv_tri)
