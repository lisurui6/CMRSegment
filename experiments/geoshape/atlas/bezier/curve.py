import math
import torch
import numpy as np
from matplotlib import pyplot as plt
import neural_renderer as nr
from scipy.special import comb
# import nibabel as nib
import ccitk
from pathlib import Path
from scipy.spatial import Delaunay
from shapely import geometry
import shutil
import cv2

# 2D RV bezier curve

img_dim = 342
lv_center = torch.nn.Parameter(torch.from_numpy(np.array([img_dim/2, img_dim/2]) / img_dim).float().cuda())
radius = torch.nn.Parameter(torch.tensor(0.3).float().cuda())

theta0 = torch.nn.Parameter(torch.tensor(0).float().cuda())
dtheta = torch.nn.Parameter(torch.tensor(1/3).float().cuda())
d_c2_c0 = torch.nn.Parameter(torch.tensor(0.3 + 5 / img_dim).float().cuda())

# lv_center.requires_grad = True

n_points = 500
n_control_points = 10


def sample_lv(lv_center, radius):
    lv_center = lv_center * img_dim
    radius = radius * img_dim
    c0_phase = torch.arange(n_points).cuda()
    c0_phase = 2 * math.pi * c0_phase / n_points
    c0_angle = torch.exp(
        torch.complex(
            real=torch.tensor(0).float().cuda(),
            imag=c0_phase,
        )
    )
    z_c0 = torch.complex(real=lv_center[0], imag=lv_center[1])
    lv_xy = z_c0 + c0_angle * radius
    lv_xy = torch.view_as_real(lv_xy)
    return lv_xy

# np_lv_xy = lv_xy.detach().cpu().numpy()


def init_rv(lv_center, radius, theta0, dtheta, d_c2_c0, param: bool = False):
    lv_center = lv_center * img_dim
    d_c2_c0 = d_c2_c0 * img_dim
    theta0 = theta0 * math.pi
    dtheta = dtheta * math.pi
    radius = radius * img_dim
    z_c0 = torch.complex(real=lv_center[0], imag=lv_center[1])

    theta_p0 = torch.exp(
        torch.complex(
            real=torch.tensor(0).float().cuda(),
            imag=theta0,
        )
    )

    theta_p1 = torch.exp(
        torch.complex(
            real=torch.tensor(0).float().cuda(),
            imag=theta0 + dtheta,
        )
    )

    p0 = z_c0 + theta_p0 * radius
    p1 = z_c0 + theta_p1 * radius

    z_c2 = z_c0 + d_c2_c0 * torch.exp(
        torch.complex(real=torch.tensor(0).float().cuda(), imag=theta0 + dtheta / 2.)
    )  # (B, n_lv)

    r2 = (torch.view_as_real(z_c2) - torch.view_as_real(p0)).norm()  # (B, n_lv)
    theta_c2_p0 = torch.log(p0 - z_c2).imag  # theta_c2_p0 = (-pi, pi), (B, n_lv)
    theta_c2_p1 = torch.log(p1 - z_c2).imag  # theta_c2_p1 = (-pi, pi), (B, n_lv)
    theta_c2_p0 = theta_c2_p0.unsqueeze(0).repeat(n_control_points)  # (B, n_lv, n_arc_points)
    theta_c2_p1 = theta_c2_p1.unsqueeze(0).repeat(n_control_points)  # (B, n_lv, n_arc_points)

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

    arc_count = torch.arange(n_control_points).cuda()
    arc_phase = theta_c2_p1 + torch.mul(theta_c2_p0 - theta_c2_p1, arc_count) / (n_control_points - 1)  # (B, n_lv, n_arc_points)
    arc_angle = torch.exp(
        torch.complex(real=torch.tensor(0).float().cuda(), imag=arc_phase)
    )  # (B, n_lv, n_arc_points)
    arc_2 = z_c2 + torch.mul(r2, arc_angle)  # (B, n_lv, n_arc_points)
    arc_2 = torch.view_as_real(arc_2)  # (B, n_lv, n_arc_points, 2)

    if param:
        np_arc_2 = arc_2.detach().cpu().numpy().copy()
        arc_2 = torch.nn.Parameter(torch.from_numpy(np_arc_2 / img_dim).float().cuda())
    else:
        arc_2 = arc_2 / img_dim

    return arc_2


def sample_rv(lv_center, radius, theta0, dtheta, arc_2, param: bool = False, weights=None):
    lv_center = lv_center * img_dim
    radius = radius * img_dim
    theta0 = theta0 * math.pi
    dtheta = dtheta * math.pi
    arc_2 = arc_2 * img_dim

    arc_count = torch.arange(n_points).cuda()
    z_c0 = torch.complex(real=lv_center[0], imag=lv_center[1])

    theta_p0 = theta0.unsqueeze(0).repeat(n_points)  # (B, n_lv, n_arc_points)
    theta_p1 = (theta0 + dtheta).unsqueeze(0).repeat(n_points)  # (B, n_lv, n_arc_points)

    theta_p0 = torch.where(
        theta_p0 < 0,
        theta_p0 + math.pi * 2,
        theta_p0,
    )

    theta_p1 = torch.where(
        theta_p1 < 0,
        theta_p1 + math.pi * 2,
        theta_p1,
    )

    theta_p1 = torch.where(
        theta_p0 > theta_p1,
        theta_p1 + math.pi * 2,
        theta_p1,
    )

    arc_phase = theta0 + torch.mul(theta_p1 - theta_p0, arc_count) / (n_points - 1)  # (B, n_rv, n_arc_points)
    arc_angle = torch.exp(
        torch.complex(real=torch.tensor(0).float().cuda(), imag=arc_phase.float())
    )
    arc_1 = z_c0 + torch.mul(radius, arc_angle)  # (B, n_lv, n_arc_points)
    # arc_1 = torch.flip(arc, dims=[0])  # p1 to p0 arc
    arc_1 = torch.view_as_real(arc_1)  # (B, n_lv, n_arc_points, 2)
    b_arc = torch.cat([arc_1[-1].unsqueeze(0), arc_2, arc_1[0].unsqueeze(0)], dim=0)
    if weights is None:
        if not param:
            weights = torch.ones(1, b_arc.shape[0]).float().cuda()
        else:
            weights = torch.nn.Parameter(torch.ones(1, b_arc.shape[0]).float().cuda())

    xvals, yvals = torch_bezier_curve(b_arc, n_times=n_points, weights=weights)
    arc_2 = torch.stack([xvals, yvals], dim=1)
    rv_xy = torch.cat([arc_1, arc_2])

    return rv_xy, b_arc, weights

# np_arc_1 = arc_1.detach().cpu().numpy()
# np_rv_xy = rv_xy.detach().cpu().numpy()

# plt.plot(np_arc_2[:, 0], np_arc_2[:, 1], "b-")
# plt.plot(np_lv_xy[:, 0], np_lv_xy[:, 1], "r-")
# plt.plot(np_rv_xy[:, 0], np_rv_xy[:, 1], "bx-")


def basis_function(i, n, t):
    return comb(n, i) * (t ** i) * (1 - t) ** (n - i)


def torch_bezier_curve(points, n_times=1000, weights=None):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    n_points = len(points)
    t = np.linspace(0.0, 1.0, n_times)
    if weights is None:
        weights = torch.ones(1, n_points).float().cuda()

    polynomial_array = torch.from_numpy(np.array([basis_function(i, n_points-1, t) for i in range(0, n_points)])).float().cuda()

    W = torch.matmul(weights, polynomial_array)  # (1, u)
    # xvals = np.dot(x_points, polynomial_array)
    # yvals = np.dot(y_points, polynomial_array)

    x_vals = torch.mul(W, torch.matmul(points[:, 0].unsqueeze(0), polynomial_array)).squeeze()
    y_vals = torch.mul(W, torch.matmul(points[:, 1].unsqueeze(0), polynomial_array)).squeeze()

    return x_vals, y_vals


# xvals, yvals = torch_bezier_curve(arc_2, n_times=n_control_points)
# plt.plot(xvals.detach().numpy(), yvals.detach().numpy(), "y.-")
# plt.show()

"""
doing learning, 

RV parameters are rv control points, arc_2, and theta, dtheta, d_c2_c0

LV parameters are center and radius
"""


def triangulate_within(vert, faces):
    polygon = geometry.Polygon(vert)
    output = []
    for f in range(faces.shape[0]):
        face = faces[f, :]
        triangle = geometry.Polygon(vert[face, :])
        if triangle.within(polygon):
            output.append(face)
    if len(output) == 0:
        vert = vert * img_dim//2 + img_dim//2
        plt.imshow(np.zeros((img_dim, img_dim)))
        plt.plot(vert[:, 0], vert[:, 1], 'bx-')
        for f in range(faces.shape[0]):
            p1 = vert[faces[f, 0], :]
            p2 = vert[faces[f, 1], :]
            p3 = vert[faces[f, 2], :]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
            plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
            plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')

        plt.show()
    output = np.stack(output)
    return output


output_dir = Path(__file__).parent.joinpath("output")
if output_dir.exists():
    shutil.rmtree(str(output_dir))
output_dir.mkdir(parents=True, exist_ok=True)

label, __ = ccitk.read_nii_image(Path(__file__).parent.joinpath("label.nii.gz"))
img, __ = ccitk.read_nii_image(Path(__file__).parent.joinpath("image.nii.gz"))
label_slice = label[:, :, 40]
img_slice = img[:, :, 40]
label_slice = cv2.resize(label_slice, dsize=(img_dim, img_dim), interpolation=cv2.INTER_CUBIC)
img_slice = cv2.resize(img_slice, dsize=(img_dim, img_dim), interpolation=cv2.INTER_CUBIC)
img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
label = np.zeros((label_slice.shape[0], label_slice.shape[1], 2))
label[:, :, 0][label_slice == 1] = 1
label[:, :, 0][label_slice == 2] = 1
label[:, :, 1][label_slice == 3] = 1
# plt.imshow(label_slice)
# plt.show()

lv_xy = sample_lv(lv_center, radius)
arc_2 = init_rv(lv_center, radius, theta0, dtheta, d_c2_c0, False)
rv_xy, b_arc, weights = sample_rv(lv_center, radius, theta0, dtheta, arc_2)

np_rv_xy = rv_xy.detach().cpu().numpy()
np_b_arc = b_arc.detach().cpu().numpy()
plt.figure()
plt.plot(np_rv_xy[:, 0], np_rv_xy[:, 1], "r-")
plt.plot(np_b_arc[:, 0], np_b_arc[:, 1], "bx")
plt.plot([np_b_arc[0, 0]], [np_b_arc[0, 1]], "yo")
plt.plot([np_b_arc[-1, 0]], [np_b_arc[-1, 1]], "go")
plt.savefig(str(output_dir.joinpath(f"rv.png")))
plt.close()

lv_tri = Delaunay(lv_xy.detach().cpu().numpy()).simplices.copy()

rv_tri = Delaunay(rv_xy.detach().cpu().numpy()).simplices.copy()
rv_tri = triangulate_within(rv_xy.detach().cpu().numpy(), rv_tri)
rv_tri = rv_tri.copy()

lv_tri = torch.from_numpy(lv_tri)
rv_tri = torch.from_numpy(rv_tri)

lv_xy = (lv_xy - img_dim//2) / img_dim * 2
rv_xy = (rv_xy - img_dim//2) / img_dim * 2


def plot(nodes0, face0, nodes2, face2):
    half_dim = img_dim//2
    nodes0[:, 1] = -nodes0[:, 1]
    nodes0 = nodes0 * half_dim + half_dim

    nodes2[:, 1] = -nodes2[:, 1]
    nodes2 = nodes2 * half_dim + half_dim

    nodes0 = nodes0.detach().cpu().numpy()
    nodes2 = nodes2.detach().cpu().numpy()

    face0 = face0.detach().cpu().numpy()
    face2 = face2.detach().cpu().numpy()
    # plt.show()
    # plt.figure()
    plt.imshow(np.zeros((img_dim, img_dim)))
    plt.plot(nodes2[0, 0], nodes2[0, 1], 'go')
    plt.plot(nodes2[-1, 0], nodes2[-1, 1], 'ro')

    plt.plot(nodes2[:, 0], nodes2[:, 1], 'bx-')
    for f in range(face2.shape[0]):
        p1 = nodes2[face2[f, 0], :]
        p2 = nodes2[face2[f, 1], :]
        p3 = nodes2[face2[f, 2], :]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
        plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
        plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')

    plt.figure()
    plt.imshow(np.zeros((img_dim, img_dim)))
    plt.plot(nodes0[0, 0], nodes0[0, 1], 'go')
    plt.plot(nodes0[-1, 0], nodes0[-1, 1], 'ro')

    plt.plot(nodes0[:, 0], nodes0[:, 1], 'bx-')
    for f in range(face0.shape[0]):
        p1 = nodes0[face0[f, 0], :]
        p2 = nodes0[face0[f, 1], :]
        p3 = nodes0[face0[f, 2], :]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
        plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
        plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')

    plt.figure()
    plt.imshow(np.zeros((img_dim, img_dim)))

    plt.show()


# plot(lv_xy, lv_tri, rv_xy, rv_tri)

renderer = nr.Renderer(
            camera_mode='look_at',
            image_size=img_dim,
            light_intensity_ambient=1,
            light_intensity_directional=1,
            perspective=False
)


def render_mask(renderer, nodes, faces):
    z = torch.ones((nodes.shape[0], 1)).to(nodes.device)
    P3d = torch.cat((nodes, z), 1)
    P3d = P3d.unsqueeze(0).cuda()
    faces = faces.unsqueeze(0).cuda()
    # P3d = torch.squeeze(P3d, dim=1)
    # faces = torch.squeeze(faces, dim=1).to(nodes.device)
    mask = renderer(P3d, faces, mode='silhouettes').squeeze(0)
    return mask


params = [lv_center, radius, theta0, dtheta, d_c2_c0]

for p in params:
    print(p.is_leaf)
optimizer = torch.optim.Adam(
    params,
    lr=1e-2,
)
loss_criterion = torch.nn.MSELoss()
label = torch.from_numpy(label).float().cuda()

pretrain_step = 300

for i in range(1000):

    lv_mask = render_mask(renderer, nodes=lv_xy, faces=lv_tri)
    rv_mask = render_mask(renderer, nodes=rv_xy, faces=rv_tri)
    # if i < pretrain_step:
    #     arc_2.requires_grad = False
    # else:
    #     arc_2.requires_grad = True
    loss = loss_criterion(lv_mask, label[:, :, 0]) + loss_criterion(rv_mask, label[:, :, 1])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # plt.imshow(rv_mask.detach().cpu().numpy())
    if i % 50 == 0:
        mask = np.zeros((img_dim, img_dim, 3))
        mask[lv_mask.detach().cpu().numpy() > 0] = (255, 0, 0)
        mask[rv_mask.detach().cpu().numpy() > 0] = (0, 255, 0)
        plt.figure()
        plt.imshow(img_slice, alpha=0.5)
        plt.imshow(mask, alpha=0.5)
        plt.savefig(str(output_dir.joinpath(f"mask_{i}.png")))
        np_rv_xy = rv_xy.detach().cpu().numpy()
        np_rv_xy = np_rv_xy * img_dim//2 + img_dim//2
        np_b_arc = b_arc.detach().cpu().numpy()
        # np_arc_2 = np_arc_2 * img_dim
        plt.figure()
        plt.plot(np_rv_xy[:, 0], np_rv_xy[:, 1], "r-")
        plt.plot(np_b_arc[:, 0], np_b_arc[:, 1], "bx")
        plt.plot([np_b_arc[0, 0]], [np_b_arc[0, 1]], "yo")
        plt.plot([np_b_arc[-1, 0]], [np_b_arc[-1, 1]], "go")
        plt.savefig(str(output_dir.joinpath(f"rv_{i}.png")))
        plt.close()
        print(lv_center, radius, theta0, dtheta, d_c2_c0)
        print(weights)

        # plt.imsave(str(output_dir.joinpath(f"rv_mask_{i}.png")), rv_mask.detach().cpu().numpy())

    lv_xy = sample_lv(lv_center, radius)
    if i < pretrain_step:
        arc_2 = init_rv(lv_center, radius, theta0, dtheta, d_c2_c0, False)
        rv_xy, b_arc, weights = sample_rv(lv_center, radius, theta0, dtheta, arc_2, False)
    elif i == pretrain_step:
        arc_2 = init_rv(lv_center, radius, theta0, dtheta, d_c2_c0, True)
        rv_xy, b_arc, weights = sample_rv(lv_center, radius, theta0, dtheta, arc_2, True)
        params = [lv_center, radius, theta0, dtheta, d_c2_c0, arc_2, weights]
        optimizer = torch.optim.Adam(
            params,
            lr=1e-4,
        )
    else:
        rv_xy, b_arc, weights = sample_rv(lv_center, radius, theta0, dtheta, arc_2, weights=weights)

    # if i > pretrain_step:
    #     rv_tri = Delaunay(rv_xy.detach().cpu().numpy()).simplices.copy()
    #     rv_tri = triangulate_within(rv_xy.detach().cpu().numpy(), rv_tri)
    #     rv_tri = rv_tri.copy()
    #     rv_tri = torch.from_numpy(rv_tri)

    lv_xy = (lv_xy - img_dim // 2) / img_dim * 2
    rv_xy = (rv_xy - img_dim // 2) / img_dim * 2
