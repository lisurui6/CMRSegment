import torch
import numpy as np


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
    b_i_p_1 = torch_basis_function_array(u, i, p - 1, knots, values)
    if (knots[i + p] - knots[i]) == 0.:
        part1 = torch.zeros(b_i_p_1.shape).float()

    else:
        part1 = torch.mul((u - knots[i]) / (knots[i + p] - knots[i]), b_i_p_1)
    b_i_p_1 = torch_basis_function_array(u, i + 1, p - 1, knots, values)
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


def tourch_b_spline_curve_control_points_given_points(points: torch.Tensor, u_hat: torch.Tensor, u_knots, p: int):
    n = u_hat.shape[0]
    values = {}
    basis_matrix = torch.stack([torch_basis_function_array(u_hat, i, p, u_knots, values) for i in range(n)], dim=1)
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
        p: int, q: int
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


points = np.array(
    [
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [2, 1, 2],
            [3, 1, 1],
            [4, 1, 0],
        ],
        [
            [0, 2, 0],
            [1, 2, 2],
            [2, 2, 3],
            [3, 2, 2],
            [4, 2, 0],
        ],
        [
            [0, 3, 0],
            [1, 3, 1],
            [2, 3, 2],
            [3, 3, 1],
            [4, 3, 0],
        ],
        [
            [0, 4, 0],
            [1, 4, 0],
            [2, 4, 0],
            [3, 4, 0],
            [4, 4, 0],
        ],
    ]

)
print(points.shape)

boundary1 = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 2, 0],
        [0, 3, 0],
        [0, 4, 0],
    ]
)

boundary2 = np.array(
    [
        [0, 4, 0],
        [1, 4, 0],
        [2, 4, 0],
        [3, 4, 0],
        [4, 4, 0],
    ]
)

boundary3 = np.array(
    [
        [4, 0, 0],
        [4, 1, 0],
        [4, 2, 0],
        [4, 3, 0],
        [4, 4, 0],
    ]
)

boundary4 = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0],
    ]
)


points1 = np.array(
    [
        [1, 1, 1],
        [1, 3, 1],
        [3, 1, 1],
        [3, 3, 1],
    ]
)

points2 = np.array(
    [
        [1, 2, 2],
        [2, 1, 2],
        [2, 3, 2],
        [3, 2, 2],
    ]
)
points3 = np.array(
    [
        [2, 2, 3],
    ]
)
from mayavi import mlab
# vertices
mlab.points3d(boundary1[:, 0], boundary1[:, 1], boundary1[:, 2], mode="sphere", color=(0, 1, 0), opacity=1, scale_factor=0.2)
mlab.points3d(boundary2[:, 0], boundary2[:, 1], boundary2[:, 2], mode="sphere", color=(0, 1, 0), opacity=1, scale_factor=0.2)
mlab.points3d(boundary3[:, 0], boundary3[:, 1], boundary3[:, 2], mode="sphere", color=(0, 1, 0), opacity=1, scale_factor=0.2)
mlab.points3d(boundary4[:, 0], boundary4[:, 1], boundary4[:, 2], mode="sphere", color=(0, 1, 0), opacity=1, scale_factor=0.2)
mlab.points3d(points1[:, 0], points1[:, 1], points1[:, 2], mode="sphere", color=(0, 0.5, 0.5), opacity=1, scale_factor=0.2)
mlab.points3d(points2[:, 0], points2[:, 1], points2[:, 2], mode="sphere", color=(0, 0, 1), opacity=1, scale_factor=0.2)
mlab.points3d(points3[:, 0], points3[:, 1], points3[:, 2], mode="sphere", color=(1, 0, 0), opacity=1, scale_factor=0.2)

cuda = False

points = torch.from_numpy(points).float()
if cuda:
    points = points.cuda()
p = 2
q = 2
n_points = 100
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
    q=q
)

surface = torch_b_spline_surface(
    control_points=control_points,
    u_knots=u_knots,
    v_knots=v_knots,
    p=p, q=q, n_points=n_points, cuda=cuda
)
np_control_points = control_points.numpy()
np_surface = surface.numpy()
print(np_surface.shape)
np_surface = np.reshape(np_surface, (np_surface.shape[0] * np_surface.shape[1], np_surface.shape[2]))

x = np.linspace(0, 4, n_points)
y = np.linspace(0, 4, n_points)
print(x.shape, y.shape)
mlab.points3d(np_surface[:, 0], np_surface[:, 1], np_surface[:, 2], mode="sphere", color=(1, 0, 0), opacity=1, scale_factor=0.05)
mlab.points3d(np_control_points[:, :, 0], np_control_points[:, :, 1], np_control_points[:, :, 2], mode="sphere", color=(0, 0, 0), opacity=1, scale_factor=0.2)

print(points.numpy())
print(control_points.numpy())

# mlab.surf(x=x, y=y, s=np_surface[:, :, 2], opacity=0.5)
mlab.show()
