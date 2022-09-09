import math
import torch
import numpy as np
import time


def basis_function(u, i, p, knots, values=None):
    if values is None:
        values = {}
    if p == 0:
        if knots[i] <= u <= knots[i + 1]:
            values[(i, p)] = 1
            return 1
        else:
            values[(i, p)] = 0
            return 0
    if (i, p) in values:
        return values[(i, p)]
    b_i_p_1 = basis_function(u, i, p - 1, knots, values)
    if (knots[i + p] - knots[i]) == 0:
        part1 = 0
    else:
        part1 = (u - knots[i]) / (knots[i + p] - knots[i]) * b_i_p_1
    b_i_p_1 = basis_function(u, i + 1, p - 1, knots, values)
    if knots[i + p + 1] - knots[i + 1] == 0:
        part2 = 0
    else:
        part2 = (knots[i + p + 1] - u) / (knots[i + p + 1] - knots[i + 1]) * b_i_p_1
    values[(i, p)] = part1 + part2
    return part1 + part2


def basis_function_array(u: np.ndarray, i, p, knots, values=None):
    if values is None:
        values = {}
    if p == 0:
        v_ip = np.where(
            (u <= knots[i + 1]) & (knots[i] <= u),
            1,
            0
        )
        values[(i, p)] = v_ip
        return v_ip
    if (i, p) in values:
        return values[(i, p)]
    b_i_p_1 = basis_function_array(u, i, p - 1, knots, values)
    if (knots[i + p] - knots[i]) == 0:
        part1 = np.zeros(b_i_p_1.shape)
    else:
        part1 = np.multiply((u - knots[i]) / (knots[i + p] - knots[i]), b_i_p_1)
    b_i_p_1 = basis_function_array(u, i + 1, p - 1, knots, values)
    if knots[i + p + 1] - knots[i + 1] == 0:
        part2 = np.zeros(b_i_p_1.shape)
    else:
        part2 = np.multiply((knots[i + p + 1] - u) / (knots[i + p + 1] - knots[i + 1]), b_i_p_1)
    values[(i, p)] = part1 + part2
    return part1 + part2


def torch_basis_function_array(u: torch.Tensor, i, p, knots: torch.Tensor, values=None):
    if values is None:
        values = {}
    if p == 0:
        v_ip = torch.where(
            (u <= knots[i + 1]) & (knots[i] <= u),
            1,
            0
        )
        values[(i, p)] = v_ip
        return v_ip
    if (i, p) in values:
        return values[(i, p)]
    b_i_p_1 = torch_basis_function_array(u, i, p - 1, knots, values)
    if (knots[i + p] - knots[i]) == 0:
        part1 = torch.zeros(b_i_p_1.shape)
    else:
        part1 = torch.mul((u - knots[i]) / (knots[i + p] - knots[i]), b_i_p_1)
    b_i_p_1 = torch_basis_function_array(u, i + 1, p - 1, knots, values)
    if knots[i + p + 1] - knots[i + 1] == 0:
        part2 = torch.zeros(b_i_p_1.shape)
    else:
        part2 = torch.mul((knots[i + p + 1] - u) / (knots[i + p + 1] - knots[i + 1]), b_i_p_1)
    values[(i, p)] = part1 + part2
    return part1 + part2


def torch_nonuniform_b_spline_curve_interpolating_points(points: torch.Tensor, degree: int, N):
    """
    Global interpolation of points, return curve evaluated and control points
    Args:
        points: (n, 3)

    Returns:
        Evaluated curve and Control points
    """
    diff = points[1:] - points[:-1]  # (n - 1, 3)
    diff = torch.sqrt(torch.sum(diff * diff, dim=-1))  # (n - 1, )
    d = torch.sum(diff)
    diff = torch.concat([torch.tensor([0]).float(), diff / d])
    u_hat = torch.cumsum(diff, dim=0)  # (n, )
    n = points.shape[0]
    p = degree
    pre_knots = torch.zeros(p + 1).float()
    post_knots = torch.ones(p + 1).float()
    knots = torch.nn.AvgPool1d(kernel_size=p, stride=1)(u_hat[1:-1].float().unsqueeze(0)).squeeze(0)
    knots = torch.concat([pre_knots, knots, post_knots], dim=0)  # (n + p + 1)
    u = torch.linspace(0, 1, N).float()
    basis_matrix = torch.stack([torch_basis_function_array(u_hat, i, p, knots) for i in range(n)], dim=1)
    control_points = torch.matmul(torch.inverse(basis_matrix), points)  # (n, 2)
    basis_matrix = torch.stack([torch_basis_function_array(u, i, p, knots, {}) for i in range(n)], dim=1)  # (N, n)
    curve = torch.matmul(basis_matrix, control_points)  # (N, 2)

    return curve, control_points


def numpy_nonuniform_b_spline_curve_interpolating_points(points: np.ndarray, degree: int, N):
    """
    Global interpolation of points, return curve evaluated and control points
    Args:
        points: (N, 3)

    Returns:
        Evaluated curve and Control points
    """
    diff = points[1:] - points[:-1]  # (N - 1, 3)
    diff = np.sqrt(np.sum(diff * diff, axis=-1))  # (N - 1, )
    d = np.sum(diff)
    diff = np.concatenate([np.array([0.]), diff / d])
    u_hat = np.cumsum(diff)  # (N, )
    n = points.shape[0]
    p = degree
    pre_knots = np.zeros(p + 1)
    post_knots = np.ones(p + 1)
    knots = torch.nn.AvgPool1d(kernel_size=p, stride=1)(torch.from_numpy(u_hat[1:-1]).float().unsqueeze(0)).squeeze(0).numpy()
    knots = np.concatenate([pre_knots, knots, post_knots], axis=0)
    u = np.linspace(0, 1, N)

    basis_matrix = np.stack([basis_function_array(u_hat, i, p, knots) for i in range(n)], axis=1)
    control_points = np.matmul(np.linalg.inv(basis_matrix), points)  # (n, 2)
    basis_matrix = np.stack([basis_function_array(u, i, p, knots, {}) for i in range(n)], axis=1)  # (N, n)
    curve = np.matmul(basis_matrix, control_points)  # (N, 2)

    return curve, control_points


points = np.array(
    [
        [0, 0],
        [3, 4],
        [-1, 4],
        [-4, 0],
        [-4, -3],
    ]
)


curve, control_points = torch_nonuniform_b_spline_curve_interpolating_points(
    points=torch.from_numpy(points).float(),
    degree=3,
    N=100,
)


# curve, control_points = numpy_nonuniform_b_spline_curve_interpolating_points(
#     points=points,
#     degree=3,
#     N=100,
# )

from matplotlib import pyplot as plt

plt.plot(points[:, 0], points[:, 1], "bx")
plt.plot(control_points[:, 0], control_points[:, 1], "ro")
plt.plot(curve[:, 0], curve[:, 1], "r-")
plt.show()

results = basis_function(
    u=5/2,
    i=4,
    p=0,
    knots=np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]),
)

print(results)


results = basis_function(
    u=5/2,
    i=3,
    p=1,
    knots=np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]),
)

print(results)

results = basis_function(
    u=5/2,
    i=2,
    p=2,
    knots=np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]),
)

print(results)

results = basis_function(
    u=5/2,
    i=4,
    p=1,
    knots=np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]),
)

print(results)

results = basis_function(
    u=5/2,
    i=3,
    p=2,
    knots=np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]),
)

print(results)

results = basis_function(
    u=5/2,
    i=4,
    p=2,
    knots=np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]),
)

print(results)


results = basis_function(
    u=5,
    i=7,
    p=2,
    knots=np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]),
)

print(results)
