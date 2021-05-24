import numpy as np
import cv2
import math
from pathlib import Path
from matplotlib import pyplot as plt


from CMRSegment.common.topology import Point, OpenBall, Curve, PointSet, L2Distance, Arc, tangent, normal
from experiments.active_contour.graph import CardiacSnakeGraph


point = Point((1, 2))
point = point + Point((2, 3))
print(point)

atlas = cv2.imread(str(Path(__file__).parent.joinpath("label2.png")))
subject = cv2.imread(str(Path(__file__).parent.joinpath("label5.png")))
# atlas in BGR
label_blue = np.where(atlas[:, :, 2] < 100)
label_red = np.where(atlas[:, :, 0] < 100)


from scipy.interpolate import splprep, splev
from scipy.integrate import simps
from skimage import measure


def mask_to_boundary_pts(mask, pt_spacing=10):
    """
    Convert a binary image containing a single object to a set
    of 2D points that are equally spaced along the object's contour.
    """
    # interpolate boundary
    boundary_pts = measure.find_contours(mask, 0)[0]
    tck, u = splprep(boundary_pts.T, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    # get equi-spaced points along spline-interpolated boundary
    x_diff, y_diff = np.diff(x_new), np.diff(y_new)
    S = simps(np.sqrt(x_diff**2 + y_diff**2))
    N = int(round(S/pt_spacing))

    u_equidist = np.linspace(0, 1, N+1)
    x_equidist, y_equidist = splev(u_equidist, tck, der=0)
    return np.array(list(zip(x_equidist, y_equidist)))


atlas_image = np.zeros((atlas.shape[0], atlas.shape[1]))
atlas_image[label_blue] = 1
atlas_image[label_red] = 0.5

subject_image = np.zeros((subject.shape[0], subject.shape[1]))
subject_image[np.where(subject[:, :, 0] < 100)] = 1
subject_image[np.where(subject[:, :, 2] < 100)] = 2

x = np.where(atlas_image == 1)
image = np.zeros((atlas.shape[0], atlas.shape[1]))
image[label_blue] = 1
label_blue = image

image = np.zeros((atlas.shape[0], atlas.shape[1]))
image[label_red] = 1
label_red = image

# blue_bdry = mask_to_boundary_pts(label_blue)
blue_bdry = measure.find_contours(label_blue, 0)[0]
blue_bdry = Curve.from_numpy_array(blue_bdry)
delta = 30
blue_bdry = Curve([blue_bdry[i] for i in range(0, len(blue_bdry), delta)])


# red_bdry = mask_to_boundary_pts(label_red)
red_bdry = measure.find_contours(label_red, 0)[0]
red_bdry = Curve.from_numpy_array(red_bdry)
red_bdry = Curve([red_bdry[i] for i in range(0, len(red_bdry), delta)])

func = L2Distance()

blue_bdry_set = blue_bdry.set()
arc = red_bdry

for idx, point in enumerate(red_bdry):
    ball = blue_bdry_set & OpenBall(point, delta/2 + 1)
    if not ball.is_empty():
        arc = arc.delete_point(point)
crossed = arc
connected1 = crossed[0]
connected2 = crossed[-1]
print(connected1, connected2)

new_blues = []
red_bdry_set = red_bdry.set()
arc = blue_bdry
for idx, point in enumerate(blue_bdry):
    ball = red_bdry_set & OpenBall(point, delta/2 + 1)
    if not ball.is_empty():
        arc = arc.delete_point(point)

new_blue_bdry = arc
print(len(new_blue_bdry), len(red_bdry))
graph = CardiacSnakeGraph.from_curves(lv_curve=red_bdry, rv_arc=new_blue_bdry, lv_connected_point1=connected1, lv_connected_point2=connected2)
print(len(graph.get_vertices()))
fig = plt.figure()
plt.imshow(atlas_image)
graph.plot(fig)
plt.show()

# # connect connected points to new blue bdry
# dist1 = func(connected1, new_blue_bdry[0])
# dist2 = func(connected1, new_blue_bdry[-1])
#
# if dist1 > dist2:
#     new_blue_bdry.append(connected1)
#     new_blue_bdry.points.insert(0, connected2)
# else:
#     new_blue_bdry.points.insert(0, connected1)
#     new_blue_bdry.append(connected2)
#
# plt.figure()
# plt.imshow(atlas_image)
# plot_new_blue_bdry = new_blue_bdry.to_numpy_array()
# plot_red_bdry = red_bdry.to_numpy_array()
# print(plot_new_blue_bdry.shape, plot_red_bdry.shape)
# plt.plot(plot_red_bdry[:, 1], plot_red_bdry[:, 0], "ro-", )
# plt.plot(plot_new_blue_bdry[:, 1], plot_new_blue_bdry[:, 0], "bo-", )
# plt.plot(connected1[1], connected1[0], "yo", )
# plt.plot(connected2[1], connected2[0], "go", )
# for point in red_bdry:
#     tangent = red_bdry.normal(point)
#     plt.arrow(point[1], point[0], 30*tangent[1], 30*tangent[0], width=3)
#
# for point in new_blue_bdry:
#     tangent = new_blue_bdry.normal(point)
#     plt.arrow(point[1], point[0], 30*tangent[1], 30*tangent[0], width=3)
# plt.show()


def distance_map(label_image: np.ndarray, label: int):
    new_image = np.zeros(label_image.shape, dtype=np.uint8)
    new_image[label_image == label] = 1
    new_image = cv2.Laplacian(new_image, cv2.CV_8UC1)
    new_new_image = np.zeros(label_image.shape, dtype=np.uint8)
    new_new_image[new_image == 0] = 1
    # new_image = np.expand_dims(new_image, axis=0)
    print(new_image.shape)
    dmap = cv2.distanceTransform(new_new_image, cv2.DIST_L2, 3)
    # l2_distance = L2Distance()
    # label_coordinates = np.where(label_image == label)
    # points = [Point((x, y)) for x, y in zip(label_coordinates[0], label_coordinates[1])]
    # point_set = PointSet(*points)
    # print(len(point_set))
    # dmap = np.zeros(label_image.shape)
    # for i in range(label_image.shape[0]):
    #     for j in range(label_image.shape[1]):
    #         print(i, j)
    #         if Point((i, j)) not in point_set:
    #             distance = [l2_distance(Point((i, j)), point) for point in point_set]
    #             dmap[i, j] = min(distance)
    return dmap

from scipy import ndimage


def signed_distance(f, label):
    """Return the signed distance to the 0.5 levelset of a function."""

    # Prepare the embedding function.
    new_image = np.zeros(f.shape, dtype=np.uint8)
    new_image[f == label] = 1
    f = new_image > 0.5

    # Signed distance transform
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f) - 0.5, -(dist_func(1-f) - 0.5))
    return distance

red_dmap = signed_distance(subject_image, 1)
red_g_x, red_g_y = np.gradient(red_dmap)
red_g_x = np.multiply(red_dmap/2, -red_g_x)
red_g_y = np.multiply(red_dmap/2, -red_g_y)

blue_dmap = signed_distance(subject_image, 2)
blue_g_x, blue_g_y = np.gradient(blue_dmap)
blue_g_x = np.multiply(blue_dmap/2, -blue_g_x)
blue_g_y = np.multiply(blue_dmap/2, -blue_g_y)


plt.figure()
plt.imshow(red_g_x)
plt.figure()
plt.imshow(red_dmap)
# plt.figure()
# plt.imshow(g_y)
plt.show()
# iterate over blue and red boundary to evolve both curves.



graph = CardiacSnakeGraph.from_curves(lv_curve=red_bdry, rv_arc=new_blue_bdry, lv_connected_point1=connected1, lv_connected_point2=connected2)
fig = plt.figure()
plt.imshow(atlas_image)
graph.plot(fig)
step_size = 1
alpha = 0.2
beta = 0
gamma = 1
# for e in range(100):
#     fig = plt.figure()
#     plt.imshow(subject_image)
#     graph.plot(fig)
#     for node in graph.get_vertices():
#
#         # img force
#         if node.label == "lv":
#             img_v_x = red_g_x[int(node.point[0]), int(node.point[1])]
#             img_v_y = red_g_y[int(node.point[0]), int(node.point[1])]
#             img_v = Point((img_v_x, img_v_y))
#         else:
#             img_v_x = blue_g_x[int(node.point[0]), int(node.point[1])]
#             img_v_y = blue_g_y[int(node.point[0]), int(node.point[1])]
#             img_v = Point((img_v_x, img_v_y))
#         # elastic force
#         elastic_force = node.elastic_force
#         stiff_force = node.stiff_force
#         # plt.arrow(node.point[1], node.point[0], img_v[1], img_v[0], width=3)
#         plt.arrow(node.point[1], node.point[0], elastic_force[1] * alpha, elastic_force[0] * alpha, width=3)
#         # plt.arrow(node.point[1], node.point[0], stiff_force[1] * beta, stiff_force[0] * beta, width=3)
#         v = img_v * gamma + elastic_force * alpha + stiff_force * beta
#         # plt.arrow(node.point[1], node.point[0], v_y, v_x, width=3)
#         # plt.arrow(node.point[1], node.point[0], v[1], v[0], width=3)
#
#         # print(node, v)
#         # plt.arrow(node.point[1], node.point[0], v[1], v[0], width=3)
#         node.point += v * step_size
#         # node = graph.get_vertex(node_uid)
#     graph.update_vectors()
#
#     print("here")
#     plt.show()
#
#     pass



for e in range(100):
    fig = plt.figure()
    plt.imshow(subject_image)
    graph.plot(fig)
    # for node in graph.get_vertices():
    #     plt.arrow(node.point[1], node.point[0], 10*node.tangent[1], 10*node.tangent[0], width=3)
    #     plt.arrow(node.point[1], node.point[0], 10*node.normal[1], 10*node.normal[0], width=3)
    for node in graph.get_vertices():

        if node.uid() not in [graph.lv_start_id, graph.lv_end_id]:
            if node.label == "lv":
                v_x = red_g_x[int(node.point[0]), int(node.point[1])]
                v_y = red_g_y[int(node.point[0]), int(node.point[1])]
            else:
                v_x = blue_g_x[int(node.point[0]), int(node.point[1])]
                v_y = blue_g_y[int(node.point[0]), int(node.point[1])]
            node.v_n = Point((v_x, v_y))
            # node.v_n = node.normal * node.normal.inner_product(Point((v_x, v_y)))
        else:
            v_x = red_g_x[int(node.point[0]), int(node.point[1])]
            v_y = red_g_y[int(node.point[0]), int(node.point[1])]
            node.v_n = Point((v_x, v_y))
            # node.v_n = node.normal * node.normal.inner_product(Point((v_x, v_y)))

            v_x = blue_g_x[int(node.point[0]), int(node.point[1])]
            v_y = blue_g_y[int(node.point[0]), int(node.point[1])]

            #
            # tangent = node.tangent
            # v_t = tangent * tangent.inner_product(Point((v_x, v_y)))
            # node.v_t = v_t
            node.v_t = Point((v_x, v_y))


            # tangent = graph.get_vertex(node.uid() - 2).tangent
            # graph.get_vertex(node.uid() - 2).v_t = tangent * tangent.inner_product(Point((v_x, v_y))) * 0.5

            # tangent = graph.get_vertex(node.uid() - 1).tangent
            # graph.get_vertex(node.uid() - 1).v_t = tangent * tangent.inner_product(Point((v_x, v_y))) * 0.8

            # tangent = graph.get_vertex(node.uid() + 1).tangent
            # graph.get_vertex(node.uid() + 1).v_t = tangent * tangent.inner_product(Point((v_x, v_y))) * 0.8

            # tangent = graph.get_vertex(node.uid() + 2).tangent
            # graph.get_vertex(node.uid() + 2).v_t = tangent * tangent.inner_product(Point((v_x, v_y))) * 0.5

            # plt.arrow(node.point[1], node.point[0], v_y, v_x, width=3)
            # plt.arrow(node.point[1], node.point[0], v_t[1], v_t[0], width=3)

        # print(node, v)
        # plt.arrow(node.point[1], node.point[0], v[1], v[0], width=3)

    for node in graph.get_vertices():
        # plt.arrow(node.point[1], node.point[0], node.stiff_force[1] * beta, node.stiff_force[0] * beta, width=3)
        plt.arrow(node.point[1], node.point[0], node.elastic_force[1] * alpha, node.elastic_force[0] * alpha, width=3)
        # plt.arrow(node.point[1], node.point[0], node.v_n[1] * gamma, node.v_n[0] * gamma, width=3)
        # if node.v_t != Point((0, 0)):
        #     print(node.v_t)
        #     plt.arrow(node.point[1], node.point[0], node.v_t[1] * gamma, node.v_t[0] * gamma, width=3)

        node.point += ((node.v_t + node.v_n) * gamma + node.elastic_force * alpha + node.stiff_force * beta) * step_size
        # print(node.uid(), node.v_t)
        # node.point += (img_v * 0.5 + node.v_t) * step_size

        # node = graph.get_vertex(node_uid)
    graph.update_vectors()

    plt.show()

    pass

plt.figure()

for e in range(100):
    new_blues = []
    step_size = 0.5
    for idx, point in enumerate(new_blue_bdry):
        print(point)
        v_x = blue_g_x[int(point[0]), int(point[1])]
        v_y = blue_g_y[int(point[0]), int(point[1])]
        plt.arrow(point[1], point[0], v_y, v_x, width=3)

        normal = new_blue_bdry.normal(point)
        v_n = normal * normal.inner_product(Point((v_x, v_y)))
        new_point = point + v_n * step_size
        new_blues.append(new_point)
    # for idx, point in zip([0, -1], [new_blue_bdry[0], new_blue_bdry[-1]]):
    #     v_x = red_g_x[int(point[0]), int(point[1])]
    #     v_y = red_g_y[int(point[0]), int(point[1])]
    #     normal = red_bdry.normal(point)
    #     v_n = normal * normal.inner_product(Point((v_x, v_y)))
    #
    #     v_x = blue_g_x[int(point[0]), int(point[1])]
    #     v_y = blue_g_y[int(point[0]), int(point[1])]
    #     tangent = red_bdry.tangent(point)
    #     v_t = tangent * tangent.inner_product(Point((v_x, v_y)))
    #     v = v_n + v_t
    #     new_point = point + v * step_size
    #     if idx == 0:
    #         new_blues.insert(0, new_point)
    #     else:
    #         new_blues.append(new_point)
    new_reds = []
    for idx, point in enumerate(red_bdry):
        v_x = red_g_x[int(point[0]), int(point[1])]
        v_y = red_g_y[int(point[0]), int(point[1])]
        normal = red_bdry.normal(point)
        v_n = normal * normal.inner_product(Point((v_x, v_y)))
        new_point = point + v_n * step_size
        new_reds.append(new_point)
    new_blue_bdry = Arc(new_blues)
    red_bdry = Curve(new_reds)
    plt.imshow(subject_image)
    plot_new_blue_bdry = new_blue_bdry.to_numpy_array()
    plot_red_bdry = red_bdry.to_numpy_array()
    plt.plot(plot_red_bdry[:, 1], plot_red_bdry[:, 0], "rx-", )
    plt.plot(plot_new_blue_bdry[:, 1], plot_new_blue_bdry[:, 0], "bx-", )
    plt.plot(new_blue_bdry[0][1], new_blue_bdry[0][0], "yo", )
    plt.plot(new_blue_bdry[-1][1], new_blue_bdry[-1][0], "go", )

    plt.show()

# g_y = np.multiply(dmap, g_y)
# x, y = np.meshgrid(np.linspace(0, dmap.shape[1], dmap.shape[1]),np.linspace(0, dmap.shape[0], dmap.shape[0]))
# plt.quiver(x, y, g_x, g_y)

# g_dmap = cv2.Laplacian(dmap, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REPLICATE)
# map = np.multiply(dmap, g_dmap)
# # cv2.normalize(g_dmap, g_dmap, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
plt.imshow(map)
plt.show()

