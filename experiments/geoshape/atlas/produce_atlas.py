import numpy as np
from scipy.special import comb



import vtk
import numpy as np
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy
from typing import Tuple
import nibabel as nib


def read_nii_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read image from nii.gz file and return image and affine matrix (4*4)"""
    nim = nib.load(str(path))
    image = nim.get_data()
    if image.ndim == 4:
        image = np.squeeze(image, axis=-1).astype(np.int16)
    image = image.astype(np.float32)
    return image, nim.affine


def read_vkt_mesh(vtk_path: Path, affine: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_path))
    reader.Update()
    mesh = reader.GetOutput()

    vertices = vtk_to_numpy(mesh.GetPoints().GetData())
    vertices = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    if affine is not None:
        vertices = np.matmul(affine, np.transpose(vertices))
        vertices = np.transpose(vertices)
    vertices = vertices[:, :3]

    triangles = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray())
    triangles = np.reshape(triangles, (-1, 3))
    return vertices, triangles


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t**(n-i)) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
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

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


# if __name__ == "__main__":
#     # from matplotlib import pyplot as plt
#     # import math
#     # r = 4
#     # thetas = np.linspace(0, 2*math.pi, num=10)
#     # xpoints = [r * math.cos(theta) for theta in thetas]
#     # ypoints = [r * math.sin(theta) for theta in thetas]
#     # points = [[x, y] for x, y in zip(xpoints, ypoints)]
#     # # nPoints = 4
#     # # points = np.random.rand(nPoints, 2)*200
#     # # xpoints = [p[0] for p in points]
#     # # ypoints = [p[1] for p in points]
#     #
#     # xvals, yvals = bezier_curve(points, nTimes=1000)
#     # plt.plot(xvals, yvals)
#     # plt.plot(xpoints, ypoints, "ro")
#     # for nr in range(len(points)):
#     #     plt.text(points[nr][0], points[nr][1], nr)
#     #
#     # plt.show()
#
#     import pyvista as pv
#     import tetgen
#     import numpy as np
#     import vtk
#     import ccitk
#
#
#     img_nii_path = Path(__file__).parent.joinpath("vtk_RV_ED.nii.gz")
#     image, affine = ccitk.read_nii_image(img_nii_path)
#     print(affine)
#
#     vertices, triangles = ccitk.read_vkt_mesh(Path(__file__).parent.joinpath("RV_ED.vtk"), affine=affine)
#     # pv.set_plot_theme('document')
#     #
#     # sphere = pv.Sphere()
#     triangles = np.concatenate([np.ones((triangles.shape[0], 1))*3, triangles], axis=1)
#     triangles = np.int32(triangles)
#     mesh = pv.PolyData(vertices, triangles)
#
#     tet = tetgen.TetGen(mesh)
#     nodes, tetras = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
#     grid = tet.grid
#     grid.plot(show_edges=True)
#
#     # get cell centroids
#     cells = grid.cells.reshape(-1, 5)[:, 1:]
#     cell_center = grid.points[cells].mean(1)
#
#     # extract cells below the 0 xy plane
#     mask = cell_center[:, 2] < 0
#     cell_ind = mask.nonzero()[0]
#     subgrid = grid.extract_cells(cell_ind)
#
#     # advanced plotting
#     plotter = pv.Plotter()
#     plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
#     plotter.add_mesh(mesh, 'r', 'wireframe')
#     plotter.add_legend([[' Input Mesh ', 'r'],
#                         [' Tessellated Mesh ', 'black']])
#     plotter.show()


import ccitk

import pyvista as pv
import tetgen
import numpy as np


ATLAS_DIR = Path(__file__).parent

atlas_lv_endo_mesh_path = ATLAS_DIR.joinpath("LVendo_ED.vtk")
atlas_lv_epi_mesh_path = ATLAS_DIR.joinpath("LVepi_ED.vtk")
atlas_rv_mesh_path = ATLAS_DIR.joinpath("RV_ED.vtk")
atlas_rv_seg_path = ATLAS_DIR.joinpath("vtk_RV_ED.nii.gz")

atlas_rv, affine = ccitk.read_nii_image(atlas_rv_seg_path)


def read_atlas_nodes(affine=None, plot=False):
    atlas_rv_mesh_decimated_path = ATLAS_DIR.joinpath("decimated_RV_ED.vtk")
    ccitk.decimate_mesh(
        mesh_path=atlas_rv_mesh_path,
        output_path=atlas_rv_mesh_decimated_path,
        downsample_rate=98.8,
        preserve_topology=True,
        match_points=True,
    )
    rv_nodes, rv_tris = ccitk.read_vkt_mesh(atlas_rv_mesh_decimated_path, None)
    print(np.min(rv_nodes[:, 0]), np.max(rv_nodes[:, 0]))
    print(np.min(rv_nodes[:, 1]), np.max(rv_nodes[:, 1]))
    print(np.min(rv_nodes[:, 2]), np.max(rv_nodes[:, 2]))
    rv_nodes, rv_tris = ccitk.read_vkt_mesh(atlas_rv_mesh_decimated_path, affine)
    print(np.min(rv_nodes[:, 0]), np.max(rv_nodes[:, 0]))
    print(np.min(rv_nodes[:, 1]), np.max(rv_nodes[:, 1]))
    print(np.min(rv_nodes[:, 2]), np.max(rv_nodes[:, 2]))

    rv_nodes, rv_tetras = ccitk.surface_mesh_to_volumetric_mesh(rv_nodes, rv_tris, plot)
    print(np.min(rv_nodes[:, 2]), np.max(rv_nodes[:, 2]))

    lv_nodes, lv_tris = ccitk.read_vkt_mesh(atlas_lv_endo_mesh_path, affine)
    lv_myo_nodes, lv_myo_tris = ccitk.read_vkt_mesh(atlas_lv_epi_mesh_path, affine)

    print("finish reading atlas meshes")


    print("finish tetgen rv mesh")

    lv_nodes, lv_tetras = ccitk.surface_mesh_to_volumetric_mesh(lv_nodes, lv_tris, plot)
    print("finish tetgen lv mesh")

    lv_myo_nodes, lv_myo_tetras = ccitk.surface_mesh_to_volumetric_mesh(lv_myo_nodes, lv_myo_tris, plot)
    print("finish tetgen lv myo mesh")

    return [lv_nodes, lv_tetras], [lv_myo_nodes, lv_myo_tetras], [rv_nodes, rv_tetras]


print(affine)
affine = np.linalg.inv(affine)
print(affine)
[lv_nodes, lv_tetras], [lv_myo_nodes, lv_myo_tetras], [rv_nodes, rv_tetras] = read_atlas_nodes(affine, plot=True)

np.save("atlas_lv_nodes.npy", lv_nodes)
np.save("atlas_lv_tetras.npy", lv_tetras)

np.save("atlas_lv_myo_nodes.npy", lv_myo_nodes)
np.save("atlas_lv_myo_tetras.npy", lv_myo_tetras)

np.save("atlas_rv_nodes.npy", rv_nodes)
np.save("atlas_rv_tetras.npy", rv_tetras)
