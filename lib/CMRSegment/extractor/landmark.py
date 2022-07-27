import numpy as np
import nibabel as nib
from pathlib import Path
from matplotlib import pyplot as plt
from boundary_curvature import curvature
from matplotlib import cm
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import os
import vtk

import shapely
from shapely.geometry import Polygon, LineString
from rasterio import features
from typing import List


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    s = x
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    y = np.concatenate([x[:(window_len - 1)//2], y, x[-(window_len - 1)//2:]])
    return y


def find_offset(shape_curv, debug: bool = False):
    shape_curv = -shape_curv
    if debug:
        plt.plot(np.arange(0, shape_curv.shape[0]), shape_curv, "x-")

    peaks, property = find_peaks(shape_curv, height=0)
    return int(peaks[0])


def find_minima(shape_curv):
    shape_curv = -shape_curv
    peaks, property = find_peaks(shape_curv)
    peak = peaks[0]
    return peak


def extract_lv_landmarks(mid_slice, LV_label) -> List[np.ndarray]:

    mid_slice[mid_slice != LV_label] = 0
    mid_slice = gaussian_filter(mid_slice, 5)
    mid_slice[mid_slice > 0.5] = 1
    mid_slice[mid_slice <= 0.5] = 0
    # plt.figure("extract lv")
    # plt.imshow(mid_slice)
    polygons = []
    for shape, value in features.shapes(mid_slice):
        if value != 0:
            polygons.append(shapely.geometry.shape(shape))
    print(len(polygons))
    for polygon in polygons:
        polygon = np.array(polygon.exterior.coords)
        # plt.scatter(polygon[:, 0], polygon[:, 1], c="r")

    assert len(polygons) == 1
    n = 10
    line = LineString(polygons[0].exterior.coords)
    distances = np.linspace(0, line.length, n)
    points = [line.interpolate(distance) for distance in distances]
    points = [np.array(p) for p in points]
    print(len(polygons))
    return points


def extract_rv_landmarks(mid_slice, RV_label, debug: bool = False):
    mid_slice[mid_slice != RV_label] = 0
    mid_slice = gaussian_filter(mid_slice, 5)
    mid_slice[mid_slice > 0.5] = 1
    mid_slice[mid_slice <= 0.5] = 0
    outputs = curvature.curvature(mid_slice, boundary_point=10)
    shape_xy, shape_curv = outputs[0]
    shape_xy = shape_xy.astype(np.int)

    n_points = shape_curv.shape[0]

    offset = find_offset(smooth(shape_curv, window_len=5), debug=debug)
    shape_curv = np.concatenate([shape_curv[offset:], shape_curv[:offset]])
    shape_xy = np.concatenate([shape_xy[offset:], shape_xy[:offset]])

    peaks, _ = find_peaks(shape_curv)
    max_peak_pos = peaks[np.argmax(shape_curv[peaks])]
    if abs(max_peak_pos) > abs(shape_curv.shape[0] - max_peak_pos):
        # max peak is closer to the end
        reflection = True
    else:
        reflection = False
    if reflection:
        shape_curv = np.flip(shape_curv, axis=0)
        shape_xy = np.flip(shape_xy, axis=0)

    peaks, _ = find_peaks(shape_curv)
    smoothed_curv = smooth(shape_curv, window_len=5)

    if debug:
        plt.figure()
        plt.plot(np.arange(0, n_points), shape_curv, "x-")
        plt.plot(peaks, shape_curv[peaks], "rx")
        plt.figure()
        plt.plot(np.arange(0, n_points), smoothed_curv, "x-")
    peaks, property = find_peaks(smoothed_curv)

    if len(peaks) == 2:
        print("two peaks", peaks)
        minima = find_minima(shape_curv)
        maxima = peaks[-1]
        mid_peak = int((minima + maxima) / 2)
        peaks = peaks.tolist()
        peaks.insert(1, mid_peak)
        peaks = np.array(peaks)

    mid_peak = int((peaks[1] + peaks[2]) / 2)
    peaks = peaks.tolist()
    peaks.append(mid_peak)
    peaks.append(0)
    peaks.append(peaks[0])
    peaks.append(peaks[0])
    peaks = np.array(peaks)
    # else:
    #     raise ValueError(f"Number of peaks found {len(peaks)}")
    # print("peaks", peaks)

    peaks_xy = shape_xy[peaks[:7], :]
    peaks_curv = shape_curv[peaks[:7]]
    if debug:
        plt.plot(peaks, smoothed_curv[peaks], "rx")

        fig = plt.figure("peaks xy")
        plt.imshow(mid_slice)
        plt.scatter(peaks_xy[:, 0], peaks_xy[:, 1], c=peaks_curv, cmap=cm.get_cmap("jet"), s=4)
        plt.scatter(peaks_xy[:, 0], peaks_xy[:, 1], s=10, c="r")
        plt.colorbar()
    lm = []
    for peak_xy in peaks_xy:
        p = np.array(peak_xy)
        lm.append(p)

    return lm


def extract_landmarks(seg, affine, output_path: Path, is_atlas: bool, debug: bool = False):

    RV_label = 3
    LV_label = 1
    z = np.nonzero(seg == RV_label)[2]

    z_min, z_max = z.min(), z.max()
    if not is_atlas:
        z_mids = [int(round(z_min + 0.33 * (z_max - z_min)))]
    else:
        z_mids = [int(round(z_min + 0.66 * (z_max - z_min)))]
    lm = []

    for z_mid in z_mids:
        mid_slice = np.copy(seg[:, :, z_mid])

        print("slice number", z_mid)
        lm = []
        rv_points = extract_rv_landmarks(mid_slice, RV_label, debug)
        rv_points = np.array(rv_points)

        if debug:
            plt.figure("rv points seg")
            plt.imshow(seg[:, :, z_mid])
            plt.scatter(rv_points[:, 0], rv_points[:, 1], s=10, c="r")

        for point in rv_points:
            p = np.dot(affine, np.array([point[1], point[0], z_mid, 1]).reshape((4, 1)))[:3, 0]
            lm.append(p)
        label = 1
        z = np.nonzero(seg == label)[2]
        z_min, z_max = z.min(), z.max()
        seg = np.squeeze(seg)
        if not is_atlas:
            temp = z_min
            z_min = z_max
            z_max = temp
        for z in [z_min, z_max]:
            x, y = [np.mean(i) for i in np.nonzero(seg[:, :, z] == label)]
            p = np.dot(affine, np.array([x, y, z, 1]).reshape((4, 1)))[:3, 0]
            lm.append(p)

        label = 3
        z = np.nonzero(seg == label)[2]
        z_min, z_max = z.min(), z.max()
        seg = np.squeeze(seg)
        if not is_atlas:
            z_min = z_max
        for z in [z_min]:
            x, y = [np.mean(i) for i in np.nonzero(seg[:, :, z] == label)]
            p = np.dot(affine, np.array([x, y, z, 1]).reshape((4, 1)))[:3, 0]
            lm.append(p)

        lv_points = extract_lv_landmarks(seg[:, :, z_mid], LV_label)
        for point in lv_points:
            p = np.dot(affine, np.array([point[1], point[0], z_mid, 1]).reshape((4, 1)))[:3, 0]
            lm.append(p)
        if debug:
            plt.figure("lv points seg")
            plt.imshow(seg[:, :, z_mid])
            lv_points = np.array(lv_points)
            plt.scatter(lv_points[:, 0], lv_points[:, 1], s=10, c="r")
            plt.show()
            plt.close()

    # Write the landmarks
    points = vtk.vtkPoints()
    for p in lm:
        points.InsertNextPoint(p[0], p[1], p[2])
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(poly)
    writer.SetFileName(str(output_path))
    writer.Write()


def extract_subject_landmarks(segmentation_path: Path, output_path: Path, debug: bool = False):
    nim = nib.load(str(segmentation_path))
    affine = nim.affine

    seg = nim.get_data()
    if seg.ndim == 4:
        seg = np.squeeze(seg, axis=-1).astype(np.int16)
    seg = seg.astype(np.float32)
    extract_landmarks(seg, affine, output_path, is_atlas=False, debug=debug)
    return output_path


def extract_atlas_landmarks(RV_segmentation_path, LV_segmentation_path, output_dir: Path):
    nim = nib.load(str(LV_segmentation_path))
    affine = nim.affine

    seg = nim.get_data()
    if seg.ndim == 4:
        seg = np.squeeze(seg, axis=-1).astype(np.int16)
    seg = seg.astype(np.float32)

    nim = nib.load(str(RV_segmentation_path))
    rv_seg = nim.get_data()
    if rv_seg.ndim == 4:
        rv_seg = np.squeeze(rv_seg, axis=-1).astype(np.int16)
    rv_seg = rv_seg.astype(np.float32)
    seg[rv_seg == 255] = 3
    output_path = output_dir.joinpath("landmarks2.vtk")
    extract_landmarks(seg, affine, output_path, True)
