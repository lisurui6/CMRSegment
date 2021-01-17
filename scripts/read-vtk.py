import vtk

import sys

import numpy

from CMRSegment.common.constants import LIB_DIR


reader = vtk.vtkGenericDataObjectReader()
reader.SetFileName(str(LIB_DIR.joinpath("data", "1", "output", "vtks", "RV_ED.vtk")))
reader.Update()

polydata = reader.GetOutput()
