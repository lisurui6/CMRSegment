# ---- voxelmorph ----
# unsupervised learning for image registration

from voxelmorph import generators
from voxelmorph import py
from voxelmorph.py.utils import default_unet_features


# import backend-dependent submodules
backend = py.utils.get_backend()

if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    try:
        import torch
    except ImportError:
        raise ImportError('Please install pytorch to use this voxelmorph backend')

    from voxelmorph import torch
    from voxelmorph.torch import layers
    from voxelmorph.torch import networks
    from voxelmorph.torch import losses

else:
    # tensorflow is default backend
    try:
        import tensorflow
    except ImportError:
        raise ImportError('Please install tensorflow to use this voxelmorph backend')

    from voxelmorph import tf
    from voxelmorph.tf import layers
    from voxelmorph.tf import networks
    from voxelmorph.tf import losses
    from voxelmorph.tf import utils
