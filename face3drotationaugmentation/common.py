from typing import NamedTuple,Any
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


FloatArray = npt.NDArray[np.floating[Any]]
IntArray = npt.NDArray[np.integer[Any]]
UInt8Array = npt.NDArray[np.uint8]

class AugmentedSample(NamedTuple):
    """A sample with augmented face parameters.

    See readme for details on the coordinate system.
    
    Attributes:
        image: The rendered image with the augmented face. HWC format.
        rot: Rotation.
        xy: Position of the face in the image.
        scale: Scaling of the face model. Approximately the radius of the face in pixels.
        pt3d_68: 3D coordinates of 68 facial landmarks. Shape (68, 3).
        roi: Face bounding box. x0y0x1y1 format.
        shapeparam: Shape parameters. 40 for actual shape. 10 for expression.
    """
    image: UInt8Array
    rot: Rotation
    xy: FloatArray
    scale: float
    pt3d_68: FloatArray | None
    roi: FloatArray | None
    shapeparam: FloatArray


deg2rad = np.pi/180.