import os
import numpy as np
from os.path import splitext
import zipfile
import io
import scipy.io
import cv2
from typing import Any, List
from scipy.spatial.transform import Rotation
from collections.abc import Mapping

from .common import AugmentedSample, FloatArray, UInt8Array


def imdecode(buffer):
    if isinstance(buffer, bytes):
        buffer = np.frombuffer(buffer, dtype='B')
    img = cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert img is not None
    return img


def convert_rotation(pitch, yaw, roll):
    # For AFLW and 300W-LP
    # It's the result of endless trial and error. Don't ask ...
    # Euler angles suck.
    rot: Rotation = Rotation.from_euler('XYZ', [pitch, -yaw, roll])
    M = rot.as_matrix()
    P = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    M = P @ M @ P.T
    rot = Rotation.from_matrix(M)
    return rot


def convert_xys(tx, ty, scale, h, w):
    ty = h - ty
    human_head_radius_micron = 100.0e3
    scale = 0.5 * scale / 224.0 * w * human_head_radius_micron
    xy = np.asarray([tx, ty])
    return xy, scale


def convert_shapeparam(params):
    """Modified for a subset of rescaled shape vectors.
    Also restricted to the first 40 and 10 parameters, respectively."""
    f_shp = params['Shape_Para'][:40, 0] / 20.0 / 1.0e5
    f_exp = params['Exp_Para'][:10, 0] / 5.0
    return np.concatenate([f_shp, f_exp])


def move_aflw_head_center_to_between_eyes(scale, rot, xy):
    offset_my_mangled_shape_data = np.array([0.0, -0.26, -0.9])
    offset = rot.apply(offset_my_mangled_shape_data) * scale
    return xy + offset[:2]


def discover_samples(zf):
    names = frozenset(['AFW', 'HELEN', 'IBUG', 'LFPW'])
    isInDataSubsets = lambda s: s.split(os.path.sep)[1] in names
    filenames = [f.filename for f in zf.filelist if (splitext(f.filename)[1] == '.mat' and isInDataSubsets(f.filename))]
    return filenames


def remove_artificially_rotated_faces(filenames: List[str]):
    return list(filter(lambda fn: fn.endswith('_0.mat'), filenames))


def get_landmarks_filename(matfile: str):
    elements = matfile.split(os.path.sep)
    name = os.path.splitext(elements[-1])[0] + '_pts.mat'
    return os.path.sep.join(elements[:-2] + ['landmarks'] + elements[-2:-1] + [name])


def parse_sample(data: Mapping[str, Any], img: UInt8Array) -> AugmentedSample:
    pitch, yaw, roll, tx, ty, tz, scale = data['Pose_Para'][0]
    h, w, _ = img.shape

    rot = convert_rotation(pitch, yaw, roll)

    xy, scale = convert_xys(tx, ty, scale, h, w)

    xy = move_aflw_head_center_to_between_eyes(scale, rot, xy)

    shapeparams = convert_shapeparam(data)

    # Note: Landmarks in the landmarks folder of 300wlp omit the z-coordinate.
    #       We want them too so the landmarks are reconstructed from the deformable model!
    # pt3d = compute_keypoints(bfm, shapeparams, proj_radius, rot, tx, ty)
    # assert (pt3d.shape == (3,68)), f"Bad shape: {pt3d.shape}"

    # The matlab file contains a bounding box which is however way too big for the image size.

    return AugmentedSample(rot=rot, xy=xy, scale=scale, pt3d_68=None, roi=None, shapeparam=shapeparams, image=img)


class Dataset300WLP(object):
    def __init__(self, filename, only_originals: bool = True):
        self._zf = zf = zipfile.ZipFile(filename)
        matfiles = discover_samples(zf)
        self._matfiles = remove_artificially_rotated_faces(matfiles) if only_originals else matfiles

    @property
    def filenames(self):
        return [(os.path.splitext(matfile)[0]).split('/')[-1] for matfile in self._matfiles]

    def __getitem__(self, i: int) -> tuple[str, AugmentedSample]:
        """Returns the sample and basename."""
        matfile = self._matfiles[i]

        with io.BytesIO(self._zf.read(matfile)) as f:
            data = scipy.io.loadmat(f)

        jpgbuffer = self._zf.read(splitext(matfile)[0] + '.jpg')
        img = imdecode(jpgbuffer)

        # with io.BytesIO(self._zf.read(get_landmarks_filename(matfile))) as f:
        #     landmarkdata = scipy.io.loadmat(f)

        name = (os.path.splitext(matfile)[0]).split('/')[-1]

        sample = parse_sample(data, img)
        return name, sample

    def close(self):
        self._zf.close()

    def __len__(self):
        return len(self._matfiles)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DatasetAFLW2k3D(object):
    def __init__(self, filename):
        self._zf = zf = zipfile.ZipFile(filename)
        self._matfiles = self._discover_samples(zf)

    @staticmethod
    def _discover_samples(zf):
        filenames = [
            f.filename
            for f in zf.filelist
            if splitext(f.filename)[1] == '.mat' and f.filename.startswith('AFLW2000/image')
        ]
        return filenames

    @property
    def filenames(self):
        return [(os.path.splitext(matfile)[0]).split('/')[-1] for matfile in self._matfiles]

    def __getitem__(self, i: int) -> tuple[str, AugmentedSample]:
        matfile = self._matfiles[i]

        with io.BytesIO(self._zf.read(matfile)) as f:
            data = scipy.io.loadmat(f)

        jpgbuffer = self._zf.read(splitext(matfile)[0] + '.jpg')
        img = imdecode(jpgbuffer)

        # with io.BytesIO(self._zf.read(get_landmarks_filename(matfile))) as f:
        #     landmarkdata = scipy.io.loadmat(f)

        name = (os.path.splitext(matfile)[0]).split('/')[-1]

        sample = parse_sample(data, img)
        return name, sample

    def close(self):
        self._zf.close()

    def __len__(self):
        return len(self._matfiles)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
