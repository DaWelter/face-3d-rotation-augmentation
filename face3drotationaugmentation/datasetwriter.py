from abc import abstractmethod
from collections.abc import Generator
import os
from pathlib import Path
from typing import Any, Literal
import h5py
from copy import copy
import numpy as np
from PIL import Image
from collections import defaultdict
import contextlib
from scipy.spatial.transform import Rotation
import scipy.io

from .common import AugmentedSample, FloatArray, UInt8Array


class FieldCategory(object):
    general = ''
    image = 'img'
    quat = 'q'
    xys = 'xys'
    roi = 'roi'
    points = 'pts'
    semseg = 'seg'


class DatasetWriter:
    @abstractmethod
    def close(self):
        """Write all data and clean up resources."""
        ...

    @abstractmethod
    def write(self, name: str, sample: AugmentedSample):
        """Add a sample.

        Adding the same name multiple times is intended. A counter will be added.
        """
        ...


class DatasetWriterCustomHdf5Format(DatasetWriter):
    '''For the extended pose dataset.

    Writes poses, 68 3d landmarks, shape parameters and rois to an hdf5 file.
    Images are stored in a directory which is given the same name as the file.
    The "images" dataset in the h5 provides the image filename per sample.
    '''

    def __init__(self, filename: str):
        if not (filename.lower().endswith('.h5') or filename.lower().endswith('.hdf5')):
            raise ValueError("outputfilename must have hdf5 filename extension")
        self._filename = filename
        self._imagedir = os.path.splitext(filename)[0]
        self._samples_by_field = defaultdict(list)
        self._counts_by_name = defaultdict(int)
        self._names = dict()
        self.jpgquality = 99

    def close(self):
        if not self._counts_by_name:
            return

        dat = self._samples_by_field
        # Convert to numpy so we can use fancy indexing
        for k, v in dat.items():
            if not isinstance(next(iter(v)), str):
                dat[k] = np.stack(v)

        N = len(next(iter(dat.values())))
        assert all(len(x) == N for x in dat.values())
        cs = min(N, 1024)

        sequence_starts = np.cumsum([0] + list(self._counts_by_name.values()))
        xys = np.concatenate([dat['xy'], dat['scale'][:, None]], axis=-1)
        quats = dat['rot']  # Already converted in .write()
        image = dat['image']
        pt3d_68 = dat['pt3d_68']
        roi = dat['roi']
        shapeparam = dat['shapeparam']

        with h5py.File(self._filename, 'w') as f:
            ds_quats = f.create_dataset('quats', (N, 4), chunks=(cs, 4), dtype='f4', data=quats)
            ds_coords = f.create_dataset('coords', (N, 3), chunks=(cs, 3), dtype='f4', data=xys)
            ds_pt3d_68 = f.create_dataset('pt3d_68', (N, 68, 3), chunks=(cs, 68, 3), dtype='f4', data=pt3d_68)
            ds_roi = f.create_dataset('rois', (N, 4), chunks=(cs, 4), dtype='f4', data=roi)
            ds_img = f.create_dataset('images', (N,), chunks=(cs,), data=image)
            ds_img.attrs['storage'] = 'image_filename'
            f.create_dataset('shapeparams', (N, 50), chunks=(cs, 50), dtype='f4', data=shapeparam)
            f.create_dataset('sequence_starts', dtype='i4', data=sequence_starts)
            for ds, category in [
                (ds_quats, FieldCategory.quat),
                (ds_coords, FieldCategory.xys),
                (ds_pt3d_68, FieldCategory.points),
                (ds_roi, FieldCategory.roi),
                (ds_img, FieldCategory.image),
            ]:
                ds.attrs['category'] = category

    def _handle_counting(self, name):
        num = self._counts_by_name[name]
        self._counts_by_name[name] += 1
        # Check if not repeating earlier name
        if self._names:
            k, _ = self._names.popitem()
            assert not name in self._names
            self._names[k] = None
            self._names[name] = None
        return num

    def _handle_image(self, name: str, sample: AugmentedSample):
        os.makedirs(os.path.dirname(os.path.join(self._imagedir, name)), exist_ok=True)
        i = self._handle_counting(name)
        imagefilename = f"{name}_{i:02d}.jpg"
        Image.fromarray(sample.image).save(os.path.join(self._imagedir, imagefilename), quality=self.jpgquality)
        return imagefilename

    def write(self, name: str, sample: AugmentedSample):
        assert sample.roi is not None
        assert sample.pt3d_68 is not None
        data = sample._asdict()
        data['image'] = self._handle_image(name, sample)
        data['rot'] = data['rot'].as_quat()
        for k, v in data.items():
            self._samples_by_field[k].append(v)


class DatasetWriter300WLPLike(DatasetWriter):
    def __init__(self, directory):
        self.directory = Path(directory)
        self._counts_by_name = defaultdict(int)
        self.jpgquality = 99
        self._have_dir = False

    def close(self):
        pass

    def _convert_sample(self, file, sample: AugmentedSample):
        human_head_radius_micron = 100.0e3
        h, w, _ = sample.image.shape
        scale = sample.scale / human_head_radius_micron / w * 224.0 / 0.5
        xy = move_head_center_back(sample.scale, sample.rot, sample.xy)
        tx = xy[0]
        ty = h - xy[1]
        tz = 0.0
        pitch, yaw, roll = inv_aflw_rotation_conversion(sample.rot)
        mat_dict = {
            # TODO: maybe pad to full size?
            'Shape_Para': np.pad(sample.shapeparam[:40, None] * 20.0 * 1.0e5, [(0, 199 - 40), [0, 0]]),
            'Exp_Para': np.pad(sample.shapeparam[40:, None] * 5.0, [(0, 29 - 10), (0, 0)]),
            'Pose_Para': [[pitch, yaw, roll, tx, ty, tz, scale]],
            #'pt3d_68' : (sample.pt3d_68 * np.asarray([1.,1.,-1]) ).T  # output shape (3,68) Not sure if correct
        }
        scipy.io.savemat(file, mat_dict)

    def write(self, name: str, sample: AugmentedSample):
        # Note: *_0.jpg would be the original image
        if not self._have_dir:
            self.directory.mkdir()
            self._have_dir = True
        number = self._counts_by_name[name]
        self._counts_by_name[name] += 1
        filename = self.directory / f"{name}_{number}"
        self._convert_sample(filename.with_suffix(".mat"), sample)
        Image.fromarray(sample.image).save(filename.with_suffix(".jpg"), quality=self.jpgquality)


def inv_aflw_rotation_conversion(rot: Rotation):
    '''Rotation object to Euler angles for AFLW and 300W-LP data

    Returns:
        Batch x (Pitch,Yaw,Roll)
    '''
    P = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    M = P @ rot.as_matrix() @ P.T
    rot = Rotation.from_matrix(M)
    euler = rot.as_euler('XYZ')
    euler *= np.asarray([1, -1, 1])
    return euler


def move_head_center_back(scale, rot, xy):
    local_offset = np.array([0.0, -0.26, -0.9])
    offset = rot.apply(local_offset) * scale
    return xy - offset[:2]


OutputFormats = Literal['custom_hdf5', '300wlp']


@contextlib.contextmanager
def dataset_writer(filename, format: OutputFormats) -> Generator[DatasetWriter, Any, None]:
    if format == 'custom_hdf5':
        writer = DatasetWriterCustomHdf5Format(filename)
    else:
        writer = DatasetWriter300WLPLike(filename)
    try:
        yield writer
    finally:
        writer.close()
