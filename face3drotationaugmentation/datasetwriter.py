import os
import h5py
from copy import copy
import numpy as np
from PIL import Image
from collections import defaultdict
from strenum import StrEnum
import contextlib


class FieldCategory(object):
    general = ''
    image = 'img'
    quat = 'q'
    xys = 'xys'
    roi = 'roi'
    points = 'pts'
    landmarks = 'lmk'
    semseg = 'seg'


class DatasetWriter(object):
    def __init__(self, filename):
        self._filename = filename
        self._imagedir = os.path.splitext(filename)[0]
        self._small_data = defaultdict(list)
        self._counts_by_name = defaultdict(int)

    def close(self):
        if not self._counts_by_name:
            return

        dat = self._small_data
        for k, v in dat.items():
            if not isinstance(next(iter(v)), str):
                dat[k] = np.stack(v)

        N = len(next(iter(self._small_data.values())))
        assert all(len(x)==N for x in self._small_data.values())
        cs = min(N, 1024)

        xys = np.concatenate([dat['xy'], dat['scale'][:,None]], axis=-1)
        quats = dat['rot'] # Already converted in .write()
        image = dat['image']
        pt3d_68 = dat['pt3d_68']
        roi = dat['roi']
        shapeparam = dat['shapeparam']

        with h5py.File(self._filename, 'w') as f:
            ds_quats = f.create_dataset('quats', (N,4), chunks=(cs,4), dtype='f4', data = quats)
            ds_coords = f.create_dataset('coords', (N,3), chunks=(cs,3), dtype='f4', data = xys)
            ds_pt3d_68 = f.create_dataset('pt3d_68', (N,68,3), chunks=(cs,68,3), dtype='f4', data = pt3d_68)
            ds_roi = f.create_dataset('rois', (N,4), chunks=(cs,4), dtype='f4', data = roi)
            ds_img = f.create_dataset('images', (N,), chunks=(cs,), data = image)
            ds_img.attrs['storage'] = 'image_filename'
            ds_shapeparams = f.create_dataset('shapeparams', (N,50), chunks=(cs,50), dtype='f4', data = shapeparam)
            for ds, category in [
                (ds_quats,FieldCategory.quat),
                (ds_coords,FieldCategory.xys),
                (ds_pt3d_68,FieldCategory.landmarks),
                (ds_roi,FieldCategory.roi),
                (ds_img,FieldCategory.image),
            ]:
                ds.attrs['category'] = category
    
    def _handle_image(self, sample):
        name = sample['name']
        os.makedirs(os.path.dirname(os.path.join(self._imagedir, name)), exist_ok=True)
        i = self._counts_by_name[name]
        imagefilename = f"{sample['name']}_{i:02d}.jpg"
        self._counts_by_name[name] += 1
        Image.fromarray(sample['image']).save(
            os.path.join(self._imagedir, imagefilename), quality=99)
        return imagefilename

    def write(self, sample):
        assert (set(sample.keys()) == set(['rot','xy','scale','image','name','pt3d_68', 'roi', 'shapeparam']))
        sample = copy(sample)
        sample['image'] = self._handle_image(sample)
        del sample['name']
        sample['rot'] = sample['rot'].as_quat()
        for k, v in sample.items():
            self._small_data[k].append(v)


@contextlib.contextmanager
def dataset_writer(filename):
    writer = DatasetWriter(filename)
    try:
        yield writer
    finally:
        writer.close()