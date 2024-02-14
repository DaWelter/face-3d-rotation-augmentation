import os
import h5py
from copy import copy
import numpy as np
from PIL import Image
from collections import defaultdict
import contextlib


class FieldCategory(object):
    general = ''
    image = 'img'
    quat = 'q'
    xys = 'xys'
    roi = 'roi'
    points = 'pts'
    semseg = 'seg'


class DatasetWriter(object):
    '''For the extended pose dataset.

    Writes poses, 68 3d landmarks, shape parameters and rois to an hdf5 file.
    Images are stored in a directory which is given the same name as the file.
    The "images" dataset in the h5 provides the image filename per sample.
    '''
    def __init__(self, filename):
        self._filename = filename
        self._imagedir = os.path.splitext(filename)[0]
        self._small_data = defaultdict(list)
        self._counts_by_name = defaultdict(int)
        self._names = dict()
        self.jpgquality = 99

    def close(self):
        if not self._counts_by_name:
            return

        dat = self._small_data
        # Convert to numpy so we can use fancy indexing
        for k, v in dat.items():
            if not isinstance(next(iter(v)), str):
                dat[k] = np.stack(v)

        N = len(next(iter(dat.values())))
        assert all(len(x)==N for x in dat.values())
        cs = min(N, 1024)

        sequence_starts = np.cumsum([0] + list(self._counts_by_name.values()))
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
            f.create_dataset('shapeparams', (N,50), chunks=(cs,50), dtype='f4', data = shapeparam)
            f.create_dataset('sequence_starts', dtype='i4', data = sequence_starts)
            for ds, category in [
                (ds_quats,FieldCategory.quat),
                (ds_coords,FieldCategory.xys),
                (ds_pt3d_68,FieldCategory.points),
                (ds_roi,FieldCategory.roi),
                (ds_img,FieldCategory.image),
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

    def _handle_image(self, name, sample):
        os.makedirs(os.path.dirname(os.path.join(self._imagedir, name)), exist_ok=True)
        i = self._handle_counting(name)
        imagefilename = f"{name}_{i:02d}.jpg"
        Image.fromarray(sample['image']).save(
            os.path.join(self._imagedir, imagefilename), quality=self.jpgquality)
        return imagefilename

    def write(self, name, sample):
        assert (set(sample.keys()) == set(['rot','xy','scale','image','pt3d_68', 'roi', 'shapeparam'])), f"Bad sample {list(sample.keys())}"
        sample = copy(sample)
        sample['image'] = self._handle_image(name, sample)
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