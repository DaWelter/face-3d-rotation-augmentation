from os.path import join, dirname, splitext, isfile
import numpy as np
import h5py
from typing import List, Dict, Optional, Iterator, Sequence, Tuple, NamedTuple, Union, Any, Callable
import enum
from functools import cached_property, lru_cache
from scipy.spatial.transform import Rotation
from PIL import Image

variable_length_hdf5_buffer_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))


class DatasetEncoding(object):
    image_filename = 'image_filename'


class ImagePathDs(object):
    def __init__(self, ds : h5py.Dataset):
        assert ds.attrs['storage'] == DatasetEncoding.image_filename
        self._ds = ds
        self._filelist = ImagePathDs._find_filenames(ds)

    @staticmethod
    def _find_filenames(ds : h5py.Dataset):
        supported_extensions = ('.jpg', '.png', '.jpeg')
        names : Sequence[bytearray] = ds[...]
        first = names[0].decode('ascii')
        extensions_to_try = supported_extensions if (splitext(first.lower())[1] not in supported_extensions) else ('',)
        directories_to_try = [ dirname(ds.file.filename), splitext(ds.file.filename)[0] ]
        found = False
        for root_dir in directories_to_try:
            for ext in extensions_to_try:
                if isfile(join(root_dir, first+ext)):
                    found = True
                    break
        if not found:
            raise RuntimeError(f"Cannot find images for image path dataset. Looking for name {first} with roots {directories_to_try} and extensions {extensions_to_try}")
        return [ join(root_dir,s.decode('ascii')+ext) for s in names ]

    def __getitem__(self, index : int):
        return Image.open(self._filelist[index])

    def __len__(self):
        return len(self._filelist)

    def attrs(self):
        return self._ds.attrs


class DirectNumpyDs(object):
    def __init__(self, ds : h5py.Dataset):
        self._ds = ds

    def __getitem__(self, index : int):
        return self._ds[index][...]

    def __len__(self):
        return len(self._ds)


MaybeWrappedH5Dataset = Union[DirectNumpyDs, ImagePathDs]
Whitelist = List[str]


def open_dataset(g : h5py.Group, name : str) -> MaybeWrappedH5Dataset:
    ds = g[name]
    if not 'storage' in ds.attrs:
        return DirectNumpyDs(ds)
    typeattr = ds.attrs['storage']
    if typeattr == DatasetEncoding.image_filename:
        return ImagePathDs(ds)
    raise RuntimeError(f"Cannot create dataset wrapper. Unknown value of attribute 'storage': {typeattr}")


class Hdf5PoseDataset(object):
    def __init__(self, filename):
        self.filename = filename
        self._h5file = f = h5py.File(self.filename, 'r')
        self._datasets = [
            (k,open_dataset(f,k)) for k, v in f.items() 
        ]
        _, d = next(iter(self._datasets))
        self._frame_count = len(d)

    @property
    def filenames(self):
        return [ splitext(b.decode('ascii'))[0] for b in self._h5file['images'][...] ]

    def close(self):
        if self._h5file is not None:
            self._h5file.close()

    def __enter__(self):
        pass

    def __exit__(self, *argv, **kwargs):
        self.close()

    def __len__(self):
        return self._frame_count

    def __getitem__(self, index):
        if index<0 or index >= len(self):
            raise IndexError(f"Index {index} on dataset of length {len(self)}")
        fields = {
            k:ds[index] for k,ds in self._datasets
        }
        fields['rot'] = Rotation.from_quat(fields.pop('quats'))
        xys = fields.pop('coords')
        fields['xy'] = xys[:2]
        fields['scale'] = xys[2]
        fields['image'] = np.asarray(fields.pop('images'))
        fields['roi'] = fields.pop('rois')
        return fields
