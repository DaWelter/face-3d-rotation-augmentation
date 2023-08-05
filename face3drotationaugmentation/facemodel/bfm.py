from typing import NamedTuple
import numpy as np
import numpy.typing as npt
import pickle
from os.path import join, dirname


def _load(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr

_current_folder = dirname(__file__)


class IntermediateBFMModel(object):
    def __init__(self, shape_dim=40, exp_dim=10):
        bfm = _load(join(_current_folder,'bfm_noneck_v3.pkl'))
        self.u = bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]
        self.tri = _load(join(_current_folder,'tri.pkl'))  # this tri/face is re-built for bfm_noneck_v3
        self.vertexcount = self.u.shape[0]//3
        
        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = bfm.get('keypoints').astype(np.int64)[::3]//3
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)
    
    @property
    def scaled_shp_base(self):
        w_shp = 20.*self.w_shp.reshape((self.vertexcount,3,-1))
        w_shp = w_shp.transpose([2,0,1])
        w_shp *= np.array([[[1.,-1.,-1.]]])
        return w_shp
        
    @property
    def scaled_exp_base(self):
        w_exp = 5.e-5*self.w_exp.reshape((self.vertexcount,3,-1))
        w_exp = w_exp.transpose([2,0,1])
        w_exp *= np.array([[[1.,-1.,-1.]]])
        return w_exp
    
    @property
    def scaled_bases(self):
        '''num eigvecs, num vertices, 3'''
        return np.concatenate([self.scaled_shp_base, self.scaled_exp_base], axis=0)
    
    @property
    def scaled_vertices(self):
        actualcenter = np.array([0., -0.26, -0.9], dtype='f4')
        vertices = self.u.reshape((-1,3))*1.e-5*np.array([[1.,-1.,-1.]],dtype='f4')
        vertices -= actualcenter[None,:]
        return np.ascontiguousarray(vertices)
    
    @property
    def scaled_tri(self):
        tri = self.tri
        tri = tri[...,[2,1,0]]
        return np.ascontiguousarray(tri)
    

class BFMModel(NamedTuple):
    tri : npt.NDArray[np.int32]
    scaled_vertices : npt.NDArray[np.float32]
    scaled_bases : npt.NDArray[np.float32]
    keypoints : npt.NDArray[np.int64]
    
    @property
    def vertexcount(self):
        return len(self.scaled_vertices)
    
    @staticmethod
    def load() -> 'BFMModel':
        m = IntermediateBFMModel()
        return BFMModel(
            m.scaled_tri,
            m.scaled_vertices,
            m.scaled_bases,
            m.keypoints
        )