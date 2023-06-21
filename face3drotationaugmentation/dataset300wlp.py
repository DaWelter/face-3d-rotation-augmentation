import os
import numpy as np
from os.path import splitext
import zipfile
import io
import scipy.io
import cv2
from typing import List
from scipy.spatial.transform import Rotation

from face3drotationaugmentation.facemodel import bfm


def imdecode(buffer):
    if isinstance(buffer, bytes):
        buffer = np.frombuffer(buffer, dtype='B')
    img = cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert img is not None
    return img


def aflw_rotation_conversion(pitch, yaw, roll):
    # For AFLW and 300W-LP
    # It's the result of endless trial and error. Don't ask ...
    # Euler angles suck.
    rot : Rotation = Rotation.from_euler('XYZ', [pitch,-yaw,roll])
    M = rot.as_matrix()
    P = np.asarray([
        [ 1, 0, 0 ],
        [ 0, 1, 0 ],
        [ 0, 0, -1 ]
    ])
    M = P @ M @ P.T
    rot = Rotation.from_matrix(M)
    return rot


def get_3ddfa_shape_parameters(params):
    """ Modified for a subset of rescaled shape vectors. 
        Also restricted to the first 40 and 10 parameters, respectively."""
    f_shp = params['Shape_Para'][:40,0]/20./1.e5
    f_exp = params['Exp_Para'][:10,0]/5.
    return np.concatenate([f_shp, f_exp])


def compute_keypoints(bfm : bfm.BFMModel, shapeparams, head_size, rotation, tx, ty):
    idx = bfm.keypoints
    pts3d = bfm.scaled_vertices[idx] + np.sum(shapeparams[:,None,None] * bfm.scaled_bases[:,idx,:], axis=0)
    pts3d *= head_size     
    pts3d = rotation.apply(pts3d)
    pts3d = pts3d.T
    pts3d[0] += tx
    pts3d[1] += ty
    return pts3d


def move_aflw_head_center_to_between_eyes(scale, rot, xy):
    offset_my_mangled_shape_data = np.array([0., -0.26, -0.9])
    offset = rot.apply(offset_my_mangled_shape_data)*scale
    return xy + offset[:2]


def discover_samples(zf):
    names = frozenset(['AFW', 'HELEN', 'IBUG', 'LFPW'])
    isInDataSubsets = lambda s: s.split(os.path.sep)[1] in names
    filenames = [
        f.filename for f in zf.filelist if 
        (f.external_attr==0x20 and splitext(f.filename)[1]=='.mat' and isInDataSubsets(f.filename)) ]
    return filenames


def remove_artificially_rotated_faces(filenames : List[str]):
    return list(filter(lambda fn: fn.endswith('_0.mat'), filenames))


def get_landmarks_filename(matfile : str):
    elements = matfile.split(os.path.sep)
    name = os.path.splitext(elements[-1])[0]+'_pts.mat'
    return os.path.sep.join(elements[:-2]+['landmarks']+elements[-2:-1]+[name])


def parse_sample(data, img):
    pitch, yaw, roll, tx, ty, tz, scale = data['Pose_Para'][0]
    rot = aflw_rotation_conversion(pitch, yaw,roll)

    h, w, _ = img.shape
    ty = h - ty
    human_head_radius_micron = 100.e3
    scale = 0.5*scale / 224. * w * human_head_radius_micron
    xy = np.asarray([ tx, ty ])

    xy = move_aflw_head_center_to_between_eyes(scale, rot, xy)

    shapeparams = get_3ddfa_shape_parameters(data)

    # Note: Landmarks in the landmarks folder of 300wlp omit the z-coordinate.
    #       We want them too so the landmarks are reconstructed from the deformable model!
    # pt3d = compute_keypoints(bfm, shapeparams, proj_radius, rot, tx, ty)
    # assert (pt3d.shape == (3,68)), f"Bad shape: {pt3d.shape}"

    # The matlab file contains a bounding box which is however way too big for the image size.
    # x0, y0, _ = np.amin(pt3d, axis=1)
    # x1, y1, _ = np.amax(pt3d, axis=1)
    # roi = np.array([x0, y0, x1, y1])

    return { 
        'rot' :  rot,
        'xy' : xy,
        'scale' : scale,
        'shapeparam' : shapeparams
    }


class Dataset300WOriginals(object):
    def __init__(self, filename):
        self._zf = zf = zipfile.ZipFile(filename)
        matfiles = discover_samples(zf)
        self._matfiles = remove_artificially_rotated_faces(matfiles)
        self._bfm = bfm.BFMModel()

    def __getitem__(self, i):
        matfile = self._matfiles[i]

        with io.BytesIO(self._zf.read(matfile)) as f:
            data = scipy.io.loadmat(f)

        jpgbuffer = self._zf.read(splitext(matfile)[0]+'.jpg')
        img = imdecode(jpgbuffer)
        
        # with io.BytesIO(self._zf.read(get_landmarks_filename(matfile))) as f:
        #     landmarkdata = scipy.io.loadmat(f)

        sample = parse_sample(data, img)
        sample.update({
            'image' : img,
            #'pt2d_68' : landmarkdata['pts_2d'],
        })
        return sample

    def close(self):
        self._zf.close()