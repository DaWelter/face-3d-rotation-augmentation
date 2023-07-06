from typing import Tuple, NamedTuple, Union, Optional
import contextlib
import os
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
import pickle
from PIL import Image
import functools
import tqdm
import trimesh
import pyrender
import torch

from . import facemodel
from .facemodel import bfm

FloatArray = npt.NDArray[Union[np.float32,np.float64]]
IntArray = npt.NDArray[Union[np.int32,np.int64]]


def get_hpb(rot : Rotation):
    '''Conversion to heading-pitch-bank.
    
    First rotation is around z (bank/roll), then x (pitch), then y (yaw/heading)
    '''
    return rot.as_euler('YXZ')


def make_rot_by_axis_rotations(hpb):
    '''For testing - to ensure we have the right convention.'''
    h, p, b = hpb.T
    z = np.zeros(h.shape)
    return Rotation.from_rotvec(np.vstack([z,h,z]).T) * Rotation.from_rotvec(np.vstack([p,z,z]).T) * Rotation.from_rotvec(np.vstack([z,z,b]).T)


def make_rot(hpb):
    '''Conversion from heading-pitch-bank.
    
    First rotation is around z (bank/roll), then x (pitch), then y (yaw/heading)
    '''
    return Rotation.from_euler('YXZ', hpb)


def affine3d_chain(Ta, Tb):
    Ra, ta = Ta
    Rb, tb = Tb
    return Ra*Rb, Ra.apply(tb) + ta


def affine3d_inv(Ta):
    Ra, ta = Ta
    RaInv = Ra.inv()
    return RaInv, -RaInv.apply(ta)


def apply_affine3d(tr, vertices):
    R, t = tr
    return R.apply(vertices) + t


def apply_s_rot_t(vertices, xy, scale, rot):
    vertices = rot.apply(vertices * scale)
    vertices[...,:2] += xy
    return vertices


def interpolate_images(values, sample_points):
    '''
    values: Image like with shape (...,C,H,W)
    sample_points: with shape (...,N,2)
    output: values at sample points with shape (...,N,C)
    '''
    C, H, W = values.shape[-3:]
    N = sample_points.size(-2)
    prefix_shape = values.shape[:-3]
    assert sample_points.shape[-2:] == (N,2)
    values = values.view(-1,C,H,W)
    sample_points = sample_points * torch.tensor([[2./W,2./H]]) - 1.
    sample_points = sample_points.view(-1,N,1,2)
    samples = torch.nn.functional.grid_sample(values, sample_points, align_corners=False, mode='bilinear', padding_mode='border')
    samples = samples.view(*prefix_shape,C,N).swapaxes(-2,-1)
    return samples


def interpolate_zero_channel_numpy_image(image, sample_points):
    return interpolate_images(torch.from_numpy(image[None,:,:]).to(torch.float32), torch.from_numpy(sample_points).to(torch.float32))[:,0].numpy()


def apply_blendshapes(vertices, blend_shape_vertices, shapeparam):
    return vertices + np.sum((blend_shape_vertices * shapeparam[:,None,None]), axis=0)



class Meshdata(NamedTuple):
    vertices : FloatArray
    tris : IntArray
    normals : FloatArray
    vertex_weights : FloatArray
    uvs : Optional[FloatArray]


class Faceparams(NamedTuple):
    xy : FloatArray
    scale : float
    rot : Rotation
    shapeparam : Optional[FloatArray]

    
def load_base_mesh(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        tris = np.asarray(data['tris'])
        base_vertices = np.asarray(data['vertices'])
        base_vertices *= 0.01  # Vertices were changed during import in 3d software
        base_vertices[:,1] *= -1 # Vertices were changed during import in 3d software
        #base_vertices[:,2] *= -1
        blend_weights = np.asarray(data['weights'])
        # Dummy is good enough for unlit scene
        normals = np.broadcast_to(np.asarray([[ 0., 0., 1.]]), (len(base_vertices),3))
        return Meshdata(base_vertices, tris, normals, blend_weights, None)


def create_pyrender_material(original_image, texture_border):
    '''
    Creates the material for the face mesh. Takes the original image of the data sample.
    Before creating the texture a border / padding is added on all sides. This is needed
    to pad the output image in black rather than with parts of the original image.
    '''
    h, w = original_image.shape[:2]
    tex = np.zeros((h+2*texture_border,w+2*texture_border,4), dtype=np.uint8)
    tex[texture_border:-texture_border, texture_border:-texture_border,:3] = original_image
    tex[:,:,3] = 255 # Alpha channel

    tex = pyrender.Texture(
        source = tex,
        source_channels = 'RGBA', # RGB doesn't work
        data_format = pyrender.texture.GL_UNSIGNED_BYTE,
        sampler = pyrender.Sampler(
            wrapS=pyrender.GLTF.CLAMP_TO_EDGE, 
            wrapT=pyrender.GLTF.CLAMP_TO_EDGE)
    )

    return pyrender.MetallicRoughnessMaterial(baseColorTexture=tex, doubleSided=False)


def compute_initial_posed_vertices(meshdata : Meshdata, bfm : bfm.BFMModel, faceparams : Faceparams):
    vertices = meshdata.vertices.copy()
    vertices[:bfm.vertexcount] = apply_blendshapes(vertices[:bfm.vertexcount], bfm.scaled_bases, faceparams.shapeparam)
    unrotated_vertices = apply_s_rot_t(vertices, faceparams.xy, faceparams.scale, Rotation.identity())
    vertices = apply_s_rot_t(vertices, faceparams.xy, faceparams.scale, faceparams.rot)
    f = meshdata.vertex_weights
    vertices = vertices*f[:,None] + (1.-f)[:,None]*unrotated_vertices
    return vertices


def re_pose(meshdata : Meshdata, bfm : bfm.BFMModel, original_faceparams : Faceparams, rot_offset, rot_offset_center, new_shapeparam):
    # First reverse the original pose transform, ignoring scale
    vertices = meshdata.vertices.copy()
    vertices[:,:2] -= original_faceparams.xy
    vertices = original_faceparams.rot.inv().apply(vertices)
    # Then pose the vertices according to the offset pose
    scale = original_faceparams.scale
    rot_offset_center = scale*rot_offset_center
    vertices[:bfm.vertexcount] = apply_blendshapes(vertices[:bfm.vertexcount], scale*bfm.scaled_bases, new_shapeparam - original_faceparams.shapeparam)
    vertices = rot_offset.apply(vertices - rot_offset_center) + rot_offset_center
    vertices = apply_s_rot_t(vertices, original_faceparams.xy, 1., original_faceparams.rot)
    f = meshdata.vertex_weights
    # Blend with the input vertices
    vertices = vertices*f[:,None] + (1.-f)[:,None]*meshdata.vertices
    return vertices


class FaceWithBackgroundModel(object):
    texture_border = 50 # pixels

    def __init__(self, meshdata : Meshdata, bfm : bfm.BFMModel, xy, scale, rot, shapeparam, image):
        self._meshdata = meshdata
        self._bfm = bfm
        self._rot = rot
        self._xy = xy
        self._scale = scale
        self._shapeparam = shapeparam
        self._faceparams = Faceparams(xy, scale, rot, shapeparam)
        # Center for the augmented rotation. It is specified in the local BFM frame.
        self._rotation_center = np.array([0., 0.5, 0.1])

        h, w = image.shape[:2]

        # Extra border to generate black pixels outside of the original image region.
        # Because the the deformation will pull parts of the mesh into the view which
        # were not part of the original image.
        
        vertices_according_to_pose = compute_initial_posed_vertices(meshdata, bfm, self._faceparams)

        self._laplacian = self._compute_laplacian(vertices_according_to_pose)
        
        vertices_according_to_pose = self._apply_smoothing(vertices_according_to_pose)

        # By default, without texture border, the range from [0,w] is mapped to [0,1]
        # With border ...
        #    Range from -border to w+border is mapped to [0,1]
        uvs = (vertices_according_to_pose[:,:2] + self.texture_border) / (np.asarray([[w, h]]) + 2*self.texture_border)
        uvs[:,1] = 1. - uvs[:,1]
        self._uvs = uvs

        self.background_plane_z_coord = np.average(vertices_according_to_pose[self._meshdata.vertex_weights < 0.01,2])

        self._meshdata = self._meshdata._replace(vertices = vertices_according_to_pose)


    def set_non_face_by_depth_estimate(self, inverse_depth):
        vertices = self._meshdata.vertices
        keypoints = vertices[self._bfm.keypoints,:]
        depth = -inverse_depth
        depth_estimate_zs = interpolate_zero_channel_numpy_image(depth, keypoints[:,:2])
        calibration_offset = np.average(keypoints[:,2] - depth_estimate_zs)
        depth = depth + calibration_offset
        depth = np.clip(depth, -1.5*self._faceparams.scale, 1.5*self._faceparams.scale)
        depth_estimate_zs += calibration_offset
        z_calibration_curves = (keypoints[:,2], depth_estimate_zs)
        zs = interpolate_zero_channel_numpy_image(depth, vertices[:,:2])
        vertices = vertices.copy()
        f = np.power(self._meshdata.vertex_weights, 2.)
        vertices[:,2] = vertices[:,2]*f + (1.-f)*zs
        self._meshdata = self._meshdata._replace(vertices = vertices)
        return z_calibration_curves


    def _compute_laplacian(self, vertices):
        # Precompute Laplacian for smoothing
        mask = np.ones((vertices.shape[0],), dtype=np.bool8)
        mask[:self._bfm.vertexcount] = False
        mask = np.logical_and(mask, self._meshdata.vertex_weights > 0.01)
        mesh = trimesh.Trimesh(vertices = vertices, faces = self._meshdata.tris)
        pinned, = np.nonzero(~mask)
        return trimesh.smoothing.laplacian_calculation(mesh, pinned_vertices=pinned)


    def _apply_smoothing(self, vertices):
        mesh = trimesh.Trimesh(vertices = vertices, faces = self._meshdata.tris)
        trimesh.smoothing.filter_laplacian(mesh, lamb = 0.2, iterations=2, implicit_time_integration=True, volume_constraint=False, laplacian_operator=self._laplacian)
        return mesh.vertices


    # def _compute_face_vertices_with_background(self, rotoffset, shapeparam):
    #     vertices = self._base_vertices.copy()
    #     vertices[:self._bfm.vertexcount] = apply_blendshapes(self._base_vertices[:self._bfm.vertexcount], self._bfm.scaled_bases, shapeparam)
    #     s, R, t = self._compute_combined_transform(rotoffset)
    #     vertices1 = apply_s_rot_t(vertices, self._xy, self._scale, Rotation.identity())
    #     vertices2 = apply_s_rot_t(vertices, t[:2], s, R)
    #     f = self._blend_weights
    #     final_vertices = vertices2*f[:,None] + (1.-f)[:,None]*vertices1
    #     return final_vertices, (R, t)


    def _compute_combined_transform(self, rotoffset):
        # The new rotation shall take place about self._rotation_center
        # 1. Transform center to world space: (center' = Tr @ center)
        # 2. transform verts according to sample parameters: (v = (Tr @ shaping_result)
        # 3. rotate verts by augmented rotation offset around center v' = (Tr2 @ (v-center)) + center
        # 4. blend: v*f + (1-f)*shaping_result
        # However, the rigid transformations can be rewritten as
        # v' = (Tr2 @ ((Tr @ s) - center) + center
        #    = Tr2' @ Tr' @ s
        # With Tr2' like Tr2 with translation + center
        # and  Tr' like Tr with translation - center
        # Then Tr2 and Tr can be combined together
        center = apply_s_rot_t(self._rotation_center, self._xy, self._scale, self._rot)
        T1_t = np.asarray([self._xy[0], self._xy[1], 0.]) - center
        T1_rot = self._rot
        T2_t = center
        T2_rot = rotoffset
        R, t = affine3d_chain((T2_rot,T2_t),(T1_rot,T1_t))
        return (R, t)


    def __call__(self, rotoffset = None, shapeparam = None) -> Tuple[Meshdata,Tuple[Rotation,np.ndarray]]:
        if rotoffset is None:
            rotoffset = Rotation.identity()
        if shapeparam is None:
            shapeparam = self._shapeparam
        vertices = re_pose(self._meshdata, self._bfm, self._faceparams, rotoffset, self._rotation_center, shapeparam)
        vertices = self._apply_smoothing(vertices)
        tr = self._compute_combined_transform(rotoffset)
        return (Meshdata(
            vertices,
            self._meshdata.tris,
            self._meshdata.normals,
            None,
            self._uvs
        ), tr)


class FaceAugmentationScene(object):
    def __init__(self, sample):
        xy = sample['xy']
        scale = sample['scale']
        shapeparam = sample['shapeparam']
        image = sample['image']
        rot = sample['rot']
        h, w, _ = image.shape
        meshdata, bfm = FaceAugmentationScene.load_assets()
        self.face_model = face_model = FaceWithBackgroundModel(meshdata, bfm, xy, scale, rot, shapeparam, image)
        self.scene = scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0.0, 0.0, 0.0])
        self.material = create_pyrender_material(image, FaceWithBackgroundModel.texture_border)
        self.keypoint_indices = bfm.keypoints
        FaceAugmentationScene.add_camera(scene, image.shape, scale, face_model.background_plane_z_coord)


    @contextlib.contextmanager
    def __call__(self, rotoffset = None, shapeparam = None):
        meshdata, tr = self.face_model(rotoffset, shapeparam)
        prim = pyrender.Primitive(positions = meshdata.vertices, indices=meshdata.tris, texcoord_0 = meshdata.uvs, normals=meshdata.normals, material=self.material)
        mesh = pyrender.Mesh(primitives = [prim])
        face_node = self.scene.add(mesh)
        try:
            keypoints = meshdata.vertices[self.keypoint_indices]
            yield (self.scene, tr, keypoints)
        finally:
            self.scene.remove_node(face_node)


    @staticmethod
    def add_camera(scene, image_shape, scale, background_plane_z_coord):
        h, w, _ = image_shape
        zdistance = 10000
        fov = 2.*np.arctan(0.5*(h)/(zdistance + background_plane_z_coord))
        cam = pyrender.PerspectiveCamera(yfov=fov, znear = zdistance-scale*2, zfar = zdistance+scale*2)
        campose = np.eye(4)
        campose[:3,3] = [ w//2, h//2, -zdistance  ]
        campose[:3,:3] = [
            [ 1, 0, 0 ],
            [ 0, 0, -1 ],
            [ 0, -1, 0 ]
        ]
        scene.add(cam, pose=campose)


    @staticmethod
    @functools.cache
    def load_assets():
        this_file_directory = os.path.dirname(__file__)
        base_mesh = load_base_mesh(os.path.join(this_file_directory,"full_bfm_mesh_with_bg_v3.pkl"))
        headmodel = bfm.BFMModel(40, 10)
        return base_mesh, headmodel


def test_euler_angle_functions():
    import numpy.testing
    ref_rots = Rotation.random(num=1000)
    hpb = get_hpb(ref_rots)
    rots = make_rot(hpb)
    numpy.testing.assert_array_less((rots.inv() * ref_rots).magnitude(), 1.e-6)
    numpy.testing.assert_array_less((rots.inv() * make_rot_by_axis_rotations(hpb)).magnitude(), 1.e-6)

test_euler_angle_functions()