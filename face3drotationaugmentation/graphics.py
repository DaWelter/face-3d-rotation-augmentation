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


def sigmoid(x):
    return 1./(1. + np.exp(-x))


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


def find_closest_points(reference_points, distanced_points):
    cross_distances = np.linalg.norm(reference_points[None,:,:]-distanced_points[:,None,:],axis=-1)
    idx_closest = np.argmin(cross_distances, axis=1)
    distances, = np.take_along_axis(cross_distances, idx_closest[:,None], axis=1).T
    return idx_closest, distances


def test_find_closest_points():
    a, b = find_closest_points(np.asarray([[0.,10.,0.],[0.,0.,0.],[0.,0.,2.]]), np.asarray([[0.,1.,0.],[0.,0.,2.]]))
    np.testing.assert_array_equal(a, [1, 2])
    np.testing.assert_allclose(b, [1., 0.])

test_find_closest_points()


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
    color : Optional[FloatArray]
    deformbasis : Optional[FloatArray] # (Bases x Points x 3)

    @property
    def num_vertices(self):
        return self.vertices.shape[0]

    @property
    def num_tris(self):
        return self.tris.shape[0]


class Faceparams(NamedTuple):
    xy : FloatArray
    scale : float
    rot : Rotation
    shapeparam : Optional[FloatArray]


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


def compute_initial_posed_vertices(meshdata : Meshdata, faceparams : Faceparams):
    vertices = apply_blendshapes(meshdata.vertices, meshdata.deformbasis, faceparams.shapeparam)
    unrotated_vertices = apply_s_rot_t(vertices, faceparams.xy, faceparams.scale, Rotation.identity())
    vertices = apply_s_rot_t(vertices, faceparams.xy, faceparams.scale, faceparams.rot)
    f = meshdata.vertex_weights
    vertices = vertices*f[:,None] + (1.-f)[:,None]*unrotated_vertices
    return vertices


def re_pose(meshdata : Meshdata, original_faceparams : Faceparams, rot_offset, rot_offset_center, new_shapeparam):
    # First reverse the original pose transform, ignoring scale
    vertices = meshdata.vertices.copy()
    vertices[:,:2] -= original_faceparams.xy
    vertices = original_faceparams.rot.inv().apply(vertices)
    # Then pose the vertices according to the offset pose
    scale = original_faceparams.scale
    rot_offset_center = scale*rot_offset_center
    vertices = apply_blendshapes(vertices, scale*meshdata.deformbasis, new_shapeparam - original_faceparams.shapeparam)
    vertices = rot_offset.apply(vertices - rot_offset_center) + rot_offset_center
    vertices = apply_s_rot_t(vertices, original_faceparams.xy, 1., original_faceparams.rot)
    f = meshdata.vertex_weights
    # Blend with the input vertices
    vertices = vertices*f[:,None] + (1.-f)[:,None]*meshdata.vertices
    return vertices


class FaceWithBackgroundModel(object):
    texture_border = 50 # pixels

    def __init__(self, meshdata : Meshdata, xy, scale, rot, shapeparam, image):
        self._keypoints = bfm.BFMModel().keypoints # TODO: restructure code to remove this
        self._meshdata = meshdata
        self._rot = rot
        self._xy = xy
        self._scale = scale
        self._shapeparam = shapeparam
        self._faceparams = Faceparams(xy, scale, rot, shapeparam)
        # Center for the augmented rotation. It is specified in the local BFM frame.
        self._rotation_center = np.array([0., 0.5, 0.1])

        h, w = image.shape[:2]

        self._dynamic_weight_parameters = self._compute_distances_from_face_and_not_face_indices(self._meshdata)

        self._unposed_vertices = self._meshdata.vertices

        self._meshdata = self._compute_weights_dynamically(rot)

        # Extra border to generate black pixels outside of the original image region.
        # Because the the deformation will pull parts of the mesh into the view which
        # were not part of the original image.
        
        vertices_according_to_pose = compute_initial_posed_vertices(self._meshdata, self._faceparams)

        #self._laplacian = self._compute_laplacian(vertices_according_to_pose)
        
        #vertices_according_to_pose = self._apply_smoothing(vertices_according_to_pose)

        # By default, without texture border, the range from [0,w] is mapped to [0,1]
        # With border ...
        #    Range from -border to w+border is mapped to [0,1]
        uvs = (vertices_according_to_pose[:,:2] + self.texture_border) / (np.asarray([[w, h]]) + 2*self.texture_border)
        uvs[:,1] = 1. - uvs[:,1]
        self._uvs = uvs

        self.background_plane_z_coord = np.average(vertices_according_to_pose[self._meshdata.vertex_weights < 0.01,2])

        self._meshdata = self._meshdata._replace(vertices = vertices_according_to_pose)


    def _compute_distances_from_face_and_not_face_indices(self, meshdata : Meshdata):
        rng = np.random.RandomState(seed=123456)
        face_vertex_mask = meshdata.vertex_weights > 0.99
        face_indices, = np.nonzero(face_vertex_mask)
        notface_indices, = np.nonzero(~face_vertex_mask)
        face_indices = face_indices[rng.randint(0,len(face_indices), size = 5000)]
        del face_vertex_mask
        _, distances = find_closest_points(meshdata.vertices[face_indices,:], meshdata.vertices[notface_indices,:])
        return distances, notface_indices


    def _compute_weights_dynamically(self, rot : Rotation):
        range, decay_start_at = 1., 0.1
        distances, notface_indices = self._dynamic_weight_parameters
        vertices = self._unposed_vertices
        if 1:
            yaw_sign = np.sign(rot.apply([0.,0.,1.])[0])
            up_axis_xy = rot.apply([0.,1.,0.])[:2]
            up_axis_xy = up_axis_xy / np.linalg.norm(up_axis_xy)
            side_axis_xy = np.asarray([-up_axis_xy[1], up_axis_xy[0]])
            #side_axis_xy = rot.apply([1.,0.,0.])[:2]
            #side_axis_xy = side_axis_xy / np.linalg.norm(side_axis_xy)
            proj = np.stack([side_axis_xy, up_axis_xy], axis=0)
            #proj = np.eye(2)
            relative_xy = (proj @ (vertices[notface_indices][:,:2]).T).T
            range_modulation = 0.5 + 2.*sigmoid(-yaw_sign*relative_xy[:,0]*2.)
            range = range * range_modulation
        falloff = np.exp(np.maximum(distances - decay_start_at, 0.) * (-1./range))
        weights = self._meshdata.vertex_weights.copy()
        weights[notface_indices] = falloff
        return self._meshdata._replace(vertex_weights = weights)


    def set_non_face_by_depth_estimate(self, inverse_depth):
        vertices = self._meshdata.vertices
        keypoints = vertices[self._keypoints,:]
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
        vertices[:,2] = np.minimum(vertices[:,2], vertices[:,2]*f + (1.-f)*zs)
        self._meshdata = self._meshdata._replace(vertices = vertices)
        return z_calibration_curves


    def _compute_laplacian(self, vertices):
        # Precompute Laplacian for smoothing
        mask = np.logical_and(self._meshdata.vertex_weights < 0.9, self._meshdata.vertex_weights > 0.01)
        mesh = trimesh.Trimesh(vertices = vertices, faces = self._meshdata.tris)
        pinned, = np.nonzero(~mask)
        return trimesh.smoothing.laplacian_calculation(mesh, pinned_vertices=pinned)


    def _apply_smoothing(self, vertices):
        mesh = trimesh.Trimesh(vertices = vertices, faces = self._meshdata.tris)
        trimesh.smoothing.filter_laplacian(mesh, lamb = 0.2, iterations=2, implicit_time_integration=True, volume_constraint=False, laplacian_operator=self._laplacian)
        return mesh.vertices


    def _compute_combined_transform(self, rotoffset):
        c = self._scale*self._rotation_center
        xyz = np.asarray([self._xy[0], self._xy[1], 0.])
        offset_trafo = affine3d_chain((rotoffset, c), (Rotation.identity(),  -c))
        sample_trafo = (self._rot, xyz)
        (R,t) = affine3d_chain(sample_trafo, offset_trafo)
        return (R, t)


    def __call__(self, rotoffset = None, shapeparam = None) -> Tuple[Meshdata,Tuple[Rotation,np.ndarray]]:
        if rotoffset is None:
            rotoffset = Rotation.identity()
        if shapeparam is None:
            shapeparam = self._shapeparam
        self._meshdata = self._compute_weights_dynamically(self._rot*rotoffset)
        vertices = re_pose(self._meshdata, self._faceparams, rotoffset, self._rotation_center, shapeparam)
        #vertices = self._apply_smoothing(vertices)
        tr = self._compute_combined_transform(rotoffset)
        return (Meshdata(
            vertices,
            self._meshdata.tris,
            self._meshdata.normals,
            None,
            self._uvs,
            self._meshdata.color,
            None
        ), tr)


class FaceAugmentationScene(object):
    def __init__(self, sample):
        xy = sample['xy']
        scale = sample['scale']
        shapeparam = sample['shapeparam']
        image = sample['image']
        rot = sample['rot']
        meshdata, keypoint_indices = FaceAugmentationScene.load_assets()
        self.face_model = face_model = FaceWithBackgroundModel(meshdata, xy, scale, rot, shapeparam, image)
        self.scene = scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0.0, 0.0, 0.0])
        self.material = create_pyrender_material(image, FaceWithBackgroundModel.texture_border)
        self.keypoint_indices = keypoint_indices
        FaceAugmentationScene.add_camera(scene, image.shape, scale, face_model.background_plane_z_coord)


    @contextlib.contextmanager
    def __call__(self, rotoffset = None, shapeparam = None):
        meshdata, tr = self.face_model(rotoffset, shapeparam)
        prim = pyrender.Primitive(
            positions = meshdata.vertices, 
            indices=meshdata.tris, 
            texcoord_0 = meshdata.uvs, 
            normals=meshdata.normals, 
            material=self.material,
            color_0 = meshdata.color)
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
    def load_mesh_data(filename):
        def _create_mesh(vertices, tris, shadowmap = None):
            tris = np.asarray(tris)
            vertices = np.asarray(vertices)
            vertices *= 0.01  # Vertices were changed during import in 3d software
            vertices[:,1] *= -1 
            # Dummy is good enough for unlit scene
            normals = np.broadcast_to(np.asarray([[ 0., 0., 1.]]), (len(vertices),3))
            color = np.tile(np.asarray(shadowmap)[:,None], (1,3)) if shadowmap is not None else None
            return Meshdata(vertices, tris, normals, None, None, color, None)

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            #print ("Loaded mesh data:", data.keys())
            md = _create_mesh(data['vertices'], data['tris'])
            md_teeth = _create_mesh(data['teeth_points'], data['teeth_tris'])
            md_surrounding = _create_mesh(data['surrounding_points'], data['surrounding_tris'])
            md_mouth = _create_mesh(data['mouth_points'], data['mouth_tris'], data['mouth_shadowmap'])
            md_surrounding = md_surrounding._replace(tris = np.ascontiguousarray(md_surrounding.tris[:,[2,1,0]]))
            idx_mouth_lower, = np.nonzero(data['mask_mouth_lower'])
            idx_mouth_upper, = np.nonzero(data['mask_mouth_upper'])
            return md, md_teeth, md_surrounding, md_mouth, (idx_mouth_lower, idx_mouth_upper)

    @staticmethod
    def join_meshes(headmesh : Meshdata, teethmesh : Meshdata, surrounding : Meshdata, mouth : Meshdata, indices : Tuple[npt.NDArray[np.integer],...], headmodel : bfm.BFMModel) -> Meshdata:
        assert headmesh.num_vertices == headmodel.vertexcount
        idx_mouth_lower, idx_mouth_upper = indices
        idx_mouth_upper_and_lower = np.concatenate([idx_mouth_lower, idx_mouth_upper])
        idx_face_subset = np.random.randint(0, headmesh.num_vertices, size=1000) # For speed
        num_bases = headmodel.scaled_bases.shape[0]
        def copy_basis(vertices_without_bases, vertices, basis, falloff=10000., decay_start_at=0.):
            # Ignore y-direction since teeth are vertical and we want the same basis for upper and lower end
            # Use the basis of the face mouth vertex which is closest
            idx_closest, distances = find_closest_points(vertices, vertices_without_bases)
            if 0:
                import matplotlib.pyplot as pyplot
                pyplot.scatter(vertices.T[0], vertices.T[2], c='r')
                pyplot.scatter(vertices[idx_closest].T[0], vertices[idx_closest].T[2],c='b',marker='x')
                pyplot.scatter(vertices_without_bases.T[0], vertices_without_bases.T[2],c='b')
                pyplot.show()
            weight = np.exp(-np.maximum(distances - decay_start_at, 0.) / falloff)
            return basis[:,idx_closest,:]*weight[None,:,None]
        def combine_triangles(meshes):
            vertexcounts = [0] + [ m.num_vertices for m in meshes[:-1] ]
            offsets = np.cumsum(vertexcounts)
            new_tris = np.concatenate([
                (m.tris + o) for m,o in zip(meshes, offsets)
            ], axis=0)
            return new_tris
        # This requires that the first N vertices of the headmesh correspond to the vertices in the prestine face model
        teeth_lower_basis = copy_basis(teethmesh.vertices[:,[0,2]], headmesh.vertices[idx_mouth_lower][:,[0,2]], headmodel.scaled_bases[:,idx_mouth_lower,:])
        teeth_upper_basis = copy_basis(teethmesh.vertices[:,[0,2]], headmesh.vertices[idx_mouth_upper][:,[0,2]], headmodel.scaled_bases[:,idx_mouth_upper,:])
        surrounding_basis = copy_basis(
            surrounding.vertices, 
            headmesh.vertices[idx_face_subset], 
            headmodel.scaled_bases[:,idx_face_subset,:], 0.01, 0.1)
        mouth_basis = copy_basis(
            mouth.vertices,
            headmesh.vertices[idx_mouth_upper_and_lower],
            headmodel.scaled_bases[:,idx_mouth_upper_and_lower,:], 1.)
        new_vertices = np.concatenate([
            headmesh.vertices,
            mouth.vertices, 
            teethmesh.vertices, 
            teethmesh.vertices,
            surrounding.vertices,], axis=0)
        new_tris = combine_triangles([ 
            headmesh, 
            mouth, 
            teethmesh, 
            teethmesh, 
            surrounding 
        ])
        new_basis = np.concatenate([
            headmodel.scaled_bases,
            mouth_basis,
            teeth_lower_basis,
            teeth_upper_basis,
            surrounding_basis
        ], axis=1)
        new_colors = np.concatenate([
            np.ones((headmesh.num_vertices,3)),
            mouth.color,
            # np.ones((mouth.num_vertices,3)),
            np.ones((teethmesh.num_vertices,3)),
            np.ones((teethmesh.num_vertices,3)),
            np.ones((surrounding.num_vertices,3))
        ], axis=0)
        new_normals = np.broadcast_to(np.asarray([[ 0., 0., 1.]]), (len(new_vertices),3))
        new_weights = np.concatenate([
            np.ones((headmesh.num_vertices,)),
            np.ones((mouth.num_vertices,)),
            np.ones((teethmesh.num_vertices*2,)),
            np.zeros((surrounding.num_vertices,))
        ], axis=0)
        #new_colors = new_weights[:,None]*np.asarray([[1.,0.,0.]])
        return Meshdata(new_vertices, new_tris, new_normals, new_weights, None, new_colors, new_basis)


    @staticmethod
    @functools.cache
    def load_assets():
        this_file_directory = os.path.dirname(__file__)
        headmesh, teethmesh, surrounding, mouth, indices = \
            FaceAugmentationScene.load_mesh_data(os.path.join(this_file_directory,"full_bfm_mesh_with_bg_v6.1.pkl"))
        headmodel = bfm.BFMModel(40, 10)
        if teethmesh is not None:
            meshdata = FaceAugmentationScene.join_meshes(headmesh, teethmesh, surrounding, mouth, indices, headmodel)
        else:
            assert ("Add padded basis vectors to return")
        return meshdata, headmodel.keypoints


def test_euler_angle_functions():
    import numpy.testing
    ref_rots = Rotation.random(num=1000)
    hpb = get_hpb(ref_rots)
    rots = make_rot(hpb)
    numpy.testing.assert_array_less((rots.inv() * ref_rots).magnitude(), 1.e-6)
    numpy.testing.assert_array_less((rots.inv() * make_rot_by_axis_rotations(hpb)).magnitude(), 1.e-6)

test_euler_angle_functions()