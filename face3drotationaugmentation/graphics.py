from typing import Tuple, NamedTuple, Union, Optional
import contextlib
import os
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import pickle
from PIL import Image
import functools
import tqdm
import trimesh
import pyrender
import torch

from .common import FloatArray, IntArray, AugmentedSample
from .facemodel import bfm


def get_hpb(rot: Rotation) -> FloatArray:
    '''Conversion to heading-pitch-bank.

    In terms of extrinsic rotations, the first rotation is around z (bank/roll), then x (pitch), then y (yaw/heading)
    '''
    return rot.as_euler('YXZ')


def make_rot_by_axis_rotations(hpb: FloatArray) -> Rotation:
    '''For testing - to ensure we have the right convention.'''
    h, p, b = hpb.T
    z = np.zeros(h.shape)
    return (
        Rotation.from_rotvec(np.vstack([z, h, z]).T)
        * Rotation.from_rotvec(np.vstack([p, z, z]).T)
        * Rotation.from_rotvec(np.vstack([z, z, b]).T)
    )


def make_rot(hpb: FloatArray) -> Rotation:
    '''Conversion from heading-pitch-bank.

    In terms of extrinsic rotations, the first rotation is around z (bank/roll), then x (pitch), then y (yaw/heading)
    '''
    return Rotation.from_euler('YXZ', hpb)


def sigmoid(x: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-x))


def affine3d_chain(Ta: Tuple[Rotation, FloatArray], Tb: Tuple[Rotation, FloatArray]) -> Tuple[Rotation, FloatArray]:
    Ra, ta = Ta
    Rb, tb = Tb
    return Ra * Rb, Ra.apply(tb) + ta


def affine3d_inv(Ta: Tuple[Rotation, FloatArray]) -> Tuple[Rotation, FloatArray]:
    Ra, ta = Ta
    RaInv = Ra.inv()
    return RaInv, -RaInv.apply(ta)


def apply_affine3d(tr: Tuple[Rotation, FloatArray], vertices: FloatArray) -> FloatArray:
    R, t = tr
    return R.apply(vertices) + t


def apply_s_rot_t(vertices: FloatArray, xy: FloatArray, scale: float, rot: Rotation) -> FloatArray:
    "Scale, rotate and translate."
    vertices = rot.apply(vertices * scale)
    vertices[..., :2] += xy
    return vertices


def find_closest_points(reference_points: FloatArray, distanced_points: FloatArray) -> Tuple[np.ndarray, FloatArray]:
    if len(reference_points) * len(distanced_points) < 100:
        cross_distances = np.linalg.norm(reference_points[None, :, :] - distanced_points[:, None, :], axis=-1)
        idx_closest = np.argmin(cross_distances, axis=1)
        (distances,) = np.take_along_axis(cross_distances, idx_closest[:, None], axis=1).T
    else:
        kdtree = cKDTree(reference_points)
        distances, idx_closest = kdtree.query(distanced_points)
    return idx_closest, distances


def interpolate_images(values: torch.Tensor, sample_points: torch.Tensor) -> torch.Tensor:
    '''
    values: Image like with shape (...,C,H,W)
    sample_points: with shape (...,N,2)
    output: values at sample points with shape (...,N,C)
    '''
    C, H, W = values.shape[-3:]
    N = sample_points.size(-2)
    prefix_shape = values.shape[:-3]
    assert sample_points.shape[-2:] == (N, 2)
    values = values.view(-1, C, H, W)
    sample_points = sample_points * torch.tensor([[2.0 / W, 2.0 / H]]) - 1.0
    sample_points = sample_points.view(-1, N, 1, 2)
    samples = torch.nn.functional.grid_sample(
        values, sample_points, align_corners=False, mode='bilinear', padding_mode='border'
    )
    samples = samples.view(*prefix_shape, C, N).swapaxes(-2, -1)
    return samples


def interpolate_zero_channel_numpy_image(image: np.ndarray, sample_points: FloatArray) -> FloatArray:
    return interpolate_images(
        torch.from_numpy(image[None, :, :]).to(torch.float32), torch.from_numpy(sample_points).to(torch.float32)
    )[:, 0].numpy()


def apply_blendshapes(vertices: FloatArray, blend_shape_vertices: FloatArray, shapeparam: FloatArray) -> FloatArray:
    '''Adds the blend-shapes to the vertices using "shapeparam" as weights.'''
    return vertices + np.sum((blend_shape_vertices * shapeparam[:, None, None]), axis=0)


def estimate_vertex_normals(vertices: FloatArray, tris: IntArray) -> FloatArray:
    face_normals = trimesh.Trimesh(vertices, tris).face_normals
    new_normals = trimesh.geometry.mean_vertex_normals(len(vertices), tris, face_normals)
    assert isinstance(new_normals, np.ndarray)
    assert new_normals.shape == vertices.shape, f"{new_normals.shape} vs {vertices.shape}"
    return new_normals


def compute_bounding_box(vertices: FloatArray) -> FloatArray:
    '''Output format is x0,y0,x1,y1 (left,top,right,bottom).'''
    min_ = np.amin(vertices[..., :2], axis=-2)
    max_ = np.amax(vertices[..., :2], axis=-2)
    return np.concatenate([min_, max_], axis=-1).astype(np.float32)


class BaseMeshdata(NamedTuple):
    vertices: FloatArray
    tris: IntArray  # (Num Tris, 3)
    normals: FloatArray
    color: FloatArray

    @property
    def num_vertices(self):
        return self.vertices.shape[0]

    @property
    def num_tris(self):
        return self.tris.shape[0]


class Meshdata(NamedTuple):
    vertices: FloatArray
    tris: IntArray  # (Num Tris, 3)
    normals: FloatArray
    color: FloatArray
    vertex_weights: FloatArray
    uvs: FloatArray
    deformbasis: FloatArray | None  # (Basis Vectors, Points, 3)

    @property
    def num_vertices(self):
        return self.vertices.shape[0]

    @property
    def num_tris(self):
        return self.tris.shape[0]

    def get_face_vertex_mask(self):
        '''By convention the original area of the BFM has weight 1.

        Refers to the full mesh. Other vertices have less. This separates the
        face from the background.
        '''
        return self.vertex_weights > 0.99


class Faceparams(NamedTuple):
    xy: FloatArray
    scale: float
    rot: Rotation
    shapeparam: FloatArray


def compute_initial_posed_vertices(meshdata: Meshdata, faceparams: Faceparams):
    """Given the pose parameters and the normalized base mesh.

    Applies scaling and translation. Then rotates, and blends the result with the unrotated
    vertices using the meshes vertex_weights.

    Returns posed vertices.
    """
    assert meshdata.deformbasis is not None
    vertices = apply_blendshapes(meshdata.vertices, meshdata.deformbasis, faceparams.shapeparam)
    unrotated_vertices = apply_s_rot_t(vertices, faceparams.xy, faceparams.scale, Rotation.identity())
    vertices = apply_s_rot_t(vertices, faceparams.xy, faceparams.scale, faceparams.rot)
    f = meshdata.vertex_weights
    vertices = vertices * f[:, None] + (1.0 - f)[:, None] * unrotated_vertices
    return vertices


def re_pose(meshdata: Meshdata, original_faceparams: Faceparams, rot_offset, rot_offset_center, new_shapeparam):
    """Handles the desired offset rotation from the baseline.

    Conceptually, the frontal face is first rotated according around "rot_offset_center" by "rot_offset". Also the new
    shape parameters are applied, overriding the baseline. Then the face is rotated by the original rotation. (i.e. the given
    offset is multiplied from the right.)

    In reality this function expects the fully posed vertices of the baseline pose!

    Returns posed vertices.
    """
    assert meshdata.deformbasis is not None
    # First reverse the original pose transform, ignoring scale
    vertices = meshdata.vertices.copy()
    vertices[:, :2] -= original_faceparams.xy
    vertices = original_faceparams.rot.inv().apply(vertices)
    # Then pose the vertices according to the offset pose
    scale = original_faceparams.scale
    rot_offset_center = scale * rot_offset_center
    vertices = apply_blendshapes(
        vertices, scale * meshdata.deformbasis, new_shapeparam - original_faceparams.shapeparam
    )
    vertices = rot_offset.apply(vertices - rot_offset_center) + rot_offset_center
    vertices = apply_s_rot_t(vertices, original_faceparams.xy, 1.0, original_faceparams.rot)
    f = meshdata.vertex_weights
    # Blend with the input vertices
    vertices = vertices * f[:, None] + (1.0 - f)[:, None] * meshdata.vertices
    return vertices


class FaceWithBackgroundModel(object):
    '''Generates the mesh with augmented face rotations.'''

    texture_border = 50  # pixels
    # Center for the augmented rotation. It is specified in the local BFM frame.
    rotation_center = np.array([0.0, 0.5, 0.1])

    def __init__(self, meshdata: Meshdata, xy, scale, rot, shapeparam, image):
        self._keypoint_idx = bfm.BFMModel.load().keypoints  # TODO: restructure code to remove this
        self._meshdata = meshdata
        self._rot = rot
        self._xy = xy
        self._scale = scale
        self._shapeparam = shapeparam
        self._faceparams = Faceparams(xy, scale, rot, shapeparam)
        # For getting the bounding box of the face region.
        # Could use the bfm.BFMModel class for this too maybe.
        # Not sure if it's on the same scale.
        self._face_vertex_mask = meshdata.get_face_vertex_mask()

        h, w = image.shape[:2]

        # Parameters to help compute the deformation weights for
        # blending the rotated bits with the stationary surrounding.
        self._dynamic_weight_parameters = self._compute_dynamic_weight_parameters()

        self._meshdata = self._compute_weights_dynamically(rot)

        vertices_according_to_pose = compute_initial_posed_vertices(self._meshdata, self._faceparams)

        if 0:
            # Looks good now without it
            self._laplacian = self._compute_laplacian(vertices_according_to_pose)
            vertices_according_to_pose = self._apply_smoothing(vertices_according_to_pose)

        self._meshdata = self._meshdata._replace(vertices=vertices_according_to_pose)

        # By default, without texture border, the range from [0,w] is mapped to [0,1]
        # With border ...
        #    Range from -border to w+border is mapped to [0,1]
        uvs = (vertices_according_to_pose[:, :2] + self.texture_border) / (
            np.asarray([[w, h]]) + 2 * self.texture_border
        )
        uvs[:, 1] = 1.0 - uvs[:, 1]
        self._uvs = uvs

        self.background_plane_z_coord = np.average(vertices_according_to_pose[self._meshdata.vertex_weights < 0.01, 2])

    def _compute_dynamic_weight_parameters(self):
        '''Computes currently ...

        Return:
            * Distance to face model for each surrounding ("notface") vertex
            * Indices of notface vertices
            * Copy of all mesh vertices. With the function called in the right place these will be
            the unposed unscaled model where the face is in the center. These can be used more easily
            due to the fixed location of the face.
        '''
        meshdata = self._meshdata
        # TODO: a better way to recover the vertices of the face model (without surrounding)
        face_vertex_mask = self._face_vertex_mask
        (face_indices,) = np.nonzero(face_vertex_mask)
        (notface_indices,) = np.nonzero(~face_vertex_mask)
        # Thin out the vertices for reasonably fast closest distance computation
        # TODO: maybe a smarter closest point computation with spatial search structures like kd-tree.
        del face_vertex_mask
        _, distances = find_closest_points(meshdata.vertices[face_indices, :], meshdata.vertices[notface_indices, :])
        return distances, notface_indices, meshdata.vertices.copy()

    def _compute_weights_dynamically(self, rot: Rotation) -> Meshdata:
        """Computes weights depending on the desired pose.

        The vertex weights of the face are all 1 to apply the full rotation there.
        From the face the weight falls off more or less slowly into the vertices
        of the surrounding.

        Using the rotation axis only a.t.m.
        """
        range, decay_start_at = 1.0, 0.1
        distances, notface_indices, vertices = self._dynamic_weight_parameters

        # It looked like in 300W-LP the range of the deformation is asymmetrical
        # between left and right, depending on the yaw. This is tried here too.
        yaw_sign = np.sign(rot.apply([0.0, 0.0, 1.0])[0])
        up_axis_xy = rot.apply([0.0, 1.0, 0.0])[:2]
        up_axis_xy = up_axis_xy / np.linalg.norm(up_axis_xy)
        side_axis_xy = np.asarray([-up_axis_xy[1], up_axis_xy[0]])
        proj = np.stack([side_axis_xy, up_axis_xy], axis=0)
        relative_xy = (proj @ (vertices[notface_indices][:, :2]).T).T
        range_modulation = 0.5 + 1.0 * sigmoid(-yaw_sign * relative_xy[:, 0] * 2.0)
        range = range * range_modulation

        falloff = np.exp(np.maximum(distances - decay_start_at, 0.0) * (-1.0 / range))
        weights = self._meshdata.vertex_weights.copy()
        weights[notface_indices] = falloff
        return self._meshdata._replace(vertex_weights=weights)

    def set_non_face_by_depth_estimate(self, inverse_depth):
        """Set the surroundings depth from a depth estimation."""
        vertices = self._meshdata.vertices
        # The provided estimate should scale linearly and with a 1:1 ratio w.r.t.
        # with the size of the items in the visual image. However it comes with an
        # unknown offset.
        # The attempt here is to calculate this offset out by matching the depth at
        # the face landmarks. There the depth of the mesh is known to be good and so
        # the offset can be calculated and applied to all over the depth estimate.
        keypoints = vertices[self._keypoint_idx, :]
        depth = -inverse_depth
        depth_estimate_zs = interpolate_zero_channel_numpy_image(depth, keypoints[:, :2])
        calibration_offset = np.average(keypoints[:, 2] - depth_estimate_zs)
        depth = depth + calibration_offset
        depth = np.clip(depth, -1.5 * self._faceparams.scale, 1.5 * self._faceparams.scale)
        depth_estimate_zs += calibration_offset
        z_calibration_curves = (keypoints[:, 0], keypoints[:, 2], depth_estimate_zs)
        # Finally the corrected depth estimate can be applied to the vertices of the
        # surrounding. Smooth blending is applied using the vertex weights in order
        # to maintain the original face vertices.
        zs = interpolate_zero_channel_numpy_image(depth, vertices[:, :2])
        vertices = vertices.copy()
        f = np.power(self._meshdata.vertex_weights, 2.0)
        vertices[:, 2] = np.minimum(vertices[:, 2], vertices[:, 2] * f + (1.0 - f) * zs)
        self._meshdata = self._meshdata._replace(vertices=vertices)
        return z_calibration_curves

    def _compute_laplacian(self, vertices):
        # Precompute Laplacian for smoothing
        mask = np.logical_and(self._meshdata.vertex_weights < 0.9, self._meshdata.vertex_weights > 0.01)
        mesh = trimesh.Trimesh(vertices=vertices, faces=self._meshdata.tris)
        (pinned,) = np.nonzero(~mask)
        return trimesh.smoothing.laplacian_calculation(mesh, pinned_vertices=pinned)

    def _apply_smoothing(self, vertices):
        mesh = trimesh.Trimesh(vertices=vertices, faces=self._meshdata.tris)
        trimesh.smoothing.filter_laplacian(
            mesh,
            lamb=0.2,
            iterations=2,
            implicit_time_integration=True,
            volume_constraint=False,
            laplacian_operator=self._laplacian,
        )
        return mesh.vertices

    def _compute_combined_transform(self, rotoffset):
        '''Takes care of computing the new face location mostly.'''
        # The output rotation is just the concatenation off the baseline rotation and the offset.
        c = self._scale * self.rotation_center
        xyz = np.asarray([self._xy[0], self._xy[1], 0.0])
        offset_trafo = affine3d_chain((rotoffset, c), (Rotation.identity(), -c))
        sample_trafo = (self._rot, xyz)
        (R, t) = affine3d_chain(sample_trafo, offset_trafo)
        return (R, t)

    def __call__(self, new_rot=None, new_shapeparam=None) -> Tuple[Meshdata, Tuple[Rotation, FloatArray], FloatArray]:
        if new_rot is None:
            new_rot = self._rot
        if new_shapeparam is None:
            new_shapeparam = self._shapeparam
        self._meshdata = self._compute_weights_dynamically(new_rot)
        rotoffset = self._rot.inv() * new_rot
        vertices = re_pose(self._meshdata, self._faceparams, rotoffset, self.rotation_center, new_shapeparam)
        normals = estimate_vertex_normals(vertices, self._meshdata.tris)
        bbox = compute_bounding_box(vertices[self._face_vertex_mask])
        if 0:
            # Looks good without it
            vertices = self._apply_smoothing(vertices)
        tr = self._compute_combined_transform(rotoffset)
        assert self._meshdata.vertex_weights is not None
        return (
            Meshdata(
                vertices=vertices,
                tris=self._meshdata.tris,
                normals=normals,
                vertex_weights=self._meshdata.vertex_weights.copy(),
                uvs=self._uvs,
                color=self._meshdata.color,
                deformbasis=None,
            ),
            tr,
            bbox,
        )


def create_pyrender_material(original_image, texture_border):
    '''
    Creates the material for the face mesh. Takes the original image of the data sample.
    Before creating the texture, a border / padding is added on all sides. This is needed
    to pad the output image in black rather than with parts of the original image.
    The UV coordinate generation will take this padding into account.
    '''
    h, w = original_image.shape[:2]
    tex = np.zeros((h + 2 * texture_border, w + 2 * texture_border, 4), dtype=np.uint8)
    tex[texture_border:-texture_border, texture_border:-texture_border, :3] = original_image
    tex[:, :, 3] = 255  # Alpha channel

    tex = pyrender.Texture(
        source=tex,
        source_channels='RGBA',  # RGB doesn't work
        data_format=pyrender.texture.GL_UNSIGNED_BYTE,
        sampler=pyrender.Sampler(wrapS=pyrender.GLTF.CLAMP_TO_EDGE, wrapT=pyrender.GLTF.CLAMP_TO_EDGE),
    )
    # The only material that pyrender supports
    return pyrender.MetallicRoughnessMaterial(
        baseColorTexture=tex,
        doubleSided=False,
        # metallicFactor=1.,
        # roughnessFactor=1.,
        emissiveFactor=0.0,
        emissiveTexture=tex,
    )


def _rotvec_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis_x_sin = np.cross(a, b)
    cos_ = np.dot(a, b)
    if cos_ < -1.0 + 1.0e-6:
        return np.array([0.0, np.pi, 0.0])
    if cos_ < 1.0 - 1.0e-6:
        return axis_x_sin / np.linalg.norm(axis_x_sin) * np.arccos(cos_)
    return np.zeros((3,))


class SpotlightLookingAtPoint(object):
    def __init__(self, distance, look_at_point, roi_radius):
        self._distance = distance
        self._look_at_point_x, self._look_at_point_y = look_at_point
        # Keep intensity at target point constant, independent of distance
        intensity = 20.0 * np.square(distance)
        innerAngle = np.arctan2(roi_radius, distance)
        outerAngle = np.arctan2(2.0 * roi_radius, distance)

        self._light = pyrender.SpotLight(
            color=[1.0, 1.0, 1.0], intensity=intensity, range=1.0, innerConeAngle=innerAngle, outerConeAngle=outerAngle
        )
        self._node = pyrender.Node(
            light=self._light, matrix=self._direction_vector_to_pose_matrix(np.array([0.0, 0.0, 1.0]))
        )

        # PointLight seems broken. Get only black image.
        # self.light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10000000.)
        # Set custom shadow texture: won't be written into / not working
        # self.light.shadow_texture = pyrender.Texture(width=4096, height=4096, source_channels='D', data_format=pyrender.light.GL_FLOAT)

    @property
    def node(self):
        return self._node

    def _direction_vector_to_pose_matrix(self, v):
        assert v[2] >= 0.0, "No backlighting?"
        v = v * (self._distance / np.linalg.norm(v))
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(_rotvec_between(np.asarray([0.0, 0.0, -1.0]), v)).as_matrix()
        pose[:3, 3] = -v + np.array([self._look_at_point_x, self._look_at_point_y, 0.0])
        return pose

    def update(self, direction_vec):
        self._node.matrix = self._direction_vector_to_pose_matrix(direction_vec)


class FaceAugmentationScene(object):
    '''Handles the pyrender parts mostly.'''

    def __init__(self, sample: AugmentedSample):
        shapeparam = np.concatenate([sample.shapeparam, [0.0, 0.0]])
        meshdata, keypoint_indices = FaceAugmentationScene.load_assets()
        self.face_model = face_model = FaceWithBackgroundModel(
            meshdata, sample.xy, sample.scale, sample.rot, shapeparam, sample.image
        )
        self.scene = scene = pyrender.Scene(ambient_light=[0.0, 0.0, 0.0], bg_color=[0.0, 0.0, 0.0])
        self.material = create_pyrender_material(sample.image, FaceWithBackgroundModel.texture_border)
        self.keypoint_indices = keypoint_indices
        FaceAugmentationScene.add_camera(scene, sample.image.shape, sample.scale, face_model.background_plane_z_coord)
        self.light = SpotlightLookingAtPoint(
            distance=sample.scale * 10, look_at_point=sample.xy, roi_radius=sample.scale
        )

    @contextlib.contextmanager
    def __call__(
        self,
        new_rot=None,
        new_shapeparam: FloatArray | None = None,
        eyes_closing: FloatArray | None = None,
        light_direction_vec=None,
    ):
        '''Temporarily assembles the scene and ...

        Returns
            The scene
            The rotation and translation of new face
            The 68 3d landmarks
        '''
        assert (new_shapeparam is None) == (eyes_closing is None)
        if new_shapeparam is not None and eyes_closing is not None:
            new_shapeparam = np.concatenate([new_shapeparam, eyes_closing])
        meshdata, tr, bbox = self.face_model(new_rot, new_shapeparam)
        prim = pyrender.Primitive(
            positions=meshdata.vertices,
            indices=meshdata.tris,
            texcoord_0=meshdata.uvs,
            normals=meshdata.normals,
            material=self.material,
            color_0=meshdata.color,
        )
        mesh = pyrender.Mesh(primitives=[prim])
        face_node = self.scene.add(mesh)
        if light_direction_vec is not None:
            self.light.update(light_direction_vec)
            self.scene.add_node(self.light.node)
        else:
            self.material.emissiveFactor = 1.0
        try:
            keypoints = meshdata.vertices[self.keypoint_indices]
            yield (self.scene, tr, keypoints, bbox)
        finally:
            self.scene.remove_node(face_node)
            if light_direction_vec is not None:
                self.scene.remove_node(self.light.node)
            self.material.emissiveFactor = 0.0

    @staticmethod
    def add_camera(scene, image_shape, scale, background_plane_z_coord):
        # Perspective camera with very long focal length to fake an orthographic camera.
        # The proper orthographic camera didn't work.
        h, w, _ = image_shape
        zdistance = 10000
        fov = 2.0 * np.arctan(0.5 * (h) / (zdistance + background_plane_z_coord))
        cam = pyrender.PerspectiveCamera(yfov=fov, znear=zdistance - scale * 2, zfar=zdistance + scale * 2)
        campose = np.eye(4)
        campose[:3, 3] = [w // 2, h // 2, -zdistance]
        campose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, -1, 0]]
        scene.add(cam, pose=campose)

    @staticmethod
    def load_mesh_data(filename):
        def _create_points(vertices):
            vertices = np.asarray(vertices)
            vertices *= 0.01  # Vertices were changed during import in 3d software
            vertices[:, 1] *= -1
            return vertices

        def _create_mesh(vertices, tris, shadowmap=None):
            tris = np.asarray(tris)
            vertices = _create_points(vertices)
            # Dummy is good enough for unlit scene
            normals = np.broadcast_to(np.asarray([[0.0, 0.0, 1.0]]), (len(vertices), 3))
            color = np.tile(np.asarray(shadowmap)[:, None], (1, 3)) if shadowmap is not None else np.ones_like(vertices)
            return BaseMeshdata(vertices, tris, normals, color)

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            md = _create_mesh(data['vertices'], data['tris'])
            md_teeth = _create_mesh(data['teeth_points'], data['teeth_tris'])
            md_surrounding = _create_mesh(data['surrounding_points'], data['surrounding_tris'])
            md_mouth = _create_mesh(data['mouth_points'], data['mouth_tris'], data['mouth_shadowmap'])
            md_surrounding = md_surrounding._replace(tris=np.ascontiguousarray(md_surrounding.tris[:, [2, 1, 0]]))
            (idx_mouth_lower,) = np.nonzero(data['mask_mouth_lower'])
            (idx_mouth_upper,) = np.nonzero(data['mask_mouth_upper'])
            shape_left_eye_close = _create_points(data['ev_left_eye'])
            shape_right_eye_close = _create_points(data['ev_right_eye'])
            return (
                md,
                md_teeth,
                md_surrounding,
                md_mouth,
                (idx_mouth_lower, idx_mouth_upper),
                (shape_left_eye_close, shape_right_eye_close),
            )

    @staticmethod
    def join_meshes(
        headmesh: BaseMeshdata,
        teethmesh: BaseMeshdata,
        surrounding: BaseMeshdata,
        mouth: BaseMeshdata,
        indices: Tuple[npt.NDArray[np.integer], ...],
        headmodel: bfm.BFMModel,
    ) -> Meshdata:
        '''Assembles bits to a large mesh.

        Initially only the face vertices have a deformation basis. This function copies from this basis to appropriate parks like the teeth so they
        move with the face model, for example. The surrounding is padded mostly with zeros otoh. Then there are other things to consider
        like colors and normals ...
        '''
        assert headmesh.num_vertices == headmodel.vertexcount
        idx_mouth_lower, idx_mouth_upper = indices
        idx_mouth_upper_and_lower = np.concatenate([idx_mouth_lower, idx_mouth_upper])

        def extend_deform_basis(
            vertices_without_bases: FloatArray,
            vertices: FloatArray,
            basis: FloatArray,
            falloff: float = 10000.0,
            decay_start_at: float = 0.0,
        ):
            '''Extends the deform basis to new vertices.

            Takes the basis vectors from the closest points in the reference mesh and applies a distance based falloff.
            The falloff is useful for background. Otherwise, for actual parts of the face, not needed.

            Returns:
                Deform basis (#bases, #vertices_without_bases, 3)
            '''
            idx_closest, distances = find_closest_points(vertices, vertices_without_bases)
            if 0:
                import matplotlib.pyplot as pyplot

                fig, axes = pyplot.subplots(1, 2)
                axes[0].scatter(vertices.T[0], vertices.T[-1], c='r')
                axes[0].scatter(vertices[idx_closest].T[0], vertices[idx_closest].T[-1], c='b', marker='x')
                axes[0].scatter(vertices_without_bases.T[0], vertices_without_bases.T[-1], c='b')
                axes[1].scatter(vertices.T[0], vertices.T[1], c='r')
                axes[1].scatter(vertices[idx_closest].T[0], vertices[idx_closest].T[1], c='b', marker='x')
                axes[1].scatter(vertices_without_bases.T[0], vertices_without_bases.T[1], c='b')
                pyplot.show()
            normalized_distance = np.maximum(distances - decay_start_at, 0.0) / falloff
            weight = 0.5 * (np.cos(np.minimum(1.0, normalized_distance) * np.pi) + 1.0)
            return basis[:, idx_closest, :] * weight[None, :, None]

        def concat_triangles(meshes):
            '''Takes care of incrementing the vertex indices.'''
            vertexcounts = [0] + [m.num_vertices for m in meshes[:-1]]
            offsets = np.cumsum(vertexcounts)
            new_tris = np.concatenate([(m.tris + o) for m, o in zip(meshes, offsets)], axis=0)
            return new_tris

        zeroy = np.asarray([[1.0, 0.0, 1.0]])
        # This requires that the first N vertices of the headmesh correspond to the vertices in the prestine face model
        teeth_lower_basis = extend_deform_basis(
            teethmesh.vertices * zeroy,
            headmesh.vertices[idx_mouth_lower] * zeroy,
            headmodel.scaled_bases[:, idx_mouth_lower, :],
        )
        teeth_upper_basis = extend_deform_basis(
            teethmesh.vertices * zeroy,
            headmesh.vertices[idx_mouth_upper] * zeroy,
            headmodel.scaled_bases[:, idx_mouth_upper, :],
        )
        surrounding_basis = extend_deform_basis(
            surrounding.vertices, headmesh.vertices[:], headmodel.scaled_bases[:, :, :], 0.3, 0.02
        )
        mouth_basis = extend_deform_basis(
            mouth.vertices,
            headmesh.vertices[idx_mouth_upper_and_lower],
            headmodel.scaled_bases[:, idx_mouth_upper_and_lower, :],
            1.0,
        )
        new_vertices = np.concatenate(
            [
                headmesh.vertices,
                mouth.vertices,
                teethmesh.vertices,
                teethmesh.vertices,
                surrounding.vertices,
            ],
            axis=0,
        )
        new_tris = concat_triangles([headmesh, mouth, teethmesh, teethmesh, surrounding])
        new_basis = np.concatenate(
            [headmodel.scaled_bases, mouth_basis, teeth_lower_basis, teeth_upper_basis, surrounding_basis], axis=1
        )
        assert mouth.color is not None
        new_colors = np.concatenate(
            [
                np.ones((headmesh.num_vertices, 3)),
                mouth.color,
                # np.ones((mouth.num_vertices,3)),
                np.ones((teethmesh.num_vertices, 3)),
                np.ones((teethmesh.num_vertices, 3)),
                np.ones((surrounding.num_vertices, 3)),
            ],
            axis=0,
        )
        new_normals = estimate_vertex_normals(new_vertices, new_tris)
        new_weights = np.concatenate(
            [
                np.ones((headmesh.num_vertices,)),
                np.ones((mouth.num_vertices,)),
                np.ones((teethmesh.num_vertices * 2,)),
                np.zeros((surrounding.num_vertices,)),
            ],
            axis=0,
        )
        new_uvs = np.zeros((new_vertices.shape[0], 2), dtype=np.float32)
        return Meshdata(new_vertices, new_tris, new_normals, new_colors, new_weights, new_uvs, new_basis)

    @staticmethod
    def extend_bfm_expression_basis(model: bfm.BFMModel, additional_deform_shapes):
        deform_deltas = [(x - model.scaled_vertices) for x in additional_deform_shapes]
        return model._replace(scaled_bases=np.concatenate([model.scaled_bases, deform_deltas], axis=0))

    @staticmethod
    @functools.cache
    def load_assets():
        this_file_directory = os.path.dirname(__file__)
        headmesh, teethmesh, surrounding, mouth, indices, additional_deform_shapes = (
            FaceAugmentationScene.load_mesh_data(os.path.join(this_file_directory, "full_bfm_mesh_with_bg_v7.pkl"))
        )
        headmodel = bfm.BFMModel.load()
        headmodel = FaceAugmentationScene.extend_bfm_expression_basis(headmodel, additional_deform_shapes)
        meshdata = FaceAugmentationScene.join_meshes(headmesh, teethmesh, surrounding, mouth, indices, headmodel)
        return meshdata, headmodel.keypoints


def test_euler_angle_functions():
    import numpy.testing

    ref_rots = Rotation.random(num=1000)
    hpb = get_hpb(ref_rots)
    rots = make_rot(hpb)
    numpy.testing.assert_array_less((rots.inv() * ref_rots).magnitude(), 1.0e-6)
    numpy.testing.assert_array_less((rots.inv() * make_rot_by_axis_rotations(hpb)).magnitude(), 1.0e-6)


def test_find_closest_points():
    a, b = find_closest_points(
        np.asarray([[0.0, 10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]), np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    )
    np.testing.assert_array_equal(a, [1, 2])
    np.testing.assert_allclose(b, [1.0, 0.0])


# Maybe I should do more of these ...
test_euler_angle_functions()
test_find_closest_points()
