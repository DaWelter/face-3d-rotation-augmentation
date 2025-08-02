from typing import List, Dict, Any
from matplotlib import pyplot
import copy
import numpy as np
import cv2

import pyrender

import face3drotationaugmentation.vis as vis
import face3drotationaugmentation.graphics as graphics
import face3drotationaugmentation.sampling as sampling
from face3drotationaugmentation import depthestimation
from face3drotationaugmentation.common import AugmentedSample


def make_sample_for_passthrough(sample: AugmentedSample):
    meshdata, keypoint_indices = graphics.FaceAugmentationScene.load_assets()
    verts = graphics.compute_initial_posed_vertices(
        meshdata,
        graphics.Faceparams(
            xy=sample.xy,
            scale=sample.scale,
            rot=sample.rot,
            shapeparam=np.concatenate([sample.shapeparam, np.zeros((2,))]),  # Append eye closing parameters.
        ),
    )
    sample = sample._replace(
        pt3d_68=verts[keypoint_indices],
        roi=graphics.compute_bounding_box(verts[meshdata.get_face_vertex_mask()]),
        name=None,
    )
    return sample


def augment_sample(
    angle_step: float,
    prob_closed_eyes: float,
    prob_spotlight: float,
    rng: np.random.RandomState,
    sample: AugmentedSample,
):
    if not depthestimation.initialized:
        depthestimation.init()

    more_rots = sampling.sample_more_face_params(sample.rot, rng, angle_step)

    new_shapeparams, eyes_closing_amounts = sampling.sample_shapeparams(
        rng, sample.shapeparam, len(more_rots), prob_closed_eyes
    )
    new_lightparams = [sampling.sample_light(rng, r, prob_spotlight) for r in more_rots]

    blurred_depth, _ = _infer_nice_depth_estimate_from_image(sample)

    augscene = graphics.FaceAugmentationScene(sample)
    augscene.face_model.set_non_face_by_depth_estimate(blurred_depth)

    h, w, _ = sample.image.shape
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    for more_rot, new_shapeparam, eyes_closing_amount, light_param in zip(more_rots, new_shapeparams, eyes_closing_amounts, new_lightparams):  # type: ignore[arg-type]
        flags = pyrender.RenderFlags.NONE
        if new_lightparams is not None:
            flags = pyrender.RenderFlags.SHADOWS_ALL

        with augscene(more_rot, new_shapeparam, eyes_closing_amount, light_param) as items:
            scene, (R, t), keypoints, roi = items
            color, _ = renderer.render(scene, flags=flags)  # type: ignore[generalTypeIssues]
            color = np.ascontiguousarray(color)

        # roi = dataset300wlp.head_bbox_from_keypoints(keypoints)

        yield AugmentedSample(
            image=color, rot=R, xy=t[:2], scale=sample.scale, pt3d_68=keypoints, roi=roi, shapeparam=new_shapeparam
        )


def _infer_nice_depth_estimate_from_image(sample: AugmentedSample):
    depth_estimate = depthestimation.inference(sample.image)
    sigma = sample.scale * 0.02
    ks = int(sigma * 3) | 1
    blurred_depth = cv2.GaussianBlur(depth_estimate, (ks, ks), sigma)
    return blurred_depth, depth_estimate


class SampleVisualizerWindow:
    def __init__(self):
        self._plot_elements = None, None, None

    def show(self, sample: AugmentedSample):
        fig, ax, imgplot = self._plot_elements
        if fig is None or imgplot is None:
            fig, ax = pyplot.subplots(1, 1)
            imgplot = ax.imshow(np.zeros((5, 5, 3), dtype=np.uint8))
        img = sample.image.copy()
        vis.draw_pose(img, sample, 255, 2)
        imgplot.set_data(img)
        pyplot.show(block=False)
        self._plot_elements = (fig, ax, imgplot)
