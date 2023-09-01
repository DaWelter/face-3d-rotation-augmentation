from typing import List
from matplotlib import pyplot
import numpy as np
import numpy.typing as npt
import cv2

import pyrender

import face3drotationaugmentation.dataset300wlp as dataset300wlp
import face3drotationaugmentation.vis as vis
import face3drotationaugmentation.graphics as graphics
import face3drotationaugmentation.sampling as sampling
from face3drotationaugmentation import depthestimation


deg2rad = np.pi/180.


def augment_sample(angle_step, prob_closed_eyes, prob_spotlight, rng, sample):
    if not depthestimation.initialized:
        depthestimation.init()

    more_rots = sampling.sample_more_face_params(sample['rot'], rng, angle_step)

    new_shapeparams, eyes_closing_amounts = sampling.sample_shapeparams(rng, sample['shapeparam'], len(more_rots), prob_closed_eyes)
    new_lightparams = [ sampling.sample_light(rng, r, prob_spotlight) for r in more_rots ]

    blurred_depth, _ = _infer_nice_depth_estimate_from_image(sample)

    augscene = graphics.FaceAugmentationScene(sample)
    augscene.face_model.set_non_face_by_depth_estimate(blurred_depth)

    h, w, _ = sample['image'].shape
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    for more_rot, new_shapeparam, eyes_closing_amount, light_param in zip(more_rots, new_shapeparams, eyes_closing_amounts, new_lightparams):
        flags = pyrender.RenderFlags.NONE
        if new_lightparams is not None:
            flags = pyrender.RenderFlags.SHADOWS_ALL

        with augscene(more_rot, new_shapeparam, eyes_closing_amount, light_param) as items:
            scene, (R,t), keypoints = items
            color, _ = renderer.render(scene, flags=flags)
            color = np.ascontiguousarray(color)

        roi = dataset300wlp.head_bbox_from_keypoints(keypoints)

        yield {
            'image' : color,
            'rot' : R,
            'xy' : t[:2],
            'scale' : sample['scale'],
            'pt3d_68' : keypoints,
            'roi' : roi,
            'shapeparam' : new_shapeparam
        }


def _infer_nice_depth_estimate_from_image(sample):
    depth_estimate = depthestimation.inference(sample['image'])
    sigma = sample['scale']*0.02
    ks = int(sigma*3) | 1
    blurred_depth = cv2.GaussianBlur(depth_estimate,(ks,ks),sigma)
    return blurred_depth, depth_estimate


class SampleVisualizerWindow:
    def __init__(self):
        self._plot_elements = None, None, None

    def show(self, sample):
        fig, ax, imgplot = self._plot_elements
        if fig is None:
            fig, ax = pyplot.subplots(1,1)
            imgplot = ax.imshow(np.zeros((5,5,3), dtype=np.uint8))
        img = sample['image'].copy()
        vis.draw_pose(img, sample, 255, 2)
        imgplot.set_data(img)
        pyplot.show(block=False)
        self._plot_elements = (fig, ax, imgplot)
