from typing import List
import os
from matplotlib import pyplot
import numpy as np
import tqdm
import cv2
from contextlib import closing
import sys
import argparse

import pyrender

import face3drotationaugmentation.dataset300wlp as dataset300wlp
import face3drotationaugmentation.vis as vis
import face3drotationaugmentation.graphics as graphics
import face3drotationaugmentation.sampling as sampling
from face3drotationaugmentation import depthestimation
from face3drotationaugmentation.datasetwriter import dataset_writer

deg2rad = np.pi/180.


def infer_nice_depth_estimate_from_image(sample):
    depth_estimate = depthestimation.inference(sample['image'])
    sigma = sample['scale']*0.02
    ks = int(sigma*3) | 1
    blurred_depth = cv2.GaussianBlur(depth_estimate,(ks,ks),sigma)
    return blurred_depth, depth_estimate


def main(filename300wlp, outputfilename, max_num_frames):
    depthestimation.init()

    rng = np.random.RandomState()

    fig, ax, imgplot = None, None, None

    with closing(dataset300wlp.Dataset300WLP(filename300wlp)) as ds300wlp, dataset_writer(outputfilename) as writer:
        num_frames = min(max_num_frames, len(ds300wlp))

        #all_shapeparams = np.asarray([ s['shapeparam'] for _,s in tqdm.tqdm(zip(range(num_frames),ds300wlp), total=num_frames) ])

        for _, sample in tqdm.tqdm(zip(range(num_frames), ds300wlp), total=num_frames):
            name = sample['name']
            assert name.endswith("_0")
            name = name[:-2]
            
            more_rots = sampling.sample_more_face_params(sample['rot'], rng)

            new_shapeparams = [ sample['shapeparam'] for _ in more_rots ]
            # TODO: Sampling an new face shape requires to transfer the texture to the teeth
            #new_shapeparams = all_shapeparams[rng.randint(0,len(all_shapeparams),size=(num_additional_frames,))]

            blurred_depth, _ = infer_nice_depth_estimate_from_image(sample)

            augscene = graphics.FaceAugmentationScene(sample)
            augscene.face_model.set_non_face_by_depth_estimate(blurred_depth)

            h, w, _ = sample['image'].shape
            renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

            for more_rot, new_shapeparam in zip(more_rots, new_shapeparams):
                with augscene(more_rot, new_shapeparam) as items:
                    scene, (R,t), keypoints = items
                    color, _ = renderer.render(scene)
                    color = np.ascontiguousarray(color)

                roi = dataset300wlp.head_bbox_from_keypoints(keypoints)

                new_sample = {
                    'image' : color,
                    'rot' : R,
                    'xy' : t[:2],
                    'scale' : sample['scale'],
                    'pt3d_68' : keypoints,
                    'roi' : roi,
                    'shapeparam' : new_shapeparam
                }

                if np.random.randint(0,100)==0:
                    if fig is None:
                        fig, ax = pyplot.subplots(1,1)
                        imgplot = ax.imshow(np.zeros((5,5,3), dtype=np.uint8))
                    img = color.copy()
                    vis.draw_pose(img, new_sample, 255, 2)
                    imgplot.set_data(img)
                    pyplot.show(block=False)
                pyplot.pause(0.001)

                writer.write(name, new_sample)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pose dataset expander")
    parser.add_argument("_300wlp", type=str, help="300 wlp zip file")
    parser.add_argument("outputfilename", type=str, help="hdf5 file")
    parser.add_argument("-n", help="subset of n samples", type=int, default=1<<32)
    args = parser.parse_args()
    if not (args.outputfilename.lower().endswith('.h5') or args.outputfilename.lower().endswith('.hdf5')):
            raise ValueError("outputfilename must have hdf5 filename extension")
    main(args._300wlp, args.outputfilename, args.n)