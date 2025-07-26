from matplotlib import pyplot
import numpy as np
import tqdm
from contextlib import closing
import argparse

import face3drotationaugmentation.dataset300wlp as dataset300wlp
from face3drotationaugmentation.generate import augment_sample, SampleVisualizerWindow, make_sample_for_passthrough
from face3drotationaugmentation.datasetwriter import dataset_writer

def main(filename300wlp : str, outputfilename : str, max_num_frames : int, enable_vis : bool, angle_step: float, prob_closed_eyes : float, prob_spotlight : float):
    rng = np.random.RandomState(seed=1234567)

    visualizer = SampleVisualizerWindow()

    with closing(dataset300wlp.Dataset300WLP(filename300wlp)) as ds300wlp, dataset_writer(outputfilename) as writer:
        num_frames = min(max_num_frames, len(ds300wlp))

        for _, (name,sample) in tqdm.tqdm(zip(range(num_frames), ds300wlp), total=num_frames):
            assert name.endswith("_0")
            name = name[:-2]
            
            # TODO: Remove regeneration of input-image (which has imperfections
            # due to the rendering) and enable this pass-through.
            #original_out = make_sample_for_passthrough(sample)
            #writer.write(name, original_out)

            generated_samples = list(augment_sample(angle_step, prob_closed_eyes, prob_spotlight, rng, sample))

            if enable_vis and np.random.randint(0,10)==0:
                visualizer.show(next(iter(generated_samples)))
            pyplot.pause(0.001)

            for new_sample in generated_samples:
                writer.write(name, new_sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pose dataset expander")
    parser.add_argument("_300wlp", type=str, help="300 wlp zip file")
    parser.add_argument("outputfilename", type=str, help="hdf5 file")
    parser.add_argument("-n", help="subset of n samples", type=int, default=1<<32)
    parser.add_argument("--yaw-step", type=float, default=5., help="the increment of yaw angle per sample")
    parser.add_argument("--prob-closed-eyes", type=float, default=0., help="probability for closing eyes (between 0 and 1)")
    parser.add_argument("--prob-spotlight", type=float, default=0., help="Probability to add spotlight shining from the side (between 0 and 1)")
    args = parser.parse_args()
    if not (args.outputfilename.lower().endswith('.h5') or args.outputfilename.lower().endswith('.hdf5')):
            raise ValueError("outputfilename must have hdf5 filename extension")
    main(args._300wlp, args.outputfilename, args.n, enable_vis=True, angle_step=args.yaw_step, prob_closed_eyes=args.prob_closed_eyes, prob_spotlight=args.prob_spotlight)