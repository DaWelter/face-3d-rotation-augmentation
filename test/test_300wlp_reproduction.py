from pathlib import Path
from PIL import Image
import numpy as np
import h5py
from collections import defaultdict

from expand_dataset import main


def assert_similar_images(reference : Path, output : Path):
    reference_images = list(x.relative_to(reference) for x in reference.glob("*"))
    output_images = list(x.relative_to(output) for x in output.glob("*"))
    assert set(reference_images) == set(output_images)
    for name in reference_images:
        ref = np.asarray(Image.open(reference / name))
        img = np.asarray(Image.open(output / name))
        np.testing.assert_allclose(img, ref, atol=5)


def assert_sequence_starts_ok(sequence_starts, n):
    assert len(sequence_starts) == 3
    assert sequence_starts[0] == 0
    assert sequence_starts[-1] == n
    assert 15 < sequence_starts[1] < 25


def assert_similar_labels(reference : Path, output : Path):
    with h5py.File(str(reference)+'.h5','r') as refh5, h5py.File(str(output)+'.h5', 'r') as outh5:
        assert set(refh5.keys()) == set(outh5.keys())
        assert all(isinstance(item, h5py.Dataset) for item in refh5.values())
        assert_sequence_starts_ok(outh5['sequence_starts'][...], len(refh5['images']))
        for k, ref in refh5.items():
            out = outh5[k]
            assert ref.dtype == out.dtype
            if ref.dtype in (np.float32, np.float64):
                np.testing.assert_allclose(out[...], ref[...], err_msg=f"Mismatch at dataset {k}")
            else:
                np.testing.assert_array_equal(out[...], ref[...], err_msg=f"Mismatch at dataset {k}")
 

def assert_similar(reference : Path, output : Path):
    assert_similar_labels(reference, output)
    assert_similar_images(reference, output)


def test_300wlp_reproduction(tmpdir):
    tmpdir = Path(tmpdir)
    main(Path(__file__).parent/'data/300wlp_miniset.zip', tmpdir/'output.h5', 1<<32, enable_vis=False, angle_step=5., prob_closed_eyes=0, prob_spotlight=0)
    assert_similar(Path(__file__).parent/'data/300wlp_reference_output', tmpdir/'output')



if __name__ == '__main__':
    test_300wlp_reproduction(Path('/tmp/debug'))