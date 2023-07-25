from pathlib import Path
from PIL import Image
import numpy as np
import h5py

from expand_dataset import main


def assert_similar_images(reference : Path, output : Path):
    reference_images = list(x.relative_to(reference) for x in reference.glob("*"))
    output_images = list(x.relative_to(output) for x in output.glob("*"))
    assert set(reference_images) == set(output_images)
    for name in reference_images:
        ref = np.asarray(Image.open(reference / name))
        img = np.asarray(Image.open(output / name))
        np.testing.assert_allclose(img, ref, atol=5)

def assert_similar_labels(reference : Path, output : Path):
    with h5py.File(str(reference)+'.h5','r') as refh5, h5py.File(str(output)+'.h5', 'r') as outh5:
        assert set(refh5.keys()) == set(outh5.keys())
        assert all(isinstance(item, h5py.Dataset) for item in outh5.values())
        for k, ref in refh5.items():
            out = outh5[k]
            assert ref.dtype == out.dtype
            if ref.dtype in (np.float32, np.float64):
                np.testing.assert_allclose(out[...], ref[...])
            else:
                np.testing.assert_array_equal(out[...], ref[...])


def assert_similar(reference : Path, output : Path):
    assert_similar_labels(reference, output)
    assert_similar_images(reference, output)


def test_300wlp_reproduction(tmpdir):
    tmpdir = Path(tmpdir)
    main(Path(__file__).parent/'data/300wlp_miniset.zip', tmpdir/'output.h5', 1<<32, enable_vis=False)
    assert_similar(Path(__file__).parent/'data/300wlp_reference_output', tmpdir/'output')



if __name__ == '__main__':
    test_300wlp_reproduction(Path('/tmp/debug'))