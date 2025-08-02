from pathlib import Path
from PIL import Image
import numpy as np
import h5py
from collections import defaultdict
import scipy.io
from contextlib import closing
import zipfile
import pytest
import io
from typing import Any
import numpy.testing

from face3drotationaugmentation.common import UInt8Array, FloatArray
from face3drotationaugmentation.datasetwriter import DatasetWriter300WLPLike
from face3drotationaugmentation.dataset300wlp import Dataset300WLP, imdecode

TESTFILE = Path(__file__).parent / 'data' / '300wlp_miniset.zip'


@pytest.fixture()
def first_sample():
    zf = zipfile.ZipFile(TESTFILE)
    with io.BytesIO(zf.read('300W_LP/AFW/AFW_1051618982_1_0.mat')) as f:
        data = scipy.io.loadmat(f)
    img = imdecode(zf.read('300W_LP/AFW/AFW_1051618982_1_0.jpg'))
    return data, img


def test_DatasetWriter300WLPLike(first_sample: tuple[dict[str, Any], UInt8Array], tmpdir: Path):
    '''Tests DatasetWriter300WLPLike

    Reads a sample from a reference files, writes it and compares the output with the original.
    '''
    with closing(Dataset300WLP(TESTFILE)) as ds:
        name, sample = next(iter(ds))

    # Hope this is deterministic
    assert name == 'AFW_1051618982_1_0', "Ensure that the reference sample is returned."

    with closing(DatasetWriter300WLPLike(tmpdir / 'ds')) as dw:
        dw.write(name, sample)
        dw.write(name, sample)

    files = {p.name for p in Path(tmpdir / 'ds').iterdir()}
    assert files == {
        'AFW_1051618982_1_0_0.jpg',
        'AFW_1051618982_1_0_0.mat',
        'AFW_1051618982_1_0_1.mat',
        'AFW_1051618982_1_0_1.jpg',
    }

    with (tmpdir / 'ds' / 'AFW_1051618982_1_0_0.mat').open('rb') as f:
        data = scipy.io.loadmat(f)

    refdata, refimage = first_sample

    pose_para = refdata['Pose_Para']
    pose_para[0, 5] = 0  # Zero out z-coord. This is currently not preserved,
    exp_para = refdata['Exp_Para']
    exp_para[10:, :] = 0.0  # Because only the first few entries were used.
    shp_para = refdata['Shape_Para']
    shp_para[40:, :] = 0.0
    numpy.testing.assert_allclose(data['Pose_Para'], pose_para)
    numpy.testing.assert_allclose(data['Exp_Para'], exp_para)
    numpy.testing.assert_allclose(data['Shape_Para'], shp_para)

    with (tmpdir / 'ds' / 'AFW_1051618982_1_0_0.jpg').open('rb') as f:
        image = imdecode(f.read())
    numpy.testing.assert_allclose(image, refimage, atol=20)  # Some error due to re-encoding is allowed.
