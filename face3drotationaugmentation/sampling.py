import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from . import graphics
from typing import Optional

from .common import deg2rad, FloatArray


def clipped_normal(rng, min, max, scale_multi=0.5, *args, **kwargs):
    return np.clip(rng.normal(0.5 * (min + max), scale_multi * 0.5 * (max - min), *args, **kwargs), min, max)


def get_h_samples(h, p, b, rng, stepsize):
    h_min = np.abs(h)
    h_max = 90.0
    num_bins = max(1, round((h_max - h_min) / stepsize))
    actual_spacing = (h_max - h_min) / num_bins
    points = np.linspace(h_min, h_max, num=num_bins, endpoint=False)
    points = points + actual_spacing * rng.uniform(0.0, 1.0, num_bins)
    return np.sign(h) * points


def get_p_bounds(h_samples, p):
    n = len(h_samples)
    # Lower range as heading tends to 90 deg. This is because
    # Pitching the head in profile view can be created easily by
    # in-plane rotation.
    normalized_h = np.clip(np.abs(h_samples) / 90.0, 0.0, 1.0)
    range = 1.0 + (1.0 - normalized_h) * 10.0
    center = (1.0 - normalized_h) * p
    p_min = center - range
    p_max = center + range
    np.testing.assert_array_less(p_min, p_max)
    return p_min, p_max


def get_b_bounds(h_samples, b):
    n = len(h_samples)
    # Similar to pitching
    normalized_h = np.clip(np.abs(h_samples) / 90.0, 0.0, 1.0)
    range = 1.0 + np.power(normalized_h, 0.25) * 10.0
    center = normalized_h * b
    b_min = center - range
    b_max = center + range
    np.testing.assert_array_less(b_min, b_max)
    return b_min, b_max


def sample_more_face_params(rot: Rotation, rng: np.random.RandomState, angle_step: float) -> Rotation:
    '''
    Args:
        angle_step : In degrees
    '''
    h, p, b = (1.0 / deg2rad) * graphics.get_hpb(rot)

    # hp_distribution = lambda a,b: clipped_normal(rng, a, b, 0.25)
    # hp_distribution = lambda a,b: rng.uniform(a, b)
    hp_distribution = lambda a, b: rng.normal(loc=0.5 * (b + a), scale=0.5 * (b - a))

    hsamples = get_h_samples(h, p, b, rng, stepsize=angle_step)
    psamples = hp_distribution(*get_p_bounds(hsamples, p))
    bsamples = hp_distribution(*get_b_bounds(hsamples, b))
    hpb = np.stack([hsamples, psamples, bsamples], axis=-1)
    hpb = np.concatenate([np.asarray([[h, p, b]]), hpb])  # <------- WARNING: Always adding the original rotation, too.
    hpb = deg2rad * np.clip(hpb, np.array([[-90.0, -60.0, -60]]), np.array([[90.0, 60.0, 60]]))

    return graphics.make_rot(hpb)


def sample_shapeparams(rng: np.random.RandomState, original_params: FloatArray, n: int, prob_closed_eyes: float):
    new_shapeparams = np.broadcast_to(original_params[None,], (n,) + original_params.shape)
    if prob_closed_eyes > 0:
        eyes_closed = rng.binomial(1, p=prob_closed_eyes, size=(n, 1))
        eyes_closing_amount = eyes_closed * rng.beta(5, 1.0, size=(n, 1))
        eyes_closing_amount = np.repeat(eyes_closing_amount, 2, axis=-1)
    else:
        eyes_closing_amount = np.zeros((n, 2))
    return new_shapeparams, eyes_closing_amount


def sample_light(rng: np.random.RandomState, rot: Rotation, prob_spotlight: float) -> Optional[npt.NDArray]:
    assert 0 <= prob_spotlight <= 1.0
    h, p, b = (1.0 / deg2rad) * graphics.get_hpb(rot)
    p = prob_spotlight
    if p <= 0:
        return None
    # prob_on_modulation = np.clip(1. - np.abs(h) / 90., 0.1, 1.)
    # p = prob_on_max * prob_on_modulation
    if rng.binomial(1, p):
        z = 1
        y = rng.uniform(-1.0, 1.0)
        x = np.sign(h) * 10.0
        return np.array([x, y, z])
    else:
        return None
