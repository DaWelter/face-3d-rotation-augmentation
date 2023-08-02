import numpy as np
from scipy.spatial.transform import Rotation
from . import graphics

deg2rad = np.pi/180.

def clipped_normal(rng, min, max, scale_multi=0.5, *args, **kwargs):
    return np.clip(rng.normal(0.5*(min+max), scale_multi*0.5*(max-min), *args, **kwargs), min, max)




def get_h_samples(h, p, b, rng, stepsize):
    h_min = np.abs(h)
    h_max = 90.
    num_bins = max(1, round((h_max-h_min)/stepsize))
    actual_spacing = (h_max-h_min)/num_bins
    points = np.linspace(h_min, h_max, num=num_bins, endpoint=False)
    points = points + actual_spacing * rng.uniform(0., 1., num_bins)
    return np.sign(h)*points

def get_p_bounds(h_samples, p):
    n = len(h_samples)
    # Lower range as heading tends to 90 deg. This is because
    # Pitching the head in profile view can be created easily by
    # in-plane rotation.
    normalized_h = np.clip(np.abs(h_samples) / 90., 0., 1.)
    range = 1. + (1. - normalized_h) * 10.
    center = (1. - normalized_h) * p
    p_min = center - range
    p_max = center + range
    np.testing.assert_array_less(p_min, p_max)
    return p_min, p_max


def get_b_bounds(h_samples, b):
    n = len(h_samples)
    # Similar to pitching
    normalized_h = np.clip(np.abs(h_samples) / 90., 0., 1.)
    range = 1. + np.power(normalized_h, 0.25) * 10.
    center = normalized_h * b
    b_min = center - range
    b_max = center + range
    np.testing.assert_array_less(b_min, b_max)
    return b_min, b_max


def sample_more_face_params(rot : Rotation, rng : np.random.RandomState, angle_step):
    '''
    Args:
        angle_step : In degrees
    '''
    h,p,b = (1./deg2rad)*graphics.get_hpb(rot)
    
    #hp_distribution = lambda a,b: clipped_normal(rng, a, b, 0.25)
    #hp_distribution = lambda a,b: rng.uniform(a, b)
    hp_distribution = lambda a,b: rng.normal(loc=0.5*(b+a), scale=0.5*(b-a))

    hsamples = get_h_samples(h,p,b, rng, stepsize=angle_step)
    psamples = hp_distribution(*get_p_bounds(hsamples, p))
    bsamples = hp_distribution(*get_b_bounds(hsamples, b))
    hpb = np.stack([hsamples,psamples,bsamples],axis=-1)
    hpb = np.concatenate([np.asarray([[h,p,b]]), hpb])
    hpb = deg2rad*np.clip(hpb, np.array([[-90.,-60.,-60]]), np.array([[90.,60.,60]]))
    
    return graphics.make_rot(hpb)