import numpy as np
from scipy.spatial.transform import Rotation
from . import graphics

deg2rad = np.pi/180.

def clipped_normal(rng, min, max, scale_multi=0.5, *args, **kwargs):
    return np.clip(rng.normal(0.5*(min+max), scale_multi*0.5*(max-min), *args, **kwargs), min, max)

def get_h_samples(h, p, b, rng):
    h_min = np.abs(h)
    h_max = 90.
    stepsize = 5.
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
    range = 1. + (1. - np.clip(h_samples / 90., 0., 1.)) * 10.
    p_min = np.full((n,),p) - range
    p_max = np.full((n,),p) + range
    np.testing.assert_array_less(p_min, p_max)
    return p_min, p_max


def get_b_bounds(h_samples, b):
    n = len(h_samples)
    # Similar to pitching
    range = 1. + np.clip(h_samples / 90., 0., 1.) * 20.
    b_min = np.full((n,),b) - range
    b_max = np.full((n,),b) + range
    np.testing.assert_array_less(b_min, b_max)
    return b_min, b_max


def sample_more_face_params(rot : Rotation, rng):
    h,p,b = (1./deg2rad)*graphics.get_hpb(rot)
    #b,h,p = (1./deg2rad)*rot.as_euler('zyx')
    hsamples = get_h_samples(h,p,b, rng)
    psamples = clipped_normal(rng, *get_p_bounds(hsamples, p), 0.25)
    bsamples = clipped_normal(rng, *get_b_bounds(hsamples, b), 0.25)
    hpb = np.stack([hsamples,psamples,bsamples],axis=-1)
    hpb = np.concatenate([np.asarray([[h,p,b]]), hpb])
    hpb = deg2rad*np.clip(hpb, np.array([[-90.,-60.,-60]]), np.array([[90.,60.,60]]))
    #return Rotation.from_euler('zyx', hpb)
    return graphics.make_rot(hpb)