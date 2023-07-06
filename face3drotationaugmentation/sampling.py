import numpy as np
from . import graphics

deg2rad = np.pi/180.

def get_euler_angle_bounds_for_offset(rot):
    h,p,b = map(lambda x: x/deg2rad, graphics.get_hpb(rot))
    h_limit_at_h_zero = 25.
    h_high_slope = 3.
    h_min = min(np.abs(h), -h_limit_at_h_zero + h_high_slope*min(np.abs(h),90.), 90.)
    h_max = min(np.abs(h)*h_high_slope+ h_limit_at_h_zero , 90.)
    if h < 0.:
        h_min, h_max = -h_max, -h_min
    # for zero yaw
    p_min = -45. + 2.*min(np.abs(p), 45.)
    p_max =  45.
    if p < 0.:
        p_min, p_max = -p_max, -p_min
    # for the current yaw, we can be more liberal because the more the face is 
    # viewed from the side, the more pitch becomes an in-plane rotation which 
    # imposes no visibility restrictions:
    mix = min(np.abs(h),90.)/90.
    p_min = -45*mix + (1.-mix)*p_min
    p_max =  45*mix + (1.-mix)*p_max
    # Roll is similar to pitch except the yaw-role is reversed
    b_min = -20 + 2.*min(np.abs(b), 20.)
    b_max =  20
    if b < 0.:
        b_min, b_max = -b_max, -b_min
    mix = min(np.abs(h),90.)/90.
    b_min = -20.*(1.-mix) + mix*b_min
    b_max =  20.*(1.-mix) + mix*b_max
    return (h_min*deg2rad, h_max*deg2rad), (p_min*deg2rad, p_max*deg2rad), (b_min*deg2rad, b_max*deg2rad)


def compute_heading_sample(h, hinterval, n):
    hmin, hmax = hinterval
    hmin, hmax = hmin/deg2rad, hmax/deg2rad
    h = np.clip(h/deg2rad,-90.,90.)
    width = hmax-hmin
    gaussian_probability = np.clip((width-90.)/90., 0., 1.)
    samples = np.random.exponential(scale=0.25, size=(n,))*width*np.sign(h) + h
    mask = np.random.binomial(1, p=gaussian_probability, size=(n,)).astype(np.bool8)
    samples[mask] = np.random.normal(scale=20.,size=(np.count_nonzero(mask),))
    samples = np.clip(samples, -100., 100.)
    samples *= deg2rad
    return samples


def sample_more_face_params(rot, n):
    h,p,b = graphics.get_hpb(rot)
    hinterval, pinterval, rinterval = get_euler_angle_bounds_for_offset(rot)
    low, high = zip(hinterval, pinterval, rinterval)
    low, high = map(np.asarray, (low, high))
    #hpb = np.random.uniform(low = low, high=high, size=(n,3))
    hpb = np.clip(np.random.normal(size=(n,3))*(high-low)[None,:]*0.25 + 0.5*(high+low)[None,:], low[None,:], high[None,:])
    #hpb[:,0] = compute_heading_sample(h, hinterval, n)
    return graphics.make_rot(hpb)