from scipy.spatial.transform import Rotation
import numpy as np
import cv2
from .common import AugmentedSample, UInt8Array

def draw_axis(img, rot, tdx=None, tdy=None, size = 100, brgt = 255, lw=3):
    rot = Rotation.as_matrix(rot)

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    rot = size*rot
    x1, x2, x3 = rot[0,:] + tdx
    y1, y2, y3 = rot[1,:] + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(brgt,0,0),lw)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,brgt,0),lw)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(0,0,brgt),lw)

    return img


def draw_points3d(img, pt3d, labels=True, brightness=255, color=None):
    assert pt3d.shape[-1] in (2,3)
    if color is not None:
        r, g, b = color
    else:
        g = brightness
        b = r = brightness//2
    for i, p in enumerate(pt3d[:,:2]):
        p = tuple(p.astype(int))
        if labels:
            cv2.putText(img, str(i), (p[0]+2,p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(img, p, 2 if i==0 else 1, (r,g,b), -1)


def draw_roi(img, roi, color, linewidth):
    cv2.rectangle(img, (round(roi[0]),round(roi[1])), (round(roi[2]),round(roi[3])), color, linewidth)


def draw_pose(img : UInt8Array, sample : AugmentedSample, brightness : int, linewidth: int) -> None:
    rot = sample.rot
    xy = sample.xy
    draw_axis(img, rot, tdx = xy[0], tdy = xy[1], brgt=brightness, lw=linewidth)
    if sample.scale <= 0.:
        print (f"Error, head size {sample.scale} not positive!")
        print (sample)
    else:
        cv2.circle(img, (int(xy[0]),int(xy[1])), int(sample.scale), (brightness,brightness,0), linewidth)
    if sample.pt3d_68 is not None:
        draw_points3d(img, sample.pt3d_68, False, brightness, (255,255,255))
    if sample.roi is not None:
        draw_roi(img, sample.roi, 255, 2)