import os.path
import numpy as np

from scipy import ndimage

FRAME_SIZE = 84

########################################################################
class Frame():
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0).reshape(4,FRAME_SIZE,FRAME_SIZE)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def as_bytes(self):
        return b"".join(self._frames)

    @staticmethod
    def halve_image(image) :
        rows, cols = image.shape
        image = image.astype("uint16")
        image = image.reshape(rows // 2, 2, cols // 2, 2)
        image = image.sum(axis=3).sum(axis=1)
        return ((image + 2) >> 2).astype("uint8").reshape(rows//2,cols//2)

    @staticmethod
    def Downsample(frame, cut=None, filter_image=False):
        # expected frame size is 352x312 (including border).
        if len(frame) != 352*312: raise Exception("Unexpected size")
        img = np.frombuffer(frame, dtype=np.uint8).reshape(312,352)

        if cut != None:
            u,d,l,r = cut
            img = img[u:d,l:r] # cut center part of img

        img = Frame.halve_image(img)
        if filter_image:
            img = ndimage.median_filter(img, (2,2))

        return bytearray(img.swapaxes(0,1)*32)

    @staticmethod
    def Join(*argv):
        lst = [x for x in argv]
        return Frame(lst)
