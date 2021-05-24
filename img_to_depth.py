import cv2
import pickle
import numpy as np
from scipy.fftpack import dst, idst


def fishye_calib(img, para):
    K, D, DIM = para
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

class ImageToDepth(object):

    def __init__(self):
        folder = "cam2"
        self.camera_parameter = pickle.load(open(folder + '/cam2_calib.pkl', 'rb'))
        self.lookup_table = pickle.load(open(folder + '/Lookuptable.pkl', 'rb'))
        self.border = self.lookup_table[9]
        frame0 = cv2.imread(folder +'/0.jpg')
        f0 = self._init_frame(fishye_calib(frame0, self.camera_parameter))
        # self.f0 = f0[101: 381, 189: 472, :]   # cam2
        self.f0 = f0[self.border[0]: self.border[1], self.border[2]: self.border[3], :]
        # self.pix_to_mm = self.lookup_table[7]
        self.pix_to_mm = 2.5*6/286

    def _init_frame(self, frame0):
        sigma = 50
        f0 = cv2.GaussianBlur(frame0, (99,99), sigma)
        height, width = frame0.shape[:2]
        frame0_ = frame0.astype('float32')
        dI = np.mean(f0 - frame0_, 2)
        mask = (dI < 5).astype('uint8')
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        f0 = f0 - f0 * mask * 0.85 + mask * frame0_ * 0.85
        f0 = f0.astype('uint8')
        return f0

    def _match_grd_bnz(self, LookupTable, dI, f0, f01=None, validmask=None):
        def fix1(num):
            num[num>1] = 1
            num[num<0] = 0
            return num

        if f01 is None:
            f0 = np.asfarray(f0)
            t = np.mean(f0)
            f01 = 1 + ((t / f0) - 1) * 2
        size1, size2 = np.shape(dI)[:2]
        img_grad_mag = np.zeros((size1, size2))
        img_grad_dir = np.zeros((size1, size2))
        dI = dI * f01
        binm = LookupTable[0] - 1

        if validmask is not None:
            sizet = size1 * size2
            sizet2 = sizet * 2
            validid = np.nonzero(validmask)
            r1 = dI(validid[0], validid[1], 0)
            g1 = dI(validid[0], validid[1], 1)
            b1 = dI(validid[0], validid[1], 2)
            r2 = (r1 - LookupTable[5]) / LookupTable[6]
            g2 = (g1 - LookupTable[5]) / LookupTable[6]
            b2 = (b1 - LookupTable[5]) / LookupTable[6]
            r2 = fix1(r2)
            g2 = fix1(g2)
            b2 = fix1(b2)
            r3 = (np.floor(r2 * binm)).astype(int)
            g3 = (np.floor(g2 * binm)).astype(int)
            b3 = (np.floor(b2 * binm)).astype(int)

            ind = (r3, g3, b3)
            img_grad_mag[validid] = LookupTable[1][ind]
            img_grad_dir[validid] = LookupTable[2][ind]

        else:
            r1 = dI[:, :, 0]
            g1 = dI[:, :, 1]
            b1 = dI[:, :, 2]
            r2 = (r1 - LookupTable[5]) / LookupTable[6]
            g2 = (g1 - LookupTable[5]) / LookupTable[6]
            b2 = (b1 - LookupTable[5]) / LookupTable[6]
            r2 = fix1(r2)
            g2 = fix1(g2)
            b2 = fix1(b2)
            r3 = (np.floor(r2 * binm)).astype(int)
            g3 = (np.floor(g2 * binm)).astype(int)
            b3 = (np.floor(b2 * binm)).astype(int)
            ind = (r3, g3, b3)
            img_grad_mag = LookupTable[1][ind]
            img_grad_dir = LookupTable[2][ind]

        img_grad_x = img_grad_mag * np.cos(img_grad_dir)
        img_grad_y = img_grad_mag * np.sin(img_grad_dir)

        return img_grad_x, img_grad_y, img_grad_mag, img_grad_dir

    def _fast_poisson(self, gx, gy):
        ydim, xdim = np.shape(gx)
        gxx = np.zeros((ydim, xdim))
        gyy = np.zeros((ydim, xdim))
        f = np.zeros((ydim, xdim))
        gyy[1:ydim, 0:xdim-1] = gy[1:ydim, 0:xdim-1]-gy[0:ydim-1, 0:xdim-1]
        gxx[0:ydim-1, 1:xdim] = gx[0:ydim-1, 1:xdim]-gx[0:ydim-1, 0:xdim-1]
        f = gxx + gyy
        height, width = f.shape[:2]
        f2 = f[1 : height - 1, 1: width - 1]
        tt = dst(f2.T, type=1).T /2
        f2sin = (dst(tt, type =1)/2)
        x, y = np.meshgrid(np.arange(1, xdim-1), np.arange(1, ydim-1))
        denom = (2*np.cos(np.pi * x/(xdim-1))-2) + (2*np.cos(np.pi*y/(ydim-1)) - 2)
        f3 = f2sin/denom
        tt = np.real(idst(f3, type=1, axis=0))/(f3.shape[0]+1)
        img_tt = (np.real(idst(tt.T, type=1, axis=0))/(tt.T.shape[0]+1)).T
        img_direct = np.zeros((ydim, xdim))
        height, width = img_direct.shape[:2]
        img_direct[1: height - 1, 1: width - 1] = img_tt
        return img_direct

    def convert(self, img):
        img = fishye_calib(img, self.camera_parameter)
        # initialize
        height, width = img.shape[:2]
        frame_ = img[self.border[0]: self.border[1], self.border[2]: self.border[3], :]
        I = np.asfarray(frame_, float) - self.f0
        ImGradX, ImGradY, ImGradMag, ImGradDir = self._match_grd_bnz(self.lookup_table, I, self.f0)
        hm = self._fast_poisson(ImGradX, ImGradY) * self.pix_to_mm
        height, width = hm.shape[:2]
        hm[hm < 0] = 0
        d_ptcd = np.zeros((height * width, 3))
        x = np.arange(width) * self.pix_to_mm
        y = np.arange(height) * self.pix_to_mm
        xgrid, ygrid = np.meshgrid(x, y)
        d_ptcd[:, 0] = xgrid.flatten()
        d_ptcd[:, 1] = ygrid.flatten()
        d_ptcd[:, 2] = hm.flatten()*20
        return d_ptcd, hm

if __name__ == "__main__":
    # result = pickle.load(open("result.pkl","rb"))
    # # print(result)
    # LookupTable = Lookuptable(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8])
    # pickle.dump(LookupTable, open("lookup_table.pkl", "wb"))
    # lookup_table = pickle.load(open('lookup_table.pkl', 'rb'))
    # print(lookup_table)
    pass