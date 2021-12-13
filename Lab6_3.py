import cv2,copy
import numpy as np
import os
from math import sqrt
import cmath
from matplotlib import pyplot as plt
img = cv2.imread('123.png')
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
F1 = np.fft.fft2((img).astype(float))
F2 = np.fft.fftshift(F1)
(w, h) = img.shape
half_w, half_h = int(w/2), int(h/2)
n = 10
F2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0
img1 = np.fft.ifft2(np.fft.ifftshift(F2)).real

cv2.imwrite('3.png',img1)