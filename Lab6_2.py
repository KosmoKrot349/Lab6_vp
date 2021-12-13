import cv2,copy
import numpy as np
import os
from math import sqrt
import cmath
from matplotlib import pyplot as plt
img = cv2.imread('123.png')
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

r = 30
ham = np.hamming(img.shape[0])[:,None]
ham1 = np.hamming(img.shape[1])[:,None]
ham2d = np.sqrt(np.dot(ham, ham1.T)) ** r
f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
f_shifted = np.fft.fftshift(f)
f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
f_filtered = f_complex*ham2d
f_filtered_shifted = np.fft.fftshift(f_filtered)
inv_img = np.fft.ifft2(f_filtered_shifted)
filtered_img = np.abs(inv_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('2.png',np.real(filtered_img))