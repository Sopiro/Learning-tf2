# from __future__ import print_function
import binascii
import numpy as np
from scipy.cluster.vq import *
import imageio
import image_to_numpy
import cv2
import colorsys

NUM_CLUSTERS = 5

print('reading image')
img = image_to_numpy.load_image_file('C:/Users/Sopiro/Desktop/20200825/3.jpg')

# img = cv2.resize(img, (160, 160))
ar = np.asarray(img)
shape = ar.shape

# Collapse image to rgb array
ar = np.reshape(ar, newshape=(-1, ar.shape[-1])).astype(float)

# print('finding clusters')
codes, dist = kmeans(ar, NUM_CLUSTERS)

# print('cluster centres:\n', codes)

vecs, dist = vq(ar, codes)  # assign codes

counts, bins = np.histogram(vecs, len(codes))  # count occurrences

index_max = np.argmax(counts)  # find most frequent
peak = codes[index_max]

color = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
print('most frequent is %s (#%s)' % (peak, color))

peak = peak / 255.0

hsv_color = colorsys.rgb_to_hsv(peak[0], peak[1], peak[2])

print(hsv_color)

c = ar.copy()
for i, code in enumerate(codes):
    c[np.r_[np.where(vecs == i)], :] = code
imageio.imwrite('clusters.png', c.reshape(*shape).astype(np.uint8))
print('saved clustered image')
