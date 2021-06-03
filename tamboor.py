"""
==================
Histogram matching
==================

This example demonstrates the feature of histogram matching. It manipulates the
pixels of an input image so that its histogram matches the histogram of the
reference image. If the images have multiple channels, the matching is done
independently for each channel, as long as the number of channels is equal in
the input image and the reference.

Histogram matching can be used as a lightweight normalisation for image
processing, such as feature matching, especially in circumstances where the
images have been taken from different sources or in different conditions (i.e.
lighting).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, io
from skimage import exposure
from skimage.transform import match_histograms
from imutils.perspective import four_point_transform
import cv2

# constants
img_height = 0
img_width = 0
card_height = 9
card_width = 8
sample_idx = (8, 4.5)
num_of_closest_colors = 4


def sample_rgb_of_square(img, square_x, square_y):
    cube_height = img_height / card_height
    cube_width = img_width / card_width
    return img[int((square_x * cube_height) + cube_height / 2),
               int((square_y * cube_width) + cube_width / 2)][::-1]

def find_color_card(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and
    # detect the markers in the input image
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    # try to extract the coordinates of the color correction card
    try:
        # otherwise, we've found the four ArUco markers, so we can
        # continue by flattening the ArUco IDs list
        ids = ids.flatten()
        # extract the top-left marker
        i = np.squeeze(np.where(ids == 923))
        topLeft = np.squeeze(corners[i])[0]
        # extract the top-right marker
        i = np.squeeze(np.where(ids == 1001))
        topRight = np.squeeze(corners[i])[1]
        # extract the bottom-right marker
        i = np.squeeze(np.where(ids == 241))
        bottomRight = np.squeeze(corners[i])[2]
        # extract the bottom-left marker
        i = np.squeeze(np.where(ids == 1007))
        bottomLeft = np.squeeze(corners[i])[3]
    # we could not find color correction card, so gracefully return
    except:
        print("[INFO] could not find color matching card in both images")
        sys.exit(0)

    # build our list of reference points and apply a perspective
    # transform to obtain a top-down, bird’s-eye view of the color
    # matching card
    cardCoords = np.array([topLeft, topRight,
                           bottomRight, bottomLeft])
    card = four_point_transform(image, cardCoords)
    # return the color matching card to the calling function
    return card

# def find_closest_colors_on_card(img):


# matched = match_histograms(input_card, reference_card, multichannel=True)
#
# # show the color matching card in the reference image and input image,
# # respectively
# cv2.imshow("Reference Color Card", reference_card)
# cv2.imshow("Input Color Card", input_card)
# # apply histogram matching from the color matching card in the
# # reference image to the color matching card in the input image
# print("[INFO] matching images...")
# imageCard = match_histograms(input_card, reference_card, multichannel=True)
# # show our input color matching card after histogram matching
# cv2.imshow("Input Color Card After Matching", imageCard)
# cv2.waitKey(0)

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
#                                     sharex=True, sharey=True)
# for aa in (ax1, ax2, ax3):
#     aa.set_axis_off()
#
# ax1.imshow(input_card)
# ax1.set_title('Source')
# ax2.imshow(reference_card)
# ax2.set_title('Reference')
# ax3.imshow(matched)
# ax3.set_title('Matched')
#
# plt.tight_layout()
# plt.show()
#
# ######################################################################
# # To illustrate the effect of the histogram matching, we plot for each
# # RGB channel, the histogram and the cumulative histogram. Clearly,
# # the matched image has the same cumulative histogram as the reference
# # image for each channel.
#
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
#
# for i, img in enumerate((input_img, reference_img, matched)):
#     for c, c_color in enumerate(('red', 'green', 'blue')):
#         img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
#         axes[c, i].plot(bins, img_hist / img_hist.max())
#         img_cdf, bins = exposure.cumulative_distribution(img[..., c])
#         axes[c, i].plot(bins, img_cdf)
#         axes[c, 0].set_ylabel(c_color)
#
# axes[0, 0].set_title('Source')
# axes[0, 1].set_title('Reference')
# axes[0, 2].set_title('Matched')
#
# plt.tight_layout()
# plt.show()

if __name__ == '__main__':
    # load images
    input_img = cv2.imread("card rabash.png")
    reference_img = cv2.imread("card rabash.png")
    # input_card = find_color_card(input_img)
    input_card = input_img
    # reference_card = find_color_card(reference_img)
    reference_card = reference_img

    # image resize
    print('Original Dimensions : ', input_card.shape)
    scale_percent = 100  # percent of original size
    img_width = int(input_card.shape[1] * scale_percent / 100)
    img_height = int(input_card.shape[0] * scale_percent / 100)
    dim = (img_width, img_height)

    # resize image
    resized = cv2.resize(input_card, dim, interpolation=cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)

    print(list(sample_rgb_of_square(resized, 7, 7)))
    print(type(sample_rgb_of_square(resized, 1, 1)))

    # show
    cv2.imshow("Input Color Card", resized)
    cv2.waitKey(0)
