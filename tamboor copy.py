
import sys

import matplotlib.pyplot as plt
import numpy as np
# from skimage import data, io
# from skimage import exposure
# from skimage.transform import match_histograms
from imutils.perspective import four_point_transform
import cv2
from cv2 import aruco

# constants

# images
ref_card = None
input_card = None

# dims
img_height = 0
img_width = 0
card_height = 11
card_width = 10

# sample hole
sample_idx = (9, 1)
sample_color = 0
num_of_closest_colors = 10


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


def find_closest_colors_on_card():
    colors_distances = {}
    for row in range(1, card_height - 2):
        for col in range(1, card_width - 1):
            current_color = sample_rgb_of_square(input_card, row, col)
            # TODO change distance formula
            # distance = np.subtract(sample_color, current_color)
            # distance_abs = np.absolute(distance)
            # distance_num = np.sum(distance_abs)
            distance_between_colors = distance_2(sample_color, current_color)
            distance_num = np.sum(distance_between_colors)
            colors_distances[f"{row},{col}"] = (current_color, distance_num)

    return sorted(colors_distances.items(), key=lambda kv: kv[1][1])[:num_of_closest_colors]

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


def distance_2(a, b):
    return np.linalg.norm(a - b)


if __name__ == '__main__':
    # load images
    input_img = cv2.imread("card - save2.png")
    reference_img = cv2.imread("tests\\ref.png")
    # TODO add markers support
    # input_card = find_color_card(input_img)
    # reference_card = find_color_card(reference_img)
    input_card = input_img
    reference_card = reference_img

    # gray = cv2.cvtColor(input_card, cv2.COLOR_BGR2GRAY)
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    # parameters = aruco.DetectorParameters_create()
    # corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # frame_markers = aruco.drawDetectedMarkers(input_card.copy(), corners, ids)
    # print(rejectedImgPoints[1])

    # image resize
    print('Original Dimensions : ', input_card.shape)
    scale_percent = 100  # percent of original size
    img_width = int(input_card.shape[1] * scale_percent / 100)
    img_height = int(input_card.shape[0] * scale_percent / 100)
    dim = (img_width, img_height)

    # resize image
    input_card = cv2.resize(input_card, dim, interpolation=cv2.INTER_AREA)
    input_card = cv2.GaussianBlur(input_card,(5,5),0)

    # gaussian blur
    wb = cv2.xphoto.createGrayworldWB()

    cv2.imshow("Input Color Card", input_card)
    cv2.waitKey(0)

    # white balance
    wb.setSaturationThreshold(0.99)
    input_card = wb.balanceWhite(input_card)
    print('Resized Dimensions : ', input_card.shape)

    cv2.imshow("Input Color Card2", input_card)
    cv2.waitKey(0)

    # get sampled color
    sample_color = sample_rgb_of_square(input_card, sample_idx[0], sample_idx[1])
    print('sampled color: ', sample_color)

    # prepare lists for ARYEH
    input_closest_colors = []
    target_closest_colors = []
    indices_and_rgbs = (find_closest_colors_on_card())
    for index_rgb in indices_and_rgbs:
        input_closest_colors.append(index_rgb[1][0])
        # extract coords from key
        x = index_rgb[0].split(sep=',')[0]
        y = index_rgb[0].split(sep=',')[1]
        rgb_color = sample_rgb_of_square(reference_card, int(x), int(y))
        target_closest_colors.append(rgb_color)
    input_closest_colors = np.array(input_closest_colors)
    target_closest_colors = np.array(target_closest_colors)

    print(f"input {input_closest_colors}")
    print(f"target {target_closest_colors}")

    delta = np.array([1,2,3])
    from get_colour import shepards_interpolation
    delta = shepards_interpolation(input_closest_colors, target_closest_colors, sample_color)

    # calculate new color RGB
    new_color = np.add(sample_color, delta)
    print(new_color)
    
    from closest_colours import get_closest_colours
    
    closest_colours = get_closest_colours(new_color)
    print(closest_colours['RGB'])
    cv2.imshow("Input Color Card", input_card)
    cv2.waitKey(0)
