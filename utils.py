import numpy as np
import cv2


def circ_aug(img, th=None, x=None, y=None, zoom_coef=2 / 3, crop_center=False):
    """
    Extracts rough breech face region of interest
    """

    def mask_circle(img):
        circle_mask = np.zeros(img.shape[:2], dtype="uint8")
        #        circle_mask = np.full(img.shape[:2], 255, dtype="uint8")

        c_x = roi.shape[1] / 2
        c_y = roi.shape[0] / 2

        if th is not None:
            x_th = np.random.randint(-1 * th, th + 1)
            y_th = np.random.randint(-1 * th, th + 1)
            r_th = np.random.randint(-1 * th, th + 1)
        else:
            x_th = 0
            y_th = 0
            r_th = 0

        center = (round(c_x + x_th), round(c_y + y_th))
        radius = np.min((round(img.shape[1] / 2), round(img.shape[0] / 2)))
        radius += r_th
        cv2.circle(circle_mask, center, radius, 255, -1)
        roi_masked = cv2.bitwise_and(img, img, mask=circle_mask)

        if crop_center:
            circle_mask = np.ones(img.shape[:2], dtype="uint8")
            cv2.circle(circle_mask, center, radius * 5 // 9, 0, -1)
            roi_masked = cv2.bitwise_and(roi_masked, roi_masked, mask=circle_mask)
        return roi_masked

    if x is not None and y is not None:
        roi_radius = int(min(img.shape[0] / 2, img.shape[1] / 2) * zoom_coef)
        start_r = round(y - roi_radius)
        end_r = round(y + roi_radius)
        start_c = round(x - roi_radius)
        end_c = round(x + roi_radius)

    else:
        roi_radius = int(min(img.shape[0] / 2, img.shape[1] / 2) * zoom_coef)
        start_r = round(img.shape[0] / 2 - roi_radius)
        end_r = round(img.shape[0] / 2 + roi_radius)
        start_c = round(img.shape[1] / 2 - roi_radius)
        end_c = round(img.shape[1] / 2 + roi_radius)

    roi = img[start_r:end_r, start_c:end_c]

    roi_masked = mask_circle(roi)
    return roi_masked


def crop_to_size(img, x, y, contrast, black_boost, light, roi_radius=672):
    """Crops BF from grayscaled shell scan using x,y center coordinates and applies contrast, blackboost and light factors
    Parameters
    ----------
    img : numpy 2d array
        Source image as numpy array
    x : int
        vertical center coordinate
    y : int
        horizontal center coordinate
    contrast : float
        contrast factor
    black_boost : float
        black boost param
    light : float
        light param
    roi_radius : int, optional
        radius of BF circle in pixels
    Returns
    -------
    numpy 2d array
        Cropped grayscaled BF with applied preprocessing
    """

    contrast *= 5.2


    std = np.zeros(33 + 1, dtype="float")
    avg = np.zeros(33 + 1, dtype="float")

    imag1 = img[(y - roi_radius):(y + roi_radius), (x - roi_radius):(x + roi_radius)].copy()

    center_x, center_y = np.array(imag1.shape) / 2
    len_n, len_m = np.array(imag1.shape) - 1

    N = np.array([np.full(len_n + 1, x) for x in np.arange(len_n + 1)])
    M = np.array([np.full(len_n + 1, x) for x in np.arange(len_n + 1)]).T
    img_sqrt = np.sqrt(np.power((N - center_x), 2) + np.power((M - center_y), 2))
    imag1[img_sqrt > roi_radius] = 0
    img_floor = np.floor(img_sqrt / 20).astype('int')

    cnt = np.bincount(img_floor[img_sqrt < roi_radius])

    for n in np.arange(24, 31):
        V = imag1[(img_floor == n) & (img_sqrt <= roi_radius)][:cnt[n]].astype('uint8')
        std[n] = np.std(V)
        avg[n] = np.mean(V)

    MIN = avg[24:31].min()
    INDX = 24 + avg[24:31].argmin()

    std_normalization = contrast / std[INDX]

    D = imag1 - MIN
    temp = np.zeros((len_n + 1, len_m + 1))
    std_K = np.full((len_n + 1, len_m + 1), std_normalization)
    K1_1 = black_boost - 1

    D_less_zero = D < 0
    std_K[D_less_zero] *= (K1_1 * ((D[D_less_zero] / MIN) ** 2 - 2 * D[D_less_zero] / MIN) + black_boost)
    temp = D * std_K + light
    temp[temp > 255] = 255
    temp[temp < 0] = 0
    return temp