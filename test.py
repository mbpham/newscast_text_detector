import cv2 as cv
import pytesseract
from TextExtracter import TextExtracter
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
from copy import copy


def sort_points(points):
    '''
    points :  assumes four points
    returns upper left, upper right, lower left, lower right
    '''
    points = points.reshape(4, 2)
    points = points[points[:, 0].argsort()]

    upper = points[:2, :]
    ## upper left, upper right
    upper = upper[upper[:, 1].argsort()]

    lower = points[2:, :]
    ## lower left, lower right
    lower = lower[lower[:, 1].argsort()]
    ret = np.array([upper[0], upper[1], lower[0], lower[1]])
    return ret

def read_text(frame, upper_left, lower_right, resize, debug=False):
    new_frame = frame[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
    new_frame = cv.resize(new_frame, (new_frame.shape[1]*resize, new_frame.shape[0]*resize), interpolation = cv.INTER_AREA)
    text = pytesseract.image_to_string(new_frame, lang='dan')
    print(text)
    return text


test = 0

if test:

    frame1 = cv.imread("../data/t3.png")
    frame1 = cv.resize(frame1, (640, 360), interpolation = cv.INTER_AREA)
    frame  = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)

    thresh = cv.inRange(frame[:, :, 0], 88, 110)

    kernel = np.ones((3,3),np.uint8)
    plt.imshow(thresh, cmap='gray')
    plt.imsave("hsv_test21_bin.png", thresh, cmap='gray')
    #plt.show()

    #thresh = cv.dilate(thresh,kernel,iterations = 1)
    # thresh = cv.erode(thresh,kernel,iterations = 3)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    contours_ = []
    for contour in contours:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.004 * peri, True)
        if len(approx) == 4:
            min_area = 20000
            max_area = 30000
            sortedp = sort_points(approx)
            upper_line = np.linalg.norm(sortedp[0]-sortedp[1])
            left_line = np.linalg.norm(sortedp[0]-sortedp[2])
            box_area = upper_line * left_line
            if box_area > min_area and box_area < max_area:
                try:
                    read_text(frame1, (sortedp[0][0], sortedp[0][1]), (sortedp[3][0], sortedp[3][1]), resize=1)
                    contours_.append(approx)
                except Exception:
                    pass

    cv.drawContours(frame1, contours_, -1, (0,255,0), 1)

    cv.imshow('frame', frame1)
    #cv.imwrite("hsv_test2_contours.png", frame1)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    path = '../data/tv_avis_test.mp4'
    cap = cv.VideoCapture(path)
    #cap.set(0, 1120000)
    td = TextExtracter(cap)
    td.process_video()
    cap.release()
    cv.destroyAllWindows()
