import cv2 as cv
from text_extracter import TextExtracter

path = '../data/tv_avis_test.mp4'
cap = cv.VideoCapture(path)
td = TextExtracter(cap)
td.process_video()
cap.release()
cv.destroyAllWindows()
