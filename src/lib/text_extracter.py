import cv2 as cv
import pytesseract
import numpy as np
from text_detection import detect_text
from skimage.metrics import structural_similarity

class TextExtracter:
    def __init__(self, video=None, date=None):
        f = cv.imread('./direkte.png')
        ol = cv.imread('./omlidt.png')

        self.video = video
        self.net = cv.dnn.readNet("frozen_east_text_detection.pb")
        self.labels = []

        self.direkte = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
        self.direkte_coords = ((558, 14), (639, 38))
        self.omlidt = cv.cvtColor(ol, cv.COLOR_BGR2GRAY)
        self.omlidt_coords = ((530, 305), (573, 318))

        self.title, self.title_coords, self.current_title = None, None, None
        self.subject = None
        self.location = None

    def process_video(self, debug=False):
        if self.video is not None:
            cap = self.video
            frame_nr = 0
            text_found = False
            while(cap.isOpened()):
                np.save("labels", np.array(self.labels))
                title, subject, omlidt, direkte, kortnyt, location = None, None, False, False, False, None
                ret, frame = cap.read()
                title = self.find_title(frame)
                if title is not None:
                    text_found = True
                    subject = self.find_subject(frame)
                    if subject is not None:
                        omlidt, kortnyt = self.find_omlidt(frame)

                # find DIREKTE box and location
                direkte, location = self.find_direkte(frame)

                if direkte is not None:
                    text_found = True

                if text_found:
                    self.labels.append(np.array([frame_nr, title, subject, omlidt, direkte, kortnyt, location]))

                text_found = False
                cv.imshow('frame', frame)
                frame_nr += 1
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

    def find_title(self, frame):
        if self.title is None and self.title_coords is None:
            title = self.find_boxes(frame, save_frame='title')
            if title is None:
                title = self.find_boxes(frame, min_area=15000, max_area=30000, \
                                        thr_lower=160, thr_upper=185, \
                                        save_frame='title')
                return title

        else:
            upper_left = self.title_coords[0]
            lower_right = self.title_coords[1]
            text_area = frame[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]

            detected_text = cv.resize(text_area, (self.title.shape[1], self.title.shape[0]), interpolation = cv.INTER_AREA)
            detected_text = cv.cvtColor(detected_text, cv.COLOR_BGR2GRAY)
            (score, _) = structural_similarity(self.title, detected_text, full=True)

            if score > 0.8:
                title = self.current_title
            else:
                self.title = None
                self.title_coords = None
                self.current_title = None
                self.subject = None
                title = self.find_boxes(frame, save_frame='title')
            return title

    def find_subject(self, frame):
        if self.subject is None:
            subject = self.find_boxes(frame, min_area=5000, max_area=20000, \
                                        thr_lower=220, thr_upper=255, sub=True)
            if subject is not None:
                self.subject = subject
        else:
            subject = self.subject
        return subject

    def find_omlidt(self, frame):
        omlidt, kortnyt = None, None
        if self.omlidt is None:
            text = self.find_boxes(frame[200:, 200:], min_area=550, max_area=800, perimeter=0.04, \
                                    thr_lower=0, thr_upper=15, sub=3, dilation=1, resize=4)
            if text is not None:
                if text.replace(',', '').replace(' ', '').lower()  == "omlidt":
                    omlidt = True
                if text.replace(',', '').replace(' ', '').lower() == 'kortnyt':
                    kortnyt = True
        else:
            upper_left = self.omlidt_coords[0]
            lower_right = self.omlidt_coords[1]
            text_area = frame[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]

            detected_text = cv.resize(text_area, (self.omlidt.shape[1], self.omlidt.shape[0]), interpolation = cv.INTER_AREA)
            detected_text = cv.cvtColor(detected_text, cv.COLOR_BGR2GRAY)
            (score, diff) = structural_similarity(self.omlidt, detected_text, full=True)
            if score > 0.8:
                omlidt = True

        return omlidt, kortnyt

    def find_direkte(self, frame):
        direkte, location = None, None
        if self.direkte is None:
            black = self.find_boxes(frame[:50, 200:], min_area=1000, max_area=1200, perimeter=0.02, \
                                    thr_lower=0, thr_upper=20, sub=5, dilation=1, resize=2)
            if black == 'DIREKTE':
                text_found = True
                direkte = True
        else:
            upper_left = self.direkte_coords[0]
            lower_right = self.direkte_coords[1]
            new_frame = frame[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]

            B = cv.resize(new_frame, (self.direkte.shape[1], self.direkte.shape[0]), interpolation = cv.INTER_AREA)
            gray_B = cv.cvtColor(B, cv.COLOR_BGR2GRAY)
            (score, diff) = structural_similarity(self.direkte, gray_B, full=True)

            if score > 0.54:
                text_found = True
                direkte = True
                if self.location is None:
                    location = self.find_boxes(frame[:50, 200:], min_area=700, max_area=3000, perimeter=0.02, \
                                            thr_lower=0, thr_upper=230, erosion=2)
                else:
                    location = self.location
            else:
                if self.location is not None:
                    self.location = None

        return direkte, location

    def read_text(self, frame, upper_left, lower_right, resize=1, debug=False):
        new_frame = frame[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        if resize is not None:
            new_frame = cv.resize(new_frame, (new_frame.shape[1]*resize, new_frame.shape[0]*resize), interpolation = cv.INTER_AREA)

        text = pytesseract.image_to_string(new_frame, lang='dan')
        if text == "DIREKTE":
            gray_new = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
            # thresh = cv.inRange(gray, 0, 50)
            self.direkte = gray_new
            self.direkte_coords = (upper_left, lower_right)
            cv.imwrite("direkte.png", self.direkte)
        print(text)
        return text

    def find_boxes(self, frame, min_area=25000, max_area=30000, perimeter=0.004, \
                    thr_lower=140, thr_upper=195, sub=None, erosion=None, dilation=None, \
                    allow_multi=False, resize=None, save_frame=False):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        thresh = cv.inRange(gray, thr_lower, thr_upper)
        thresh = thresh if dilation is None else cv.dilate(thresh,np.ones((3,3),np.uint8),iterations = dilation)
        thresh = thresh if erosion is None else cv.erode(thresh, np.ones((3,3),np.uint8), iterations=erosion)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


        if contours is None:
            return
        else:
            contours_ = []
            text = []
            for contour in contours:
                peri = cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, perimeter * peri, True)
                if len(approx) == 4:
                    sortedp = self.sort_points(approx)
                    left_line = np.linalg.norm(sortedp[0]-sortedp[1])
                    upper_line = np.linalg.norm(sortedp[0]-sortedp[2])
                    box_area = upper_line * left_line
                    # TODO: check if coordinates are inside the image
                    sortedp[0][0] = max(sortedp[0][0], 0)
                    sortedp[0][1] = max(sortedp[0][1], 0) if sub is None else sortedp[0][1] - sub
                    sortedp[3][0] = min(sortedp[3][0], frame.shape[1])
                    sortedp[3][1] = min(sortedp[3][1], frame.shape[0]) if sub is None else sortedp[3][1] + sub

                    if box_area > min_area and box_area < max_area and upper_line > left_line:
                        try:
                            t = self.read_text(frame, (sortedp[0][0], sortedp[0][1]), (sortedp[3][0], sortedp[3][1]), resize)
                            if t != "":
                                text.append(t)
                                contours_.append(approx)
                                # Saves text box and coordinates of title text box
                                if save_frame == 'title':
                                    self.title = gray[sortedp[0][1]:sortedp[3][1], sortedp[0][0]:sortedp[3][0]]
                                    self.title_coords = ((sortedp[0][0], sortedp[0][1]), (sortedp[3][0], sortedp[3][1]))
                                    self.current_title = t

                        except Exception:
                            pass

        if not allow_multi:
            if len(text) == 1:
                return text[0]
            return None
        return text

    def sort_points(self, points):
        '''
        points :  assumes four points
        returns upper left, lower left, lower right, upper right
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

# boxes, rW, rH = detect_text(frame, self.net)
#
# ## TODO : Check if any of the boxes are words from the same sentence - join overlapping boxes ?
# ## TODO : Check if width is larger than height
# ## TODO : Make boxes larger to ensure we get the whole text
# ## TODO : Run OCR inside text boxes
# for (startX, startY, endX, endY) in boxes:
#     sX = max(int(startX * rW - 0.1 * startX * rW), 0)
#     sY = max(int(startY * rH - 0.1 * startY * rH), 0)
#     eX = min(int(endX * rW + 0.1 * endX * rW), frame.shape[1])
#     eY = min(int(endY * rH + 0.1 * endY * rH), frame.shape[0])
#
#     # check if the box overlaps with another box in boxes
#     if (eX-sX) > (eY-sY):
#         print("box found")
#         #self.read_text(frame, (sX, sY), (eX, eY))
