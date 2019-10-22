import imutils
import os
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from pythonRLSA import rlsa

from clams.serve import ClamApp
from clams.serialize import *
from clams.vocab import AnnotationTypes
from clams.vocab import MediaTypes
from clams.restify import Restifier


class RSLA(ClamApp):
    def appmetadata(self):
        metadata = {
            "name": "RSLA Document Segmentation",
            "description": "This tool applies RSLA document segmentation to the image.",
            "vendor": "Team CLAMS",
            "requires": [MediaTypes.I],
            "produces": [AnnotationTypes.TBOX],
        }
        return metadata

    def sniff(self, mmif):
        # this mock-up method always returns true
        return True

    def annotate(self, mmif):
        if not type(mmif) is Mmif:
            mmif = Mmif(mmif)
        image_filename = mmif.get_medium_location(MediaTypes.I)
        RSLA_output = self.run_RSLA(
            image_filename
        )  # RSLA_output is a list of [(x1, y1, x2, y2)] boxes

        new_view = mmif.new_view()
        contain = new_view.new_contain(AnnotationTypes.TBOX)
        contain.producer = self.__class__

        for int_id, box in enumerate(RSLA_output):
            annotation = new_view.new_annotation(int_id)
            annotation.start = str(0)
            annotation.end = str(
                0
            )  # TODO figure out what we want to store here for images
            annotation.feature = {"box": box}
            annotation.attype = AnnotationTypes.TBOX

        for contain in new_view.contains.keys():
            mmif.contains.update({contain: new_view.id})
        return mmif

    @staticmethod
    def run_RSLA(image_filename, scale_percent=25, rsla_thresh_h=10, rsla_thresh_v=10, contour_area=5): #todo revisit these defaults
        '''

        :param image_filename: path to image
        :param scale_percent:  percent to scale image before rsla, should divide 100
        :param rsla_thresh_h: threshold for horizontal rsla
        :param rsla_thresh_v: threshold for vertical rsla
        :param contour_area: minimum acceptible contour region area
        :return: list of bounding boxes
        '''
        bounding = []
        image = cv2.imread(image_filename)
        orig_wh = image.shape[:-1]

        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        unscale = 100/scale_percent

        dim = (width, height)
        # resize image
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        (thresh, image_binary) = cv2.threshold(
            gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        image_rlsa_horizontal = rlsa.rlsa(image_binary, True, False, rsla_thresh_h)
        image_rlsa_vertical = rlsa.rlsa(image_binary, False, True, rsla_thresh_v)
        combo = np.bitwise_or(image_rlsa_horizontal, image_rlsa_vertical)
        combo = cv2.bitwise_not(combo)
        _, contours, _ = cv2.findContours(combo, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > contour_area:
                # combo = cv2.drawContours(combo, contour, -1, (0, 0, 255), thickness=10)
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(image, (x * 4, y * 4), ((x+w)*4,(y+h)*4), color=(0,0,255))
                bounding.append((x * unscale, y * unscale, (x+w)*unscale, (y+h)*unscale))

        return bounding


if __name__ == "__main__":
    td_tool = RSLA()
    td_service = Restifier(td_tool)
    td_service.run()
    # print (td_tool.run_RSLA("test.jpg"))
