import numpy as np
import cv2 as cv
import time
import mnist


def measure(image):
    blur = cv.GaussianBlur(image, (5, 5), 0)
    gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    _, out = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    erode = cv.dilate(thresh, kernel)
    clone_image, contours, hierarchy = cv.findContours(erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 1)
    # cv.imshow('contour', image)

    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        print('area', area)

        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if h > 50:
            result = out[y:(y + h), x:(x + w)]

            black = [0, 0, 0]
            constant = cv.copyMakeBorder(result, 40, 40, 40, 40, cv.BORDER_CONSTANT, value=black)

            _dir = './data/' + str(i) + '.png'
            cv.imwrite('./data/' + str(i) + '.png', constant)

            predict = mnist.recognition(_dir)
            print(predict)

            text = "{}".format(predict[0])
            cv.putText(image, text, (np.int(x), np.int(y-10)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv.imshow('rect', image)
            cv.waitKey(0)


src = cv.imread('./data/test7.jpeg')
# cv.imshow('origin', src)

measure(src)

cv.waitKey(0)
cv.destroyAllWindows()

