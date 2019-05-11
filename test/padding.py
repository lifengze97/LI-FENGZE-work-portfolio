import cv2 as cv
import numpy as np


black = [0, 0, 0]
img = cv.imread('./data/0.png')

constant = cv.copyMakeBorder(img, 120, 120, 120, 120, cv.BORDER_CONSTANT, value=black)

while True:
    cv.imshow('image', constant)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
