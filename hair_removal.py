import cv2
import numpy as np


path = 'train/train/nevus/nev00391.jpg'
image = cv2.imread(path, cv2.IMREAD_COLOR)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)


x, y, w, h = cv2.boundingRect(largest_contour)
roi = image[y:y+h, x:x+w]  



gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
blackhat = cv2.morphologyEx(gray_roi, cv2.MORPH_BLACKHAT, kernel)


bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)


_, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)


clean_roi = cv2.inpaint(roi, mask, 6, cv2.INPAINT_TELEA)


cv2.imshow("Original Image", image)
cv2.imshow("Dynamic ROI", roi)
cv2.imshow("Binary Mask", mask)
cv2.imshow("Cleaned Image", clean_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()