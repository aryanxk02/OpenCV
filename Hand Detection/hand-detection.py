import cv2

hand = cv2.imread('hand1.png', 0)

ret, the = cv2.threshold(hand, 70, 255, cv2.THRESH_BINARY)

contours, y = cv2.findContours(the.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull = [cv2.convexHull(c) for c in contours]

final = cv2.drawContours(hand, hull, -1, (255, 0, 0))

cv2.imshow('Original Image', hand)
cv2.imshow('Threshold Output', the)
cv2.imshow('Convex Hull', final)

cv2.waitKey(0)
cv2.destroyAllWindows()

