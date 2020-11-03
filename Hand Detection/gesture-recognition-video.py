import cv2
import numpy as np
import math

# opens camera
capture = cv2.VideoCapture(0)

#isopened() : checks if the camera is on
while capture.isOpened():

    # Captures the frames from the camera
    # ret has data, frame shows the data

    ret, frame = capture.read()

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0) #output data, (x1,y1) (x2,y2) (colour), width
    crop_image = frame[100:300, 100:300]

    #apply gaussian blur : removes noise
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # change color from rgg to hsv
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # create binary image where skin color is white and the rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    #kernel for morphological transformation
    kernel = np.ones((5, 5))

    # dilation means adding pixels and erosion means removing
    # apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=2)
    erosion = cv2.erode(dilation, kernel, iterations=2)
    # so we added and removed some pixels now we have to filter the frame
    # thus using gaussian blur
    # gaussian blur and threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret2, thresh = cv2.threshold(filtered, 127, 255, 0)

    cv2.imshow("Threshold", thresh)

    #Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # find contour with max area
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x+w, y+h), (0, 0, 255), 0)

        # find convex hull
        hull = cv2.convexHull(contour)

        #Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0) # used to draw contour
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0) # used to draw hull

        # find convexity defect
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        count_defects = 0

        # following code counts the defects
        for i in range(defects.shape):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
            b = math.sqrt((far[0]-start[0])**2+(far[1]-start[1])**2)
            c = math.sqrt((end[0]-far[0])**2+(far[1]-start[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2) / (2*b*c))*180) / 3.14

            # if angle>90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)
            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # print fingers
        if count_defects == 0:
            cv2.putText(frame, "One", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2) # font, originpoint, colour, width
        elif count_defects==1:
            cv2.putText(frame, "Two", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)  # font, originpoint, colour, width
        elif count_defects == 2:
            cv2.putText(frame, "Three", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)  # font, originpoint, colour, width
        elif count_defects == 3:
            cv2.putText(frame, "Four", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)  # font, originpoint, colour, width
        elif count_defects ==4:
            cv2.putText(frame, "Five", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)  # font, originpoint, colour, width
        else:
            pass
    except:
        pass
        print(frame)
        cv2.imshow("Gesture", frame)
        ALL_Images = np.hstack((drawing, crop_image))
        cv2.imshow("Contour Image", ALL_Images)
        if cv2.waitKey(1) == ord('q'):
            break

    # show req images

    cv2.imshow("Gesture", frame)
    # all_image = np.hstack((drawing, crop_image))  # horiz stack : combines drawing and cropped img
    # cv2.imshow('Contours', all_image)

    # if cv2.waitKey(1) == ord('q'):
    #     break

capture.release()
cv2.destroyAllWindows()
