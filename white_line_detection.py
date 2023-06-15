import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

# video = cv2.VideoCapture(0)
video = cv2.VideoCapture('race_track_speedup.mp4')

while True:
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # HSV = Hue Saturation Value 
    sensitivity = 70
    lower_white = np.array([0,0,255 - sensitivity])
    upper_white = np.array([255,sensitivity,255])

    ROI_vertices = np.array([[(0,470), (750, 220), (1150, 220), (1882, 470)]], dtype=np.int32) 

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_pixels_only = cv2.bitwise_and(frame, frame, mask = white_mask)
    edges_image = cv2.Canny(white_pixels_only, 100, 300)

    cv2.imshow('edges_image', edges_image)

    edges_mask = np.zeros_like(edges_image)   
    cv2.fillPoly(edges_mask, ROI_vertices, 255)
    edges_in_ROI = cv2.bitwise_and(edges_image, edges_mask)

    cv2.imshow('edges_in_ROI', edges_in_ROI)


    rho = 2            # distance resolution in pixels 
    theta = np.pi/180  # angular resolution in radians 
    threshold = 40     # minimum number of votes 
    min_line_len = 100  # minimum number of pixels making up a line
    max_line_gap = 150  # maximum gap in pixels between connectable line segments    
    
    lines = cv2.HoughLinesP(edges_in_ROI, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    line_image = np.zeros((edges_in_ROI.shape[0], edges_in_ROI.shape[1], 3), dtype=np.uint8)
    
    for line in lines:
        for x1,y1,x2,y2 in line:      
            cv2.line(line_image, (x1, y1), (x2, y2), [255, 0, 0], 20)
    lines 

    a = 1
    b = 1
    y = 0    

    # Resultant weighted image is calculated as follows: original_img * α + img * β + γ
    image_with_lines = cv2.addWeighted(frame, a, line_image, b, y)

    cv2.imshow('image_with_lines', image_with_lines)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cv2.release()