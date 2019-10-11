import cv2
import matplotlib.pyplot as plt
import numpy as np 

# define the goodfeaturetofind params (Shi-Tomasi)
corner_track_params = dict(maxCorners=10,qualityLevel=0.3,minDistance=7,blockSize=7)

# define the Lucas params
lk_params = dict(winSize=(200,200),maxLevel=2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))



cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()

prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)


# pts to track
prevPts =  cv2.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)

# This mask is drawing lines on the video when tracking points (visualize)
mask = np.zeros_like(prev_frame)



while True:
    
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Call the function to calc nextPts
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray,prevPts,None,**lk_params)

    # Define the status (If find the food feature then set 1)
    good_new = nextPts[status==1]
    good_prev = prevPts[status==1]

    for i, (new,prev) in enumerate(zip(good_new,good_prev)):

        # Reshape vetor 
        x_new , y_new = new.ravel()
        x_prev , y_prev = prev.ravel()

        # Draw the line of tracking line
        mask = cv2.line(mask,(x_new,y_new),(x_prev,y_prev),(0,255,0),3)

        # Draw the current track points
        frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)

    img = cv2.add(frame,mask)
    cv2.imshow('Tracking',img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

    prev_gray = frame_gray.copy()
    prevPts = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()