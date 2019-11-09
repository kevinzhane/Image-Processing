import cv2
import numpy as np 
from sklearn.metrics import pairwise

background = None

accumulated_weight = 0.5


# build the roi area (rectangle)
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600


# STEP 1
# calculate the average score of detect
def calc_accum_avg(frame,accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype('float')
        return None

    cv2.accumulateWeighted(frame,background,accumulated_weight)

# STEP 2
# segment the hand in ROI
# Use the thresholding to grab the hand segment from the ROI
# And draw the conture of hand
def segment(frame,threshold_min=20):
    
    # calculate the absolute diff between the background and frame
    diff = cv2.absdiff(background.astype('uint8'),frame)
    
    # calculate the threshold
    ret,thresholded = cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY)

    # find the contoures
    contours, hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    else:
        # assum the largest externel contoures in ROI, is the hand
        hand_segment = max(contours,key=cv2.contourArea)

        return (thresholded,hand_segment)


# STEP 3
# Finger counting with Convex Hull
# Convex Hull: Find the externel point and connected, so it can build the area.
# And use this area to define the center of the hand
def count_fingers(thresholded,hand_segment):

    conv_hull  = cv2.convexHull(hand_segment)

    # find the extreme point of rectangle
    top    = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])

    # clc the center point
    cX = (left[0] + right[0]) // 2
    cY = (top[0] + bottom[0]) // 2

    # use function to clc the distance between center and externel point
    distance = pairwise.euclidean_distances([(cX,cY)],Y=[left,right,top,bottom])[0]

    max_distance = distance.max()

    # build the circle area 
    radius = int(0.9*max_distance)
    circumfrence = (2*np.pi*radius)

    # create the ROI for the circle
    circular_roi = np.zeros(thresholded.shape[:2],dtype='uint8')

    # draw the interst circle
    cv2.circle(circular_roi,(cX,cY),radius,255,10)

    # build the circle ROI to MASK
    circular_roi = cv2.bitwise_and(thresholded,thresholded,mask=circular_roi)

    # draw the contours of ROI
    contours,hierarchy = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    count = 0

    for cnt in contours:

        (x,y,w,h) = cv2.boundingRect(cnt)

        # ignore the point if it is too far
        out_of_wrist = (cY + (cY*0.25)) > (y+h)

        # ignore the noise
        limit_points = ((circumfrence*0.25) > cnt.shape[0])

        if out_of_wrist and limit_points:
            count += 1

    return count

# STEP 4
# bring it together


cam = cv2.VideoCapture(0)

num_frames = 0

while True:

    ret, frame = cam.read()

    frame_copy = frame.copy()

    roi = frame[roi_top:roi_bottom,roi_right:roi_left]

    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray,(7,7),0)

    if num_frames < 60:
        calc_accum_avg(gray,accumulated_weight)

        if num_frames <= 59:
            cv2.putText(frame_copy,'WAIT. GETTING BACKGROUND',(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Finger Count',frame_copy)

    else:

        hand = segment(gray)

        if hand is not None:

            thresholded, hand_segment = hand

            # draw the contours of hand in live stream
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5)

            fingers = count_fingers(thresholded,hand_segment)

            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            cv2.imshow('Threshold',thresholded)

    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)

    num_frames += 1 
    
    cv2.imshow('Finger Count',frame_copy)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()