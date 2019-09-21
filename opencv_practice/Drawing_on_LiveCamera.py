import cv2


## CallBack Function Rectangle
def draw_rectangle(event,x,y,flags,param):
    global pt1,pt2,topLeft_clicked,botRight_clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        
        # RESET THE RECTANGLE (IT CHECK IF THE RECT THERE)
        # 兩個點以標示完
        if topLeft_clicked == True and botRight_clicked == True:
            pt1 = (0,0)
            pt2 = (0,0)
            topLeft_clicked = False
            botRight_clicked = False
        
        # 標示左上的點 
        if topLeft_clicked == False:
            pt1 = (x,y)
            topLeft_clicked = True

        # 標示右下的點
        elif botRight_clicked == False:
            pt2 = (x,y)
            botRight_clicked = True

## Global Variables
pt1 = (0,0)
pt2 = (0,0)
topLeft_clicked = False
botRight_clicked = False

## CONNECT TO THE CALLBACK
cap = cv2.VideoCapture(0)

cv2.namedWindow('Test')
cv2.setMouseCallback('Test',draw_rectangle)

while True:

    ret, frame = cap.read()

    # Drawing on the frame based off the global variables
    if topLeft_clicked:
        cv2.circle(frame,center=pt1,radius=5,color=(0,0,255),thickness=-1)
    
    if topLeft_clicked and botRight_clicked:
        cv2.rectangle(frame,pt1,pt2,color=(0,0,255),thickness=3)

    cv2.imshow('Test',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resource when closing
cap.release()
cv2.destroyAllWindows()