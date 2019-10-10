import cv2

def draw_cirlce(event,x,y,flags,param):
    global center,clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x,y)
        clicked = False

    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


# global variable
center = (0,0)
clicked = False

cap = cv2.VideoCapture(0)
cv2.namedWindow('Test')
cv2.setMouseCallback('Test',draw_cirlce)

while True:

    ret, frame = cap.read()

    if clicked:
        cv2.circle(frame,center=center,radius=30,color=(255,0,0),thickness=3)

    cv2.imshow('Test',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resource when closing
cap.release()
cv2.destroyAllWindows()
