import cv2
import matplotlib.pyplot as plt
import numpy as np



nadia = cv2.imread('../opencv_practice/DATA/Nadia_Murad.jpg')

denis = cv2.imread('../opencv_practice/DATA/Denis_Mukwege.jpg')

solvay = cv2.imread('../opencv_practice/DATA/solvay_conference.jpg')


face_cascade = cv2.CascadeClassifier('../opencv_practice/DATA/haarcascades/haarcascade_frontalface_default.xml')

# Use base haarCascade
# ----> Found the problem like: double match,detect false
def detect_face(img):
    
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img)
    print (face_rects)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)

    return face_img

new_img = detect_face(solvay)
plt.imshow(new_img,cmap='gray')
plt.show()

# Add parameter of detectMultiScale
# ---> More Better
def adj_detect_face(img):
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    #print (face_rects)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)

    return face_img

new_img = adj_detect_face(solvay)
plt.imshow(new_img,cmap='gray')
plt.show()

# Use Video

cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read(0)

    frame = detect_face(frame)

    cv2.imshow('Video Face Detect',frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()