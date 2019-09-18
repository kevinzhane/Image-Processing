import cv2

# Build the object for camera
cap = cv2.VideoCapture(0)

# Read camera config
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# WINDOWS -- *'DIVX'
# MAC or LINUX -- *'XVID'
# Writer is command to saving the video
writer = cv2.VideoWriter('myvideo.mp4',cv2.VideoWriter_fourcc(*'XVID'),25,(width,height))

# Build the frame to show video
while True:

    # read() is to read the continue image
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Operations (Drawing)
    writer.write(frame)

    # Show the frame of video
    cv2.imshow('frame',frame)
    
    # Setting the waitkey 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resource when closing
cap.release()
writer.release()
cv2.destroyAllWindows()