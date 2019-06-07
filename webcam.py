import cv2
 
cam = cv2.VideoCapture(0)

if cam.isOpened() == False:
    print("Unable to read camera feed")
 
while True:
    ret, frame = cam.read()
    
    if ret == True:
        cv2.imwrite('test_img.jpg', frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 32: #Press space bar to capture image
            break
    else:
        break 
 
cam.release()
cv2.destroyAllWindows()