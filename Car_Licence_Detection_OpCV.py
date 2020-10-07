import cv2
import numpy as np



car_no_plates_casecade = cv2.CascadeClassifier('D:/MS CS/CNN Classifiers/Car_No_Plate_Detection/haarcascade_russian_plate_number.xml')

cap = cv2.VideoCapture('D:/MS CS/CNN Classifiers/Car_No_Plate_Detection/car2.jpg')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 80)


if (cap.isOpened()==False):
    print('Error Reading video')

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    plates = car_no_plates_casecade.detectMultiScale(gray,scaleFactor=1.2,
    minNeighbors = 5, minSize=(25,25))

    for (x,y,w,h) in plates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        frame[y:y+h,x:x+w] = cv2.blur(frame[y:y+h,x:x+w],(25,25),cv2.BORDER_DEFAULT)
        
    if ret == True:
        cv2.imshow('Video',frame)
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()
