import cv2
import numpy as np
import face_recognition
image=cv2.imread("test_image_work.jpeg")
grey_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
classifier=cv2.CascadeClassifier("frontalface.xml")
detect=classifier.detectMultiScale(grey_image,1.1,3)
faceencoding=[]
for (x,y,w,h) in detect:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
    img=image[y:h,x:w]
    faceloc=face_recognition.face_locations(img)
    faceencoding1=face_recognition.face_encodings(img)
    faceencoding.append(faceencoding1)


    #encodings.append(faceencoding)
image_test=cv2.imread("class_image_onam.jpeg")
grey_image_test=cv2.cvtColor(image_test,cv2.COLOR_BGR2GRAY)
detecttest=classifier.detectMultiScale(grey_image_test,1.1,3)
for (x1,y1,w1,h1) in detecttest:
     cv2.rectangle(image_test,(x1,y1),(x1+w1,y1+h1),(0,255,255),2)
     img_test=image_test[y:h,x:w]
     faceloc = face_recognition.face_locations(img_test)
     faceencoding1 = face_recognition.face_encodings(img_test)
     if faceencoding1 in faceencoding:
         print("attendence marked")

 #encodings.append(faceencoding)

cv2.imshow("test image",image_test)
cv2.waitKey(0)




cv2.imshow("detected image",image)
cv2.waitKey(0)
