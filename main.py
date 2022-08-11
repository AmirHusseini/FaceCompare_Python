import cv2
import numpy as np
import face_recognition

imgkung = face_recognition.load_image_file('images/carl.jpg')
imgkung = cv2.cvtColor(imgkung, cv2.COLOR_RGB2BGR)

imgtest = face_recognition.load_image_file('images/carl2.jpg')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_RGB2BGR)

#find face locations
facelock = face_recognition.face_locations(imgkung)[0]
encodeKung = face_recognition.face_encodings(imgkung)[0]

#RectangleFace
cv2.rectangle(imgkung,(facelock[3], facelock[0]), (facelock[1], facelock[2]),(255,0,255),2)
##-----------------------------------TEST
#find face locations
facelockTest = face_recognition.face_locations(imgtest)[0]
encodeKungTest = face_recognition.face_encodings(imgtest)[0]

#RectangleFace
cv2.rectangle(imgtest,(facelockTest[3], facelockTest[0]), (facelockTest[1], facelockTest[2]),(255,0,255),2)


results = face_recognition.compare_faces([encodeKung],encodeKungTest)
print(results)

cv2.imshow('Carl Gustav',imgkung)
cv2.imshow('Carl Test',imgtest)
cv2.waitKey(0)