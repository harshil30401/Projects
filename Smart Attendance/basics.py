# import cv2
# import face_recognition
# import numpy as np


# imgBill = face_recognition.load_image_file("images/Jeff.jpeg")
# imgBill = cv2.cvtColor(imgBill, cv2.COLOR_BGR2RGB)

# imgBillTest = face_recognition.load_image_file("images/JeffAngle.jpeg")
# imgBillTest = cv2.cvtColor(imgBillTest, cv2.COLOR_BGR2RGB)


# faceLocBill = face_recognition.face_locations(imgBill)[0]
# encodeBill = face_recognition.face_encodings(imgBill)[0]
# cv2.rectangle(imgBill, (faceLocBill[3], faceLocBill[0]), (faceLocBill[1], faceLocBill[2]), (255, 0, 255), 2)

# faceLocBillTest = face_recognition.face_locations(imgBillTest)[0]
# encodeBillTest = face_recognition.face_encodings(imgBillTest)[0]
# cv2.rectangle(imgBillTest, (faceLocBillTest[3], faceLocBillTest[0]), (faceLocBillTest[1], faceLocBillTest[2]), (255, 0, 255), 2)

# #Finding the best match
# results = face_recognition.compare_faces([encodeBill], encodeBillTest)[0]
# faceDist = round(face_recognition.face_distance([encodeBill], encodeBillTest)[0], 2)
# cv2.putText(imgBillTest, f"{faceDist} {results}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)

# #Frontend
# cv2.imshow("Bill Gates", imgBill)
# cv2.imshow("Bill Gates Angle", imgBillTest)
# cv2.waitKey(0)




import face_recognition
known_image = face_recognition.load_image_file("images/twin1.jpg")  
unknown_image = face_recognition.load_image_file("images/twin2.jpg") 

T1 = face_recognition.face_encodings(known_image)[0]
T2 = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([T1], T2)
print(results[0])  #Returns true (Does not recognize twins accurately)