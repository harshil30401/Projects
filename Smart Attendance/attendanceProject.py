import cv2, face_recognition, os, csv
import numpy as np
import pandas as pd
from datetime import datetime

#1 Import Images & convert bgr -> rgb

todayDate = str(datetime.today())[:4] + str(datetime.today())[5:7] + str(datetime.today())[8:10]

current_time = datetime.now().strftime("%H:%M:%S")
# current_time = '11:15:00'

# day = datetime.today().strftime('%A').upper()
day = "TUESDAY"

if((day == "SUNDAY") or (day == "SUNDAY")):
    print("Schools are closed on weekends!")
    exit()

# day = "WEDNESDAY"

identity = day[:2]+todayDate


tt = pd.read_csv('timetable.csv')
timeframes = tt["Time"].to_list()

teachers = {
    "French":"Jeniffer",
    "English":"Jean",
    "Hindi":"Kavita",
    "Science":"Susan",
    "Math":"Felix",
    "History":"Domnic",
    "Geography":"Sheetal",
    "Swimming":"George",
    "PE":"Glen"
}

path = "RegisteredCandidates"
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f"{path}\{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


#2 Find encodings of each image

def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgEncode = face_recognition.face_encodings(img)[0]

        encodeList.append(imgEncode)
    return encodeList

def createCSV(filename):
    peopleData = [
    [ "StudentName", "Time", "Subject", "Teacher", "Remark"],
    ]
    with open('records/'+filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(peopleData)

createCSV(identity+'.csv')

# with open('timetable.csv', 'r') as tt:


#['09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']

def markAttendance(name):

# Subject Teacher Remark


    with open('records/'+identity+'.csv', 'r+') as f:
        myDataList = f.readlines()

        nameList = []    #All the names that are found will be appended here

        current_time = datetime.now().strftime("%H:%M:%S")


        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if ( (current_time <  timeframes[1])):
            subject = tt[day][0]
            teacher = teachers[subject]
            
            hourDelay = (datetime.strptime(timeframes[0], "%H:%M:%S").hour - 
            datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S").hour)
            
            minDelay = (datetime.strptime(timeframes[0], "%H:%M:%S").minute - 
            datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S").minute)

            if (int(hourDelay) == 0):
                if(int(minDelay < 0)):
                    remark = minDelay
            elif (int(hourDelay) < 0):
                remark = minDelay
            else:
                remark = str(hourDelay)+'H:'+str(minDelay*-1)+'M'



        elif ( current_time <  timeframes[2]):
            subject = tt[day][1]
            teacher = teachers[subject]


            hourDelay = (datetime.strptime(timeframes[1], "%H:%M:%S").hour - 
            datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S").hour)
            
            minDelay = (datetime.strptime(timeframes[1], "%H:%M:%S").minute - 
            datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S").minute)

            if (int(hourDelay) == 0):
                if(int(minDelay < 0)):
                    remark = minDelay
            elif (int(hourDelay) < 0):
                remark = minDelay
            else:
                remark = str(hourDelay)+'H:'+str(minDelay*-1)+'M'

        elif ( current_time <  timeframes[3]):
            subject = tt[day][2]
            teacher = teachers[subject]


            hourDelay = (datetime.strptime(timeframes[2], "%H:%M:%S").hour - 
            datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S").hour)
            
            minDelay = (datetime.strptime(timeframes[2], "%H:%M:%S").minute - 
            datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S").minute)

            if (int(hourDelay) == 0):
                if(int(minDelay < 0)):
                    remark = minDelay
            elif (int(hourDelay) < 0):
                remark = minDelay
            else:
                remark = str(hourDelay)+'H:'+str(minDelay*-1)+'M'

        elif ( datetime.now().strftime("%H:%M:%S") <  timeframes[4]):
            subject = tt[day][3]
            teacher = teachers[subject]


            hourDelay = (datetime.strptime(timeframes[3], "%H:%M:%S").hour - 
            datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S").hour)
            
            minDelay = (datetime.strptime(timeframes[3], "%H:%M:%S").minute - 
            datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S").minute)

            if (int(hourDelay) == 0):
                if(int(minDelay < 0)):
                    remark = minDelay
            elif (int(hourDelay) < 0):
                remark = minDelay
            else:
                remark = str(hourDelay)+'H:'+str(minDelay*-1)+'M'
        
        else:
            subject = tt[day][4]
            teacher = teachers[subject]


            hourDelay = (datetime.strptime(timeframes[4], "%H:%M:%S").hour - 
            datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S").hour)
            
            minDelay = (datetime.strptime(timeframes[4], "%H:%M:%S").minute - 
            datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S").minute)

            if (int(hourDelay) == 0):
                if(int(minDelay < 0)):
                    remark = minDelay
            elif (int(hourDelay) < 0):
                remark = minDelay
            else:
                remark = str(hourDelay)+'H:'+str(minDelay*-1)+'M'


        if name not in nameList:
            f.writelines(f'{name}, {current_time}, {subject}, {teacher}, {remark}\n')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

#3. Find Matches (Webcam)

cap = cv2.VideoCapture(0)   
while True:
    success, img = cap.read()

    #Since we are doing this in real time, we will reduce size of the image which speeds up the process
    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)  #1/4th of its size 
    imgSmall = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgSmall)
    imgEncodeCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(imgEncodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        #Lowest Distance = Best Match
        print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            # y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2,y2), (0, 255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    #Display Bounding box and name

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)