import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import smtplib
from email.message import EmailMessage

# def email_alert(subject,body,to):
#     msg=EmailMessage()
#     msg.set_content(body)
#     msg['subject']=subject
#     msg['to']=to
    
#     user="aforestfireproject01@gmail.com"
#     password="nbyeykjcvelormjn"
#     msg['from']=user

#     server=smtplib.SMTP("smtp.gmail.com",587)
#     server.starttls()
#     server.login(user,password)
#     server.send_message(msg)
#     server.quit()





# Define video writer outside the loop
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Adjust codec if needed (e.g., 'MJPG')
out = cv2.VideoWriter('output_videos\\output_video1.avi', fourcc, 20.0, (1020, 500))  # Adjust filename and frame rate



model=YOLO('best.pt')
flag=0
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        point = [x, y]
        print(point)



cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('videos\\video1.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)

count=0




while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))


    results=model.predict(frame)
    # print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)

    list=[]
    for index,row in px.iterrows():
#        print(row)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]

        
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
        cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
        # if(c=="Fire" and flag==0):
        #     email_alert("Fire alert","Fire has been detected on these longitudes","ahujaaditya04@gmail.com")
        #     flag=1



    cv2.imshow("RGB", frame)
    out.write(frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()