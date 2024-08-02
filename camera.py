import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone

# Define video writer codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'XVID' codec for AVI format
out = cv2.VideoWriter('output_videos/output_video.avi', fourcc, 20.0, (640, 480))  # Adjust filename and frame size if needed

model = YOLO('best.pt')
flag = 0

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
        # if(c=="Fire" and flag==0):
        #     email_alert("Fire alert","Fire has been detected on these longitudes","ahujaaditya04@gmail.com")
        #     flag=1

    cv2.imshow("RGB", frame)
    out.write(frame)  # Write frame to video file

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and video writer, then close all OpenCV windows
cap.release()
out.release()  # Ensure video writer is properly released
cv2.destroyAllWindows()
