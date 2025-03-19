import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture('Videos/1.mp4')

mpFaceDetection=mp.solutions.face_detection
faceDetection=mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils

desired_width = 900
desired_height = 600

pTime=0
cTime=0

while True:
    success,img=cap.read()
    img = cv2.resize(img, (desired_width, desired_height))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(img,detection)   # inbuilt code to create rectangle on face
            # print(id,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            # Code to build rectangle on face of person
            h,w,c=img.shape
            bboxC=detection.location_data.relative_bounding_box
            bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()