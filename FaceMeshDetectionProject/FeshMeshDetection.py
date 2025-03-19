import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture('Videos/1.mp4')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1) # to change the size of points that are made

desired_width = 900
desired_height = 600

pTime=0
cTime=0

while True:
    success,img=cap.read()
    img = cv2.resize(img, (desired_width, desired_height))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            #mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_TESSELATION)
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                # print(lm)
                h,w,c=img.shape
                x,y=int(lm.x*w),int(lm.y*h)
                print(id,x,y)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()