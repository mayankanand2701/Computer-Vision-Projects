import cv2
import mediapipe as mp
import time

desired_width = 900
desired_height = 600

class FaceMeshDetector():
    def __init__(self,staticMode=False,maxFaces=2,minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetectionCon=minDetectionCon
        self.minTrackCon=minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)

    def findFaceMesh(self,img,draw=True):
        img = cv2.resize(img, (desired_width, desired_height))
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                face=[]
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    #cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.3, (0, 255, 0), 1)
                    face.append([x,y])
                faces.append(face)
        return img,faces

def main():
    pTime = 0
    cTime = 0
    detector=FaceMeshDetector()
    cap = cv2.VideoCapture('Videos/2.mp4')

    while True:
        success, img = cap.read()
        img,faces=detector.findFaceMesh(img)
        if len(faces)!=0:
            print(len(faces))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()