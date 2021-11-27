import cv2
import numpy as np
from time import sleep
min_side=80 # it just controlls amount of red dot on each car having reverse relatioon
min_front=80 #the same is true for this
offset=6 #
pos_linha=550#
delay= 60# controlls the speed of the car movment and detectin
detec = []
carros= 0
def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy
abebe = cv2.VideoCapture('video.mp4')
abdi = cv2.createBackgroundSubtractorMOG2()
while True:
    ret , frame1 = abebe.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = abdi.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thesis_code = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    thesis_code = cv2.morphologyEx (thesis_code, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(thesis_code,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255,127,0), 3) 
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= min_side) and (h >= min_front)
        if not validar_contorno:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)
        for (x,y) in detec:
            if y<(pos_linha+offset) and y>(pos_linha-offset):
                carros+=1
                #three is the thickness of the horizontal line
                #the 000 is the color of the horizontal line while the car is moving
                cv2.line(frame1, (5, pos_linha), (1200, pos_linha), (0,0,0), 3)
                detec.remove((x,y))
                print("adey abeba square number of car detected : "+str(carros))
    cv2.putText(frame1, "ADEY ABEBA-car : "+str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detectar",thesis_code)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
abebe.release()