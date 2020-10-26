# resolutia necesara 640 pe 480 px 
# 




import cv2
import numpy as np
from numpy import matrix
import time
import math

def nothing(x):## cand am facut interfata grafica , trebuia functia asta
    pass


row=640
col=480

sensivity=20                                                              ################### cat de sensibila ii masca 
#------------------------------
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
#capture = cv2.VideoCapture("http://192.168.1.101:8080/video")           ### pentru ip camera
capture=cv2.VideoCapture(0)               ### pentru webcam
#capture=cv2.VideoCapture("nume_video.mp4") ###pentru video  ((((Cu resolutia 640 pe 480)))

capture.set(3,row)
capture.set(4,col)

x1=320
y1=350
a=0
b=0
intoarcere=0
#vx=[]
#vy=[]
print("3")
#time.sleep(1)
print("2")
#time.sleep(1)
print("1")
#time.sleep(1)
print("START !!!")
while(True):

    c=0
    c1=row
    c2=0
    c3=row
    c4=0
    c5=row
    c6=0
    c7=row
    ret, frame = capture.read()
    dst = cv2.GaussianBlur(frame,(1,1),cv2.BORDER_DEFAULT) 
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    window=frame  #######################################################################      am luat separat window sa nu interfereze cu frame
    lower_range = np.array([0,0,0])#60,80
    upper_range = np.array([255,255,30])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    kernel=np.ones((25,25),np.uint8)
    kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    erosion=cv2.dilate(mask,kernel2,5)
    final = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel2)#finall 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    pic=np.array(final)

    #------------------------------------------------------------------------------------------------------------------
    for i in range((int)(row/2)):
        if pic[200][i]==255: #and c<i:
            #c==i
            #cv2.circle(window,(i,200),2,(0,0,255),3)
            if c2<i:
                c2=i
        if pic[100][i]==255:
            
            #cv2.circle(window,(i,100),2,(0,0,255),3)
            if c<i:
                c=i
        if pic[280][i]==255: 
            #c==i
            #cv2.circle(window,(i,280),2,(0,0,255),3)
            if c4<i:
                 c4=i
        if pic[370][i]==255: 
            #c==i
            #cv2.circle(window,(i,370),2,(0,0,255),3)
            if c6<i:
                 c6=i
    cv2.circle(window,(c,100),2,(255,0,0),5) # \
    cv2.circle(window,(c2,200),2,(255,0,0),5)#   afisare puncte de pe stanga
    cv2.circle(window,(c4,280),2,(255,0,0),5)#   
    cv2.circle(window,(c6,370),2,(255,0,0),5)# / 
    
    for i in range((int)(row/2),row):
        if pic[100][i]==255 :#and c1>i:
            
            #cv2.circle(window,(i,100),2,(0,0,255),3)
            if c1>i:
                c1=i
        if pic[200][i]==255 :#and c1>i:
            #c1==i
            #cv2.circle(window,(i,200),2,(0,0,255),3)
            if c3>i:
                c3=i
        if pic[280][i]==255 :#and c1>i:
            #c1==i
            #cv2.circle(window,(i,280),2,(0,0,255),3)
            if c5>i:
                c5=i
        if pic[370][i]==255 :
            #cv2.circle(window,(i,370),2,(0,0,255),3)
            if c7>i:
                c7=i
    cv2.circle(window,(c1,100),2,(255,0,0),5) #
    cv2.circle(window,(c3,200),2,(255,0,0),5) # afisam punctele pe dreapta
    cv2.circle(window,(c5,280),2,(255,0,0),5) #
    cv2.circle(window,(c7,370),2,(255,0,0),5) # 
   #----------------------------------------------
    cv2.line(window,(c,100),(c2,200),(100,255,255),2)
    cv2.line(window,(c1,100),(c3,200),(0,255,255),2)
    cv2.line(window,(c2,200),(c4,280),(0,255,255),2)               ####   aici am unit punctele sa defineasca suprafata drumului
    cv2.line(window,(c3,200),(c5,280),(0,255,255),2)
    cv2.line(window,(c4,280),(c6,370),(0,255,255),2)
    cv2.line(window,(c5,280),(c7,370),(0,255,255),2)
   #------------------------------------------------------------------------------
    x_form=[c,c1,c2,c3,c4,c5,c6,c7]
    y_form=[100,100,200,200,280,280,370,370]
    cv2.arrowedLine(window,(320,480),((int)(sum(x_form)/8),(int)(sum(y_form)/8)),(255,255,0),3)
    a1=sum(x_form)/8
    a2=sum(y_form)/8
    cv2.putText(window,"A ",((int)(a1),(int)(a2)),font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(window,"B ",(600,470),font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(window,"0 ",(310,470),font, 1,(255,255,255),1,cv2.LINE_AA)  #####  calculam unghiul
    A=np.array([a1,a2])
    O=np.array([600,470])
    B=np.array([320,470])
    AB = B - A
    OB = B - O
    cosine_angle = np.dot(AB, OB) / (np.linalg.norm(AB) * np.linalg.norm(OB))
    angle = np.degrees(np.arccos(cosine_angle))
    print(np.around(angle,2))
    #--------------------------------------------------------------------------
    Lm=180-np.around(angle)
    Rm=np.around(angle)                                 #valorile pentru motoare
    #----------------------------------------------------------------------------
    cv2.putText(window,"Angle "+str((int)(np.around(angle,2))),((int)(a1),(int)(a2)),font, 1,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(window,"L "+str(Lm),(20,440),font, 1,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(window,"R "+str(Rm),(520,440),font, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('video', window)                      ########################          interfata modificata
    out.write(window)                               ########################        sa salvezi video
    cv2.imshow('Final Mask',final)                 ########################              masca
   
    if cv2.waitKey(1) == 27:
        break
 
capture.release()
cv2.destroyAllWindows()
