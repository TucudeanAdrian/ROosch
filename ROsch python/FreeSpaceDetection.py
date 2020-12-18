import cv2
import numpy as np
def FSD(src,out,xsize,ysize):
    
  
    
  
    font = cv2.FONT_HERSHEY_SIMPLEX
    height=ysize
    y_cil=int(ysize/2)
    width=xsize

    #kernel=np.ones((10,20),np.uint8)
    #kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,50))
    c=y_cil
    c1=y_cil
    c2=y_cil
    c3=y_cil
    c4=y_cil
    c5=y_cil
    c6=y_cil
    c7=y_cil
    c8=y_cil
    c9=y_cil
    y_form=[0,0,0,0,0,0,0,0,0,0]
    x_form=[c,c1,c2,c3,c4,c5,c6,c7,c8,c9]
    
    
   
    row=width
    col=height
    
    #ret, frame = capture.read()
    
    x_cil=(int)(width/2)
    y_cil=(int)(height/2)
    l=0
    
    #-----------------------------------------------------------------------------------------------------------------------------------
    #window=out
    #erosion=cv2.dilate(src,kernel2,1)
    pic=np.array(src)
    for i in range(10):
       y_form[i]=int(width/2-80)+16*i
    for i in range(int(height/2),height-int(height/10)):
        for j in range(10):
            if pic[i][y_form[j]]: #and c1>i:
                        if x_form[j]<i and x_form[j]<int(y_cil):
                            x_form[j]=i
    cv2.line(out,(10,0),(10,height-10),(255,255,255),1)
    cv2.line(out,(0,height-10),(width-10,height-10),(255,255,255),1)
    for i in range(0,height,(int)(height/23)):
        if(i%4==0):
            cv2.line(out,(10,i),(40,i),(255,255,255),2)
        else:
            cv2.line(out,(10,i),(20,i),(255,255,255),2)
    cv2.circle(out,((int)(width/2),height),40,(255,255,255),2)
    for j in range(10):
        if x_form[j]>(4*height/5) :
            cv2.circle(out,(y_form[j],x_form[j]),2,(0,0,255),5)
            
    
        else:
            #contor=contor+1
            cv2.circle(out,(y_form[j],x_form[j]),2,(0,255,0),5)
            

    
              
    