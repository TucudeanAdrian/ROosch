import numpy as np
import cv2
from sklearn import linear_model
import time
from lanes import extract_lane, compute_slope,split_append,sort,ransac_drawlane,ransac_drawlane
from FreeSpaceDetection import FSD


if __name__== "__main__":
    cap = cv2.VideoCapture('project_video.mp4')
    ret,frame=cap.read()
    scale_percent = 50 # percent of original size
    #xsize = int(frame.shape[1] * scale_percent / 100)
    #ysize = int(frame.shape[0] * scale_percent / 100)
    #dim = (xsize, ysize)
    ##ysize = frame.shape[0]
   ##xsize = frame.shape[1]
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    #out = cv2.VideoWriter('output.avi', fourcc, 20.0, (xsize,ysize))
    # resize image
    #while(1):
    while(cap.isOpened()):
        ret, frame = cap.read()
        fps_init=time.time()
        #Escape when no frame is captured / End of Video
        if frame is None:
            break
        
        # Color space conversion
        scale_percent = 50# percent of original size
        xsize = int(frame.shape[1] * scale_percent / 100)
        ysize = int(frame.shape[0] * scale_percent / 100)
        dim = (xsize, ysize)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2. cvtColor(frame, cv2.COLOR_BGR2HLS)
        #ysize = img_gray.shape[0]
        #xsize = img_gray.shape[1]

        #Detecting yellow and white colors
        low_yellow = np.array([20, 100, 100])
        high_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(img_hsv, low_yellow, high_yellow)
        mask_white = cv2.inRange(img_gray, 200, 255)

        mask_yw = cv2.bitwise_or(mask_yellow, mask_white)
        mask_onimage = cv2.bitwise_and(img_gray, mask_yw)

        #Smoothing for removing noise
        gray_blur = mask_onimage

        #Region of Interest Extraction
        mask_roi = np.zeros(img_gray.shape, dtype=np.uint8) 
        left_bottom = [0, ysize]
        right_bottom = [xsize-0, ysize]
        apex_left = [((xsize/2)-50), ((ysize/2)+50)]
        apex_right = [((xsize/2)+50), ((ysize/2)+50)]
        mask_color = 255
        roi_corners = np.array([[left_bottom, apex_left, apex_right, right_bottom]], dtype=np.int32)
        try:    
            cv2.fillPoly(mask_roi, roi_corners, mask_color)
            image_roi = cv2.bitwise_and(gray_blur, mask_roi)

        #Thresholding before edge
        
            ret, img_postthresh = cv2.threshold(image_roi, 50, 255, cv2.THRESH_BINARY)

        #Use canny edge detection
            edge_low = 50
            edge_high = 200
            img_edge = cv2.Canny(img_postthresh, edge_low, edge_high)

        #Hough Line Draw
            minLength = 20
            maxGap = 10
        
            road_lines = cv2.HoughLinesP(img_postthresh, 1, np.pi/180, 20, minLength, maxGap)
            left_lane, right_lane, left_slope, right_slope = extract_lane(road_lines)
            left_lane_sa, right_lane_sa = split_append(left_lane, right_lane)
            ransac_drawlane(left_lane_sa, right_lane_sa,frame)
            #draw_lanes(left_lane_sa, right_lane_sa,frame)
            
        except:
            pass
        ##------------------------
        fps_last=time.time()-fps_init
      
        fps  = 1/fps_last  #calcul fps
        
        print(fps)
        ##---------------------------
        cv2.imshow('Image',frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


