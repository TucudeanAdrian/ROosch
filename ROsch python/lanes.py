import numpy as np
import cv2
from sklearn import linear_model
import time
import lanes
from FreeSpaceDetection import FSD
def extract_lane(road_lines):
    left_lane = []
    right_lane = []
    left_slope = []
    right_slope = []

    if road_lines is not None:
        for x in range(0, len(road_lines)):
            for x1,y1,x2,y2 in road_lines[x]:
                slope = compute_slope(x1,y1,x2,y2)
                if (slope < 0):
                    left_lane.append(road_lines[x])
                    left_slope.append(slope)
                else:
                    if (slope > 0):
                        right_lane.append(road_lines[x])
                        right_slope.append(slope)
                
        return left_lane, right_lane , left_slope, right_slope
    
#Compute slope of the line when points are given
def compute_slope(x1,y1,x2,y2):
    if x2!=x1:
        return ((y2-y1)/(x2-x1))


def split_append(left_lane, right_lane):
    left_lane_sa = []
    right_lane_sa = []
    
    for x in range(0, len(left_lane)):
        for x1,y1,x2,y2 in left_lane[x]:
            left_lane_sa.append([x1, y1])
            left_lane_sa.append([x2, y2])

    for y in range(0, len(right_lane)):
        for x1,y1,x2,y2 in right_lane[y]:
            right_lane_sa.append([x1,y1])
            right_lane_sa.append([x2,y2])
            
    left_lane_sa = np.array(left_lane_sa)
    right_lane_sa = np.array(right_lane_sa)
    left_lane_sa,right_lane_sa = sort(left_lane_sa,right_lane_sa)
    return left_lane_sa,right_lane_sa

#This fucntion prints the lanes after the frame is split and merged
      

def sort(left_lane_sa,right_lane_sa):
    left_lane_sa = left_lane_sa[np.argsort(left_lane_sa[:, 0])]
    right_lane_sa = right_lane_sa[np.argsort(right_lane_sa[:, 0])]

    return left_lane_sa, right_lane_sa

def draw_lanes(left_lane_sa, right_lane_sa, frame):
    (vx_left,vy_left,x0_left,y0_left) = cv2.fitLine(left_lane_sa,cv2.DIST_L2,0,0.01,0.01)
    (vx_right,vy_right,x0_right,y0_right) = cv2.fitLine(right_lane_sa,cv2.DIST_L2,0,0.01,0.01)
    left_len = len(left_lane_sa)
    right_len = len(right_lane_sa)
    slope_left = vy_left / vx_left
    slope_right = vy_right / vx_right
    intercept_left = y0_left - (slope_left * x0_left)
    intercept_right = y0_right - (slope_right * x0_right)

    ysize = frame.shape[0]
    xsize = frame.shape[1]
    y_limit_low = int(0.95*ysize)
    y_limit_high = int(0.65*ysize)

    #Coordinates for point 1(Bottom Left)
    y_1 = ysize
    x_1 = int((y_1-intercept_left)/slope_left)

    #Coordinates for point 2(Bottom Left)
    y_2 = y_limit_high
    x_2 = int((y_2-intercept_left)/slope_left)

    #Coordinates for point 3(Bottom Left)
    y_3 = y_limit_high
    x_3 = int((y_3-intercept_right)/slope_right)
    
    #Coordinates for point 4(Bottom Right)
    y_4 = ysize
    x_4 = int((y_4-intercept_right)/slope_right)

    #Draw lines
    cv2.line(frame,(x_1,y_1),(x_2,y_2),(0,255,255),3)
    cv2.line(frame,(x_3,y_3),(x_4,y_4),(0,255,255),3)
    pts = np.array([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]])
    mask_color = (255,255,0)
    frame_copy = frame.copy()
    cv2.fillPoly(frame_copy, np.int32([pts]), mask_color)
    opacity = 0.4
    cv2.addWeighted(frame_copy,opacity,frame,1-opacity,0,frame)

    #Print Routine
    #print(intercept_left,slope_left,intercept_right,slope_right)

def ransac_drawlane(left_lane_sa, right_lane_sa,frame):
    left_lane_x = []
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []

    for x1,y1 in left_lane_sa:
        left_lane_x.append([x1])
        left_lane_y.append([y1])

    for x1,y1 in right_lane_sa:
        right_lane_x.append([x1])
        right_lane_y.append([y1])

    left_ransac_x = np.array(left_lane_x)
    left_ransac_y = np.array(left_lane_y)

    right_ransac_x = np.array(right_lane_x)
    right_ransac_y = np.array(right_lane_y)

        
    left_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    #print(left_ransac_x,left_ransac_y,len(left_ransac_x),len(left_ransac_y), left_ransac_x.shape )
    left_ransac.fit(left_ransac_x, left_ransac_y)
    slope_left = left_ransac.estimator_.coef_
    intercept_left = left_ransac.estimator_.intercept_

    right_ransac = linear_model.RANSACRegressor()
    right_ransac.fit(right_ransac_x, right_ransac_y)
    slope_right = right_ransac.estimator_.coef_
    intercept_right = right_ransac.estimator_.intercept_

    ysize = frame.shape[0]
    xsize = frame.shape[1]
    y_limit_low = int(0.95*ysize)
    y_limit_high = int(0.65*ysize)

    #stanga jos
    y_1 = ysize
    x_1 = int((y_1-intercept_left)/slope_left)

    #Cdreaptasus
    y_2 = y_limit_high
    x_2 = int((y_2-intercept_left)/slope_left)

    # stanga sus
    y_3 = y_limit_high
    x_3 = int((y_3-intercept_right)/slope_right)
    #dreapta jos
    y_4 = ysize
    x_4 = int((y_4-intercept_right)/slope_right)
    #
    try:
        
        GrX=int(x_2/4+x_4/4+x_1/4+x_3/4)
        GrY=int(y_1/4+y_3/4+y_4/4+y_2/4)
        cv2.line(frame,(x_1,y_1),(x_2,y_2),(255,255,255),2)
        cv2.line(frame,(x_3,y_3),(x_4,y_4),(255,255,255),2)
    #cv2.line(frame,(x_2,y_2),(x_4,y_4),(0,255,255),3)
    #-------------------------------------------------
        #cv2.circle(frame,(GrX,GrY),5,(0,255,255),3)
    #cv2.circle(frame,(GrX,ysize),5,(0,255,255),3)
        cv2.circle(frame,(x_3,y_3),6,(255,255,255),1)
        cv2.circle(frame,(x_2,y_2),6,(255,255,255),1)
        #cv2.circle(frame,(int(x_2/2+x_3/2),int(y_2/2+y_3/2)),6,(255,255,255),3)
        #cv2.circle(frame,(int(x_1/2+x_4/2),int(y_1/2+y_4/2)),6,(255,255,255),3)
        cv2.arrowedLine(frame,(int(x_1/2+x_4/2),int(y_1/2+y_4/2)),(int(x_2/2+x_3/2),int(y_2/2+y_3/2)),(0,255,255),1)
        #cv2.line(frame,(int(xsize/2),ysize),(GrX,GrY),(255,255,255),1)
        #cv2.line(frame,(int(xsize/2),ysize),(int(x_2/2+x_3/2),int(y_2/2+y_3/2)),(255,255,255),1)
    except:
        pass

    pts = np.array([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]])
    mask_color = (0,255,0)
    frame_copy = frame.copy()
    cv2.fillPoly(frame_copy, np.int32([pts]), mask_color)
    opacity = 0.2
    cv2.addWeighted(frame_copy,opacity,frame,1-opacity,0,frame)