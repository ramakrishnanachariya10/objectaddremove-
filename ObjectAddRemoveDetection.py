#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from roipoly import MultiRoi
from queue import Queue
import logging
from datetime import datetime
import sys
import imutils
import joblib
from shapely.geometry import Point, Polygon


def timestamp():
    return datetime.now().strftime('%H_%M_%S_%d_%m_%Y')

def scene_obj_Change(c,regularized=0.0001,acceptance=0.01,objsize=5,roi=False,diffvisualize=False,viewFeatures=False,kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))):
   
    LOG_FILENAME = datetime.now().strftime('logs/logfile_%H_%M_%S_%d_%m_%Y.log')
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Scene Object change detection Job Started...')
    logging.debug('Monitering started...')
      
    #try:
    rt,f = c.read()
    if rt:
        f=cv2.resize(f,(imsize,imsize))
        avg1 = np.float32(cv2.resize(f,(imsize,imsize)))
        avg2 = np.float32(cv2.resize(f,(imsize,imsize)))
        avg3 = np.float32(cv2.resize(f,(imsize,imsize)))
    if roi:
        plt.imshow(f, interpolation='nearest', cmap="Greys")
        multiroi_named = MultiRoi(roi_names=[])
        
        for name, roi in multiroi_named.rois.items():
            print(roi.x)
        polyX=roi.x
        polyY=roi.y         
        poly=[(polyX[i],polyY[i]) for i in range(len(polyX))]
        polyr=[[polyX[i],polyY[i]] for i in range(len(polyX))]
        print(poly)
                
   # except:
   #     logging.error('file reading problem is occured....')
    #    sys.exit('file has some issue')

    while(1):
        ret,f = c.read()
        of=f.copy()
        if ret: 
            f=cv2.resize(f,(imsize,imsize))
            #mask = roi.get_mask(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            
            if roi:
                mask = np.zeros(f.shape, dtype=np.uint8)
                roi_corners = np.array([poly], dtype=np.int32)
                # fill the ROI so it doesn't get wiped out when the mask is applied
                channel_count = f.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (255,)*channel_count
                cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                # apply the mask
                f= cv2.bitwise_and(f, mask)

            frm1=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

            cv2.accumulateWeighted(f,avg1,acceptance)
            cv2.accumulateWeighted(f,avg2,regularized)

            res1 = cv2.convertScaleAbs(avg1)
            res2 = cv2.convertScaleAbs(avg2)
            if diffvisualize:
                cv2.imshow('avg1',res1)
                cv2.imshow('avg2',res2)


            gry1=cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
            gry2=cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)


            change=cv2.absdiff(cv2.absdiff(res2,res1),cv2.subtract(res2,res1))
            cv2.accumulateWeighted(change,avg3,0.0001)
            res3 = cv2.convertScaleAbs(avg3)


            objchangemodel=cv2.absdiff(cv2.absdiff(res2,res1),cv2.subtract(res2,res1))
            objchangemodel=cv2.cvtColor(objchangemodel, cv2.COLOR_BGR2GRAY)
            ret, objchangemodel = cv2.threshold(objchangemodel, 20,255 ,0)
            opening2 = cv2.morphologyEx(objchangemodel, cv2.MORPH_OPEN, kernel)
            #opening2 = cv2.dilate(opening2,kernel,iterations = 1)
            contours,thh,uu= cv2.findContours(opening2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

            #print("Number of Contours found = " + str(len(contours))) 

            if len(contours)>1:
                x,y,w,h = cv2.boundingRect(contours)
                if w>objsize and h>objsize:
                    if roi:
                        
                        cv2.rectangle(of,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.circle(of,(int(x+w/2),int(y+h/2)),3,(0,255,0),2)
                        if Polygon(poly).contains(Point(x+(w/2),y+(h/2))):#.within(Polygon(poly)):
                            print('change detected')
                            cv2.rectangle(of,(x,y),(x+w,y+h),(0,0,255),4)
                            logging.info(timestamp()+ ': Change detected')
            cv2.imshow('img',of)
            #result1.write(objchangemodel)
            if viewFeatures:
                cv2.imshow('res3',opening2)
                cv2.imshow('diff2',cv2.absdiff(cv2.absdiff(res2,res1),cv2.subtract(res2,res1)))
            if cv2.waitKey(50)& 0xFF == ord('q'):
                break
        else:
            print('Video End')
            break

    cv2.destroyAllWindows()
    c.release()
    
if __name__ == "__main__":

    video='/home/ram/pivotchain/out4.mp4'
    imsize = 600
    #c = cv2.VideoCapture(0)
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
    c = cv2.VideoCapture(video)
    scene_obj_Change(c,regularized=0.001,acceptance=0.09,objsize=10 ,roi=True,viewFeatures=True,kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))




