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

def scene_obj_Change(c,regularized=0.0001,acceptance=0.01,roi=False,diffvisualize=False,viewFeatures=False,kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))):
   
    LOG_FILENAME = datetime.now().strftime('logfile_%H_%M_%S_%d_%m_%Y.log')
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
    logging.info('Scene Object change detection Job Started...')
    logging.debug('Monitering started...')
      
    #try:
    rt,f = c.read()
    if rt:
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
        print(poly)
                
   # except:
   #     logging.error('file reading problem is occured....')
    #    sys.exit('file has some issue')

    while(1):
        ret,f = c.read()
        if ret:
            f=cv2.resize(f,(imsize,imsize))
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
            ret, objchangemodel = cv2.threshold(objchangemodel, 30, 255,0)
            opening2 = cv2.morphologyEx(objchangemodel, cv2.MORPH_OPEN, kernel)
            opening2 = cv2.dilate(opening2,kernel,iterations = 1)
            contours,thh,uu= cv2.findContours(opening2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

            #print("Number of Contours found = " + str(len(contours))) 

            if len(contours)>1:
                x,y,w,h = cv2.boundingRect(contours)
                if w or h <20:
                    if roi:
                        cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,0),2)
                        if Point(x+w/2,y+h/2).within(Polygon(poly)):
                            print('change detected')
                            cv2.rectangle(f,(x,y),(x+w,y+h),(255,255,0),2)
                            logging.info(timestamp()+ ': Change detected')
            cv2.imshow('img',f)
            #result1.write(objchangemodel)
            if viewFeatures:
                cv2.imshow('res3',opening2)

            #Object locate position
            mask = np.zeros(f.shape[:2],np.uint8)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            mask[opening2 == 0] = 0
            mask[opening2 == 255] = 1
            #mask, bgdModel, fgdModel = cv2.grabCut(f,opening2,None,bgdModel,fgdModel,5,1)
            mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            objimg = f**opening2[:,:,np.newaxis]
           
            if viewFeatures:
                cv2.imshow('diff2',cv2.absdiff(cv2.absdiff(res2,res1),cv2.subtract(res2,res1)))
            if cv2.waitKey(50)& 0xFF == ord('q'):
                break
        else:
            print('Video End')
            break

    cv2.destroyAllWindows()
    c.release()
    
if __name__ == "__main__":

    video='/home/ram/pivotchain/atm/kotak_helmet_addition_removal_EDIT.mp4'#'1.mp4'
    imsize = 600
    #c = cv2.VideoCapture(0)
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    c = cv2.VideoCapture(video)
    scene_obj_Change(c,regularized=0.001,acceptance=0.01,roi=True,viewFeatures=False,kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))




