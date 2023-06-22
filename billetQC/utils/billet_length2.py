# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 01:19:00 2021

@author: barton

A standalone version of the math for more accurately calculating billet length.

To call this file, use "import billet_length", then call the function lengthCalc(source = 'source/location'). Make sure that the source is set to wherever the .txt labels are.

Importantly, it should only be called once all images have been scanned for billets, as it requires the entire load to be complete in order to get a better idea of how to modify the lengths appropriately.


"""

import argparse
import numpy as np
from glob import glob
import matplotlib
matplotlib.use('Qt5Agg')
#%matplotlib inline
import matplotlib.pyplot as plt
import os


def lengthCalc2(
        conf=0.5,
        source='runs/detect/exp',
        length_multiplier=142.3906,
        load_width=0.75,
        filter_ratio=0.5,
        keep_width=0.5,
        save_img=False,
        force_diagonal = False
        ):
    file_names = glob(source + '/labels/*')
    
    lengthGraphOut = []
    allDetections = np.array([[],[],[],[]])
    imgDetections = np.array([[],[],[],[]])
    maxRatio = [[],[]]
    minWidth = []
    load_height = []
    h = 1
    if load_width != 0:
        h = 1/load_width
    
    #print(conf)
    #print(length_multiplier)
    #print(load_height)
    #print(camera_height)
    #print(filter_width)
    

    for f in file_names:
        #print(f)
        data = np.array(np.loadtxt(f, delimiter=" "))
        #print(data)
 
                # This entire chunk of code is just to deal with multiple different cases - whether or not the .txt has confidence values built in, whether or not there is only a single line (or no lines at all). 
                # Then, it removes edge billets
                # Then, it scales billets according to the outermost visible billet.
 
 
        if(data.shape[0] != 0):
            if(data.ndim == 2):
                if (data.shape[1] == 5):
                    #print('no confidence values associated with labels')
                    for i in range(data.shape[0]):
                        if((data[i][1] - data[i][3]/2.0) > 0.01) and ((data[i][1] + data[i][3]/2.0) < 0.99) and ((data[i][2] - data[i][4]/2.0) > 0.01) and ((data[i][2] + data[i][4]/2.0) < 0.99):
                            imgDetections = np.append(imgDetections, [data[i][1], data[i][2], data[i][3], data[i][4]])
                            
                            if (data[i][4]*filter_ratio < data[i][3] ) or (data[i][3]*filter_ratio < data[i][4]):
                                maxRatio = np.append(maxRatio, [[data[i][3]], [data[i][4]]])
                        #print(imgDetections)
                        if (imgDetections != [[],[],[],[]]):
                            if ((np.max(imgDetections[:][1:2]) > 0.73) or (np.min(imgDetections[:][1:2]) < 0.27)):
                                diff = np.max([np.max(imgDetections[:][1:2]) - 0.73, 0.27 - np.min(imgDetections[:][1:2])])
                                

                                load_height = np.append(load_height, (h*(((diff)/2)/(h+(diff+0.5)/2))))
                        allDetections = np.append(allDetections, imgDetections)
                        imgDetections = [[],[],[],[]]
                    
                if (data.shape[1] == 6):
                    #print('applying confidence threshold')
                    for i in range(data.shape[0]):
                        if(data[i][5] >= conf):
                            if((data[i][1] - data[i][3]/2.0) > 0.01) and ((data[i][1] + data[i][3]/2.0) < 0.99) and ((data[i][2] - data[i][4]/2.0) > 0.01) and ((data[i][2] + data[i][4]/2.0) < 0.99):
                                imgDetections = np.append(imgDetections, [data[i][1], data[i][2], data[i][3], data[i][4]])
                                
                                if (data[i][4]*filter_ratio < data[i][3] ) or (data[i][3]*filter_ratio < data[i][4]):
                                    maxRatio = np.append(maxRatio, [[data[i][3]], [data[i][4]]])

                            #print(imgDetections)
                            #print(imgDetections[:][1:2])
                        if (imgDetections != [[],[],[],[]]):
                            if ((np.max(imgDetections[:][1:2]) > 0.73) or (np.min(imgDetections[:][1:2]) < 0.27)):
                                diff = np.max([np.max(imgDetections[:][1:2]) - 0.73, 0.27 - np.min(imgDetections[:][1:2])])
                                

                                load_height = np.append(load_height, (h*(((diff)/2)/(h+(diff+0.5)/2))))
                        allDetections = np.append(allDetections, imgDetections)
                        imgDetections = [[],[],[],[]]
            else:
                if (data.shape == 5):
                    #print('no confidence values associated with labels')
                    for i in range(data.shape[0]):
                        if((data[1] - data[3]/2.0) > 0.01) and ((data[1] + data[3]/2.0) < 0.99) and ((data[2] - data[4]/2.0) > 0.01) and ((data[2] + data[4]/2.0) < 0.99):
                            imgDetections = np.append(imgDetections, [data[1], data[2], data[3], data[4]])
                            
                            if (data[4]*filter_ratio < data[3] ) or (data[3]*filter_ratio < data[4]):
                                maxRatio = np.append(maxRatio, [[data[3]], [data[4]]])

                        #print(imgDetections)
                        if (imgDetections != [[],[],[],[]]):
                            if ((np.max(imgDetections[:][1:2]) > 0.73) or (np.min(imgDetections[:][1:2]) < 0.27)):
                                    diff = np.max([np.max(imgDetections[:][1:2]) - 0.73, 0.27 - np.min(imgDetections[:][1:2])])
                                    

                                    load_height = np.append(load_height, (h*(((diff)/2)/(h+(diff+0.5)/2))))
                        allDetections = np.append(allDetections, imgDetections)
                        imgDetections = [[],[],[],[]]
                        
                if (data.shape == 6):
                    #print('applying confidence threshold')
                    for i in range(data.shape[0]):
                        if(data[5] >= conf):
                            if((data[1] - data[3]/2.0) > 0.01) and ((data[1] + data[3]/2.0) < 0.99) and ((data[2] - data[4]/2.0) > 0.01) and ((data[2] + data[4]/2.0) < 0.99):
                                imgDetections = np.append(imgDetections, [data[1], data[2], data[3], data[4]])
                                
                                if (data[4]*filter_ratio < data[3] ) or (data[3]*filter_ratio < data[4]):
                                    maxRatio = np.append(maxRatio, [[data[3]], [data[4]]])

                            #print(imgDetections)
                        if (imgDetections != [[],[],[],[]] and load_width != 0):
                            if ((np.max(imgDetections[:][1:2]) > 0.73) or (np.min(imgDetections[:][1:2]) < 0.27)):
                                diff = np.max([np.max(imgDetections[:][1:2]) - 0.73, 0.27 - np.min(imgDetections[:][1:2])])
                                
                                
                                load_height = np.append(load_height, (h*(((diff)/2)/(h+(diff+0.5)/2))))
                        allDetections = np.append(allDetections, imgDetections)
                        imgDetections = [[],[],[],[]]
    #print(load_height)
    maxRatio = np.reshape(maxRatio, (-1, 2))
    scale = ((h-np.average(load_height))/h)
    if load_width == 0:
        scale = 1
    for i in range(int(len(allDetections)/4)):

        allDetections[i*4+2] = allDetections[i*4+2]*scale
        allDetections[i*4+3] = allDetections[i*4+3]*scale
    allDetections = np.reshape(allDetections, (-1, 4))
    
    #print(maxRatio)
    minWidth = []
    for i in range(len(maxRatio)):
        if maxRatio[i][0] < maxRatio[i][1]:
            minWidth = np.append(minWidth, maxRatio[i][0])
        else:
            minWidth = np.append(minWidth, maxRatio[i][1])
    
    minWidth = np.sort(minWidth)
    minWidth = minWidth[0:int(np.ceil(len(minWidth)*keep_width)):1]
    widthAvg = np.average(minWidth)
    
    
    c = 0                       #temp variable to keep track of coeffecient
    w = 0                       #test variable to check if 'a' is close enough to correct
    ls = 0                      #long/short. If 0, the width is SHORT. if 1, the width is LONG
    for i in range(len(allDetections)):
        #print(allDetections[i])
        if allDetections[i][2] > allDetections[i][3]:
            l = allDetections[i][2]
            h = allDetections[i][3]
            #print(" l > h")
        else:
            h = allDetections[i][2]
            l = allDetections[i][3]
            #print(" l < h")
        if(force_diagonal == False):
            c = -0.25*(min([l, h])**2)
            #print("a max: ", a)
            #print("L (should be larger): ", l)
            #print("h (should be smaller): ", h)
            x1 = (-l + (l**2 + 4*c)**0.5)/(-2)
            y1 = (-h - (h**2 + 4*c)**0.5)/(-2)
            y2 = (-h + (h**2 + 4*c)**0.5)/(-2)
            #print("x1: ", x1)
            #print("y1: ", y1)
            #print("y2: ", y2)
            
            w = (y1**2 + x1**2)**0.5
            #print(w)
            #print(widthAvg)
            
            ls = (w < widthAvg)        
            
            c = c/2.0
            step = -c*(1 - 2*ls)
            
                   
            x1 = (-l - (l**2 + 4*c)**0.5)/(-2)
            y1 = (-h - (h**2 + 4*c)**0.5)/(-2)
            w = (y1**2 + x1**2)**0.5
            
            if (l > widthAvg) and (h > widthAvg):
                while abs(widthAvg - w) > 0.0001:
                    if widthAvg < w:
                        c = c + step
                        # print("a + step")
                    else:
                        c = c - step
                        #print("a - step")
    
                    if c < -0.25*(min([l, h])**2):
                        c = -0.25*(min([l, h])**2)
                        #print("a reset to max")
    
                    if c > 0:
                        c = 0
                        #print("a reset to 0")
    
                    step = step/1.5
    
                    
                    x1 = (-l + (l**2 + 4*c)**0.5)/(-2)
                    y1 = (-h - (h**2 + 4*c)**0.5)/(-2)
                    y2 = (-h + (h**2 + 4*c)**0.5)/(-2)
                    
                    if ls:
                        w = (y1**2 + x1**2)**0.5      # for shortening the hypothetical width
    
                    else:
                        w = (y2**2 + x1**2)**0.5      # for lengthening the hypothetical width
                    
                    
                if ls:
                    lengthGraphOut = np.append(lengthGraphOut, ((h-y1)**2 + (l-x1)**2)**0.5)        # this will write the SHORT hypothetical length
                else:
                    lengthGraphOut = np.append(lengthGraphOut, ((h-y2)**2 + (l-x1)**2)**0.5)        # this will write the LONG hypothetical length
            else:
                lengthGraphOut = np.append(lengthGraphOut, max([l, h]))                             # if the shortest side of the bounding box is smaller than the width estimate, then the longer side is used as the length.
        else:
            lengthGraphOut = np.append(lengthGraphOut, (h**2 + l**2)**0.5)
    for i in range(len(lengthGraphOut)):
        lengthGraphOut[i] = lengthGraphOut[i] * length_multiplier
        #lengthGraphOut[i] = (lengthGraphOut[i] - 10)*2
    
    
                
        
    # print(lengthGraphOut)
    # print(np.average(lengthGraphOut))
    # print(np.std(lengthGraphOut))
    if(save_img):
        plt.figure()
        plt.hist(lengthGraphOut, bins=160, range=(0, 40), density='true')
        plt.axvline(np.average(lengthGraphOut), color='k', linestyle='dashed', linewidth=1)
        plt.axvline(np.average(lengthGraphOut) + np.std(lengthGraphOut), color='r', linestyle='dashed', linewidth=1)
        plt.axvline(np.average(lengthGraphOut) - np.std(lengthGraphOut), color='r', linestyle='dashed', linewidth=1)
        s = "Average: " + str(round(np.average(lengthGraphOut), 4))
        plt.text( np.average(lengthGraphOut) + 4, 1.5/np.average(lengthGraphOut), s)
        s = "Deviation: " + str(round(np.std(lengthGraphOut), 4))
        plt.text( np.average(lengthGraphOut) + 4, 1.5/np.average(lengthGraphOut) - 0.02, s)
        plt.title("Estimated Billet Lengths" + str())
        plt.xlabel("Billet length (cm)")
        plt.ylabel("Frequency")
    
        if(os.path.exists(source + '/graphs') != True):
            os.mkdir(source + '/graphs')
    
        plt.savefig(source + '/graphs/' + 'hr' + str(load_width) + '_c' + str(conf) + '_m' + str(length_multiplier) + '_f' + str(filter_ratio) + '.png')
        print("Done")
    
    # print(source)
    # print(np.average(lengthGraphOut))
    return(np.average(lengthGraphOut))

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=float, default=0.48)
parser.add_argument('--source', type=str, default='runs/detect/exp12', help='directory of .txt files relative to this program')
parser.add_argument('--length_multiplier', type=float, default=142)
parser.add_argument('--load_width', type=float, default=0.54)
parser.add_argument('--filter_ratio', type=float, default=4.3)
parser.add_argument('--keep_width', type=float, default=0.27)
parser.add_argument('--save_img', action='store_true', default = True)
parser.add_argument('--force_diagonal', action='store_true', default=False)

opt = parser.parse_args()

if __name__ == "__main__":
    lengthCalc2(**vars(opt))
