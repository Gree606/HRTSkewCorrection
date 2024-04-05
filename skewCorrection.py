"""
@author: Gree
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
import traceback
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict

mpl.rcParams['legend.fontsize'] = 10

pd.set_option('display.expand_frame_repr', False)
fn=0
# path='./result/'

img = cv.imread('./TestImages/plain-paper-images/1.png')
# img = cv.imread('./sample_images/a.png')

# In[wordSegment]
#*****************************************************************************#
def wordSegment(textLines):
    wordImgList=[]
    counter=0
    cl=0
    for txtLine in textLines:
        if len(img.shape) > 2 and img.shape[2] > 1:
            gray = cv.cvtColor(txtLine, cv.COLOR_BGR2GRAY)
        else:
            gray = txtLine
        th, threshed = cv.threshold(gray, 170, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
        final_thr = cv.dilate(threshed,None,iterations = 20)

        plt.imshow(final_thr)
        plt.show()
        
        contours, hierarchy = cv.findContours(final_thr,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv.boundingRect(c) for c in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0], reverse=False))
       
        for cnt in contours:
            area = cv.contourArea(cnt)
 
#            print area
            if area > 10000:
                # print ('Area= ',area)
                x,y,w,h = cv.boundingRect(cnt)
                # print (x,y,w,h)
                letterBgr = txtLine[0:txtLine.shape[1],x:x+w]
                wordImgList.append(letterBgr)
 
                cv.imwrite("./result/words/" + str(counter) +".jpg",letterBgr)
                counter=counter+1
        cl=cl+1
       
    return wordImgList
#*****************************************************************************#
    
# In[fitToSize]
#*****************************************************************************#
def fitToSize(thresh1):
    
    mask = thresh1 > 0
    coords = np.argwhere(mask)

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    cropped = thresh1[x0:x1,y0:y1]
    return cropped
   
#*****************************************************************************#
    
# In[lineSegment]
#*****************************************************************************#
def lineSegment(img):
    if img is None:
        print("Error: Unable to load image")
    else:
        # Convert the image to grayscale if it's not already in grayscale
        if len(img.shape) > 2 and img.shape[2] > 1:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img
    th, threshed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    cv.imwrite('./MidImages/cvImg.png',threshed)
    upper=[]
    lower=[]
    flag=True
    for i in range(threshed.shape[0]):

        col = threshed[i:i+1,:]
        cnt=0
        if flag:
            cnt=np.count_nonzero(col == 255)
            if cnt >0:
                upper.append(i)
                flag=False
        else:
            cnt=np.count_nonzero(col == 255)
            if cnt <2:
                lower.append(i)
                flag=True
    textLines=[]
    if len(upper)!= len(lower):lower.append(threshed.shape[0])
#    print upper
#    print lower
    for i in range(len(upper)):
        timg=img[upper[i]:lower[i],0:]
        
        if timg.shape[0]>5:
#            plt.imshow(timg)
#            plt.show()
            timg=cv.resize(timg,((timg.shape[1]*5,timg.shape[0]*8)))
            textLines.append(timg)

    return textLines

#*****************************************************************************


#################################Image Straighten######################################



# In[deskew]:
def deskew(img):
    thresh=img
    edges = cv2.Canny(thresh,50,200,apertureSize = 3)
    
    lines = cv2.HoughLines(edges,1,np.pi/1000, 55)
    try:
        d1 = OrderedDict()
        for i in range(len(lines)):
            for rho,theta in lines[i]:
                deg = np.rad2deg(theta)
#                print(deg)
                if deg in d1:
                    d1[deg] += 1
                else:
                    d1[deg] = 1
                    
        t1 = OrderedDict(sorted(d1.items(), key=lambda x:x[1] , reverse=False))
        print(list(t1.keys())[0],'Angle' ,thresh.shape)
        non_zero_pixels = cv2.findNonZero(thresh)
        center, wh, theta = cv2.minAreaRect(non_zero_pixels)
        angle=list(t1.keys())[0]
        if angle>160:
            angle=180-angle
        if angle<160 and angle>20:
            angle=12        
        root_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rows, cols = img.shape
        rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
        
    except:
        rotated=img
        pass
    return rotated

def unshear(img):

    gray = img
    thresh = img.copy()
    #print(thresh)
    # plt.imshow(thresh)
    # plt.show()
    trans = thresh.transpose()

    arr=[]
    for i in range(thresh.shape[1]):
        arr.insert(0,trans[i].sum())

    arr=[]
    for i in range(thresh.shape[0]):
        arr.insert(0,thresh[i].sum())
    
    y = thresh.shape[0]-1-np.nonzero(arr)[0][0]
    y_top = thresh.shape[0]-1-np.nonzero(arr)[0][-1]

    trans1 = thresh.transpose()
    sum1=[]
    for i in range(trans1.shape[0]):
        sum1.insert(i,trans1[i].sum())

    height = y - y_top
    max_value = 255*height
    prev_num = len([i for i in sum1 if i>=(0.6*max_value)])
    final_ang = 0

    # # print(arr)
    # # print(x,y)
    for ang in range(-10,11):
        ang=ang/2
        thresh = gray.copy()
        #print(thresh[0].shape)
        #print(ang)
        # print('Ang',ang)
        if ang>0:
            #print(ang)
            for i in range(y):
                temp = thresh[i]
                move = int((y-i)*(math.tan(math.radians(ang))))
                if move >= temp.size:
                    move = temp.size
                thresh[i][:temp.size-move]=temp[move:]
                thresh[i][temp.size-move:] = [0 for m in range(move)]
        else:
            #print(ang)
            for i in range(y):
                temp = thresh[i]
                move = int((y-i)*(math.tan(math.radians(-ang))))
                if move >= temp.size:
                    move = temp.size
                #print(temp[:-3])
                #print(temp[:temp.size-move].shape, thresh[i][move%temp.size:].shape)
                thresh[i][move:]=temp[:temp.size-move]
                thresh[i][:move]=[0 for m in range(move)]

#         plt.imshow(thresh)
#         plt.show()
        trans1 = thresh.transpose()
        sum1=[]
        for i in range(trans1.shape[0]):
            sum1.insert(i,trans1[i].sum())
        #print(sum1)
        num = len([i for i in sum1 if i>=(0.60*max_value)])
        #print(num, prev_num)
        if(num>=prev_num):
            prev_num=num
            final_ang = ang
        #plt.imshow(thresh)
        #plt.show()
    print("final_ang:", final_ang)

    thresh= gray.copy()
    if final_ang>0:
        for i in range(y):
            temp = thresh[i]
            move = int((y-i)*(math.tan(math.radians(final_ang))))
            if move >= temp.size:
                move = temp.size
            thresh[i][:temp.size-move]=temp[move:]
            thresh[i][temp.size-move:] = [0 for m in range(move)]
    else:
        for i in range(y):
            temp = thresh[i]
            move = int((y-i)*(math.tan(math.radians(-final_ang))))
            #print(move)
            if move >= temp.size:
                move = temp.size
            thresh[i][move:]=temp[:temp.size-move]
            thresh[i][:move]=[0 for m in range(move)]

#    plt.imshow(thresh)
#    plt.show()
    return thresh


# In[Main]:

def pad_with(vector, pad_width, iaxis, kwargs):
     pad_value = kwargs.get('padder', 40)
     vector[:pad_width[0]] = pad_value
     vector[-pad_width[1]:] = pad_value
     return vector

def clear_folder(folder_path):
    # Iterate over all files and subdirectories in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # Check if the item is a file
        if os.path.isfile(item_path):
            # Delete the file
            os.remove(item_path)
        # Check if the item is a subdirectory
        elif os.path.isdir(item_path):
            # Recursively delete the subdirectory and its contents
            shutil.rmtree(item_path)

#*****************************************************************************#
if __name__ == '__main__':
    try:
        clear_folder('/home/greeshma/CNN-for-character-Recogonition/Segmentation/Cursive_handwriting_recognition/TestImages/skewedSections')
        clear_folder('/home/greeshma/CNN-for-character-Recogonition/Segmentation/Cursive_handwriting_recognition/TestImages/line')
        clear_folder('/home/greeshma/CNN-for-character-Recogonition/Segmentation/Cursive_handwriting_recognition/TestImages/letterGrey')
        textLines=lineSegment(img)
        print ('No. of Lines',len(textLines))
        lineList=[]
        counter = 0
        fileNo=0
        fileNos=0
        for linesAll in textLines:
            height, width = linesAll.shape[:2]
            # print("Height:", height)
            # print("Width:", width)
            if(height>600):
                gray_image = cv2.cvtColor(linesAll, cv2.COLOR_BGR2GRAY)
                writeLoc1='./TestImages/skewedSections/val'+str(fileNo)+'.png'
                # print(writeLoc1)
                cv2.imwrite(writeLoc1, linesAll)
                fileNo=fileNo+1
                # ret, thresh = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                thresh = cv.threshold(gray_image,200,255,1)[1]
                # cv2.imwrite('./result/data/rTest0_5.png', thresh)
                thresh=np.pad(thresh, 100, pad_with, padder=0)
                # print("reached")
                
                thresh = cv.rotate(thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)
                thresh=np.pad(thresh, 200, pad_with, padder=0)
                deskewed_img=deskew(thresh)
                # cv2.imwrite('./result/data/rTest0_5.png', deskewed_img)
                sheared_img = unshear(thresh)
                cv2.imwrite('./result/data/rTest0_5.png', sheared_img)
                ret, thresh = cv2.threshold(sheared_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                
                # thresh=np.pad(thresh, 100, pad_with, padder=255)
                thresh=cv2.rotate(thresh,cv2.ROTATE_90_CLOCKWISE)
                writeLoc1='./TestImages/skewedSections/val'+str(fileNo)+'.png'
                # print(writeLoc1)
                cv2.imwrite(writeLoc1, thresh)
                fileNo=fileNo+1
                skewLines=lineSegment(thresh)
                print ('No. of skewed Lines in segment: ',len(skewLines))
                # print("writing skew image")
                for linesSkall in skewLines:
                    lineList.append(linesSkall)
                    writeLoc='./TestImages/line/val'+str(fileNos)+'.png'
                    # print(writeLoc)
                    print("Skewing or Overlapping is present at line", fileNos+1)
                    cv.imwrite(writeLoc,linesSkall)
                    fileNos=fileNos+1
            else:
                lineList.append(linesAll)
                writeLoc='./TestImages/line/val'+str(fileNos)+'.png'
                # print(writeLoc)
                cv.imwrite(writeLoc,linesAll)
                fileNos=fileNos+1  

        imgList=wordSegment(textLines)
        print ('No. of Words',len(imgList))
        for letterGray in imgList:
            # print ('LetterGray shape: ',letterGray.shape)
            gray = cv.cvtColor(letterGray, cv.COLOR_BGR2GRAY)
            th, letterGray = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
            letterGray = fitToSize(letterGray)
            writeLoc='./TestImages/letterGrey/val'+str(fileNo)+'.png'
            fileNo=fileNo+1
            cv.imwrite(writeLoc,letterGray) 
            letter2 = letterGray.copy()
            letterGray = cv.dilate(letterGray,None,iterations = 4)

    except Exception as e:
        print ('Error Message ',e)
        cv.destroyAllWindows()
        traceback.print_exc()
        pass

    traceback.print_exc()   

