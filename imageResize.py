import urllib2
import urllib
import json
import numpy as np
import cv2
import untangle
import time
import datetime
import os
maxsize = 512


path = "tmp"

if __name__ == "__main__":


    files= os.listdir(path)
    count = 0
    for file in files:
        if "jpg" in file or "png" in file:
            resp = open(path+"/"+file, 'rb')
            print (path+"/"+file)

            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            if height > width:
                scalefactor = (maxsize*1.0) / width
                res = cv2.resize(image,(int(width * scalefactor), int(height*scalefactor)), interpolation = cv2.INTER_CUBIC)
                cropped = res[0:maxsize,0:maxsize]
            if width > height:
                scalefactor = (maxsize*1.0) / height
                res = cv2.resize(image,(int(width * scalefactor), int(height*scalefactor)), interpolation = cv2.INTER_CUBIC)
                center_x = int(round(width*scalefactor*0.5))
                #print center_x
                cropped = res[0:maxsize,center_x - maxsize/2:center_x + maxsize/2]
            cv2.imwrite("val/"+str(count)+".jpg",cropped)
            count += 1
