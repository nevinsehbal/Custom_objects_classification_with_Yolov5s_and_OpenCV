## Kuru Uzum ve Kabak Cekirdegi Detection and Labelling (from video file) ##

# Bu kod iki ayrı hsv color space maskesi içeir.
# Verilen bir videonun her bir frame'i üzerine maske uygular.
# Maskeleme sonucu oluşan binary image'lardan contour çıkarır.
# Minimum Contour alanı koşulunu sağlayan, iki maskeye ait objelerin konumları tespit edilir.
# Tespit edilen objelerin frame'deki konumları xml formatında kaydedilir.
# Her frame .jpg dosya formatında kaydedilir.
# Etiketleme işlemi otomatik olarak tamamlanmış olur.

import cv2
import numpy as np
import xml.etree.cElementTree as ET
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('videopath',help='Path to the input video file (mp4)')
p = parser.parse_args()
# Determine below directories before execution:
videopath = p.videopath
path = os.path.dirname(p.videopath)
os.chdir(path)
path = os.getcwd()
labeldir = r"\annotations"
xmlsavedirr = path + labeldir
imdir = r"\frames"
imagesavedir = path + imdir
if not os.path.exists(xmlsavedirr):
    os.mkdir(xmlsavedirr)
if not os.path.exists(imagesavedir):
    os.mkdir(imagesavedir)

#Resize for good visualization, cv2.imshow komutu icin helper fonksiyon
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

#Function to write for the proper xml format
def writeXML(contours_uzum,contours_kabak,counter,imsize,xmlsavedir):
    impath=r"C:\Users\me"+r"\\"+str(counter)+"img.jpg"
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder").text = "images"
    filename = ET.SubElement(root, "filename").text = r"\\"+str(counter)+"img.jpg"
    path = ET.SubElement(root, "path").text = impath
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(imsize[0])
    ET.SubElement(size, "height").text = str(imsize[1])
    ET.SubElement(size, "depth").text = str(imsize[2])
    segmented = ET.SubElement(root, "segmented").text = "0"
    for i in range(len(contours_uzum)):
        cnt1 = contours_uzum[i]
        x, y, w, h = cv2.boundingRect(cnt1)
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        object = ET.SubElement(root, "object")
        ET.SubElement(object, "name").text = "uzum"
        ET.SubElement(object, "pose").text = "Unspecified"
        ET.SubElement(object, "truncated").text = "0"
        ET.SubElement(object, "difficult").text = "0"
        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    for j in range(len(contours_kabak)):
        cnt2 = contours_kabak[j]
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        x2min, y2min, x2max, y2max = x2, y2, x2 + w2, y2 + h2
        object = ET.SubElement(root, "object")
        ET.SubElement(object, "name").text = "kabak"
        ET.SubElement(object, "pose").text = "Unspecified"
        ET.SubElement(object, "truncated").text = "0"
        ET.SubElement(object, "difficult").text = "0"
        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x2min)
        ET.SubElement(bndbox, "ymin").text = str(y2min)
        ET.SubElement(bndbox, "xmax").text = str(x2max)
        ET.SubElement(bndbox, "ymax").text = str(y2max)
    tree = ET.ElementTree(root)
    tree.write(xmlsavedir + r'\\' + str(counter) + 'img.xml')

cap = cv2.VideoCapture(videopath)

if (cap.isOpened()== False):
  print("Error opening video stream or file")

lower_hsv_uzum = np.array([90, 10, 0])
higher_hsv_uzum = np.array([180, 180, 180])
lower_hsv_kabak = np.array([0, 25, 0])
higher_hsv_kabak = np.array([40, 255, 255])

counter = 2000
while(cap.isOpened()):
  counter += 1
  ret, image = cap.read()
  if ret == True:
    frame = ResizeWithAspectRatio(image, width=640) # Resize by width OR
    imagesavedirr = imagesavedir +"\\"+str(counter)+"img.jpg"
    cv2.imwrite(imagesavedirr, frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_hsv_uzum, higher_hsv_uzum)
    mask2 = cv2.inRange(hsv, lower_hsv_kabak, higher_hsv_kabak)

    frame1 = cv2.bitwise_and(frame, frame, mask=mask1)
    frame2 = cv2.bitwise_and(frame, frame, mask=mask2)

    contours_uzum, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_kabak, hierarchy2 = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    uzum_number = 0
    kabak_number = 0
    min_area = 100
    valid_uzum = []
    valid_kabak = []

    for i in range(len(contours_uzum)):
        #print(cv2.contourArea(contours_uzum[i]))
        if min_area < cv2.contourArea(contours_uzum[i]) and hierarchy[0,i,3]==-1:
            cnt1 = contours_uzum[i]
            valid_uzum.append(cnt1)
            x,y,w,h = cv2.boundingRect(cnt1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            xmin,ymin,xmax,ymax = x,y,x+w,y+h
            uzum_number += 1
            #print(xmin,ymin,xmax,ymax)
    for i in range(len(contours_kabak)):
        if min_area < cv2.contourArea(contours_kabak[i]) and hierarchy2[0,i,3]==-1:
            cnt2 = contours_kabak[i]
            valid_kabak.append(cnt2)
            x2,y2,w2,h2 = cv2.boundingRect(cnt2)
            cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,0,255),2)
            x2min,y2min,x2max,y2max = x2,y2,x2+w2,y2+h2
            kabak_number += 1
            #print(x2min,y2min,x2max,y2max)

    imsize = frame.shape
    print('Su anki resimde {} adet kuru üzüm, {} adet kabak cekirdegi vardir.'.format(uzum_number, kabak_number))
    #print(imsize[0],imsize[1],imsize[2])
    writeXML(valid_uzum, valid_kabak, counter, imsize,xmlsavedirr)

    cv2.imshow('contoured', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  else:
    break

print('Finished successfully!')
cap.release()
cv2.destroyAllWindows()