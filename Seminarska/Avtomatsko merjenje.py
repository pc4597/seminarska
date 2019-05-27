
# coding: utf-8

# In[333]:


import cv2 as cv
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from rvlib import showImage
from rvlib import findLocalMax
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import scipy.ndimage as ni
import PIL.Image as Image

# nalozi knjiznico za morfoloske operacije 
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage.measure import label

#Izberi stran
#0 - neaktivna
#1 - aktivna 
leva = 0
desna = 1

"""REALNI PROBLEM!!!"""

########################
slika = Image.open("data/RTG3.png") #Vse nastavitve so primerne le za primer RTG3.png
img = np.array(slika)
imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgO = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#showImage(imgG)

imgG_shape = imgG.shape
#print(imgG_shape)
imgG_shape_x = imgG_shape[1]
imgG_shape_y = imgG_shape[0]

#Prikaži le polovico slike 
if leva == 1:
    imgG = imgG[0:imgG_shape_y,0:int(imgG_shape_x/2)] #levo
    imgO = imgO[0:imgG_shape_y,0:int(imgG_shape_x/2)] #levo
    imgO = imgO+50
    imgO = -imgO
if desna == 1:
    imgG = imgG[0:imgG_shape_y,int(imgG_shape_x/2):imgG_shape_x] #desno
    imgO = imgO[0:imgG_shape_y,int(imgG_shape_x/2):imgG_shape_x] #desno

showImage(imgG, "iščem kroge")
iThr = 200
imgT = 255 * (imgG < iThr).astype('uint8')
imgG = np.array(imgT)

prag1 = 200
prag2 = 50
edge = cv.Canny(imgG, prag1, prag2)
#showImage(edge)
krogi = np.array([])
krogi = cv.HoughCircles(imgG, cv.HOUGH_GRADIENT, 2 ,10000, krogi, prag1, prag2, 20, 200)

#showImage(imgG)
stevilo_krogov = krogi.shape[1]
#print(stevilo_krogov)

for i in range(stevilo_krogov):
    center = tuple(krogi[0, i, :2])
    cv.circle(imgG,center, krogi[0, i, 2], (0, 0, 0))
#showImage(imgG,"krog")

#računanje območja zanimanja
radij = int((krogi[0, i, 2] + 0.006*imgG_shape_x))
x_center, y_center = center
x_center = int(x_center)
y_center = int(y_center)
#showImage(imgT,'krogi')


if leva == 1:
    y_obrezana_minus = y_center - radij
    y_obrezana_plus = y_center + radij + int(radij*0.2)
    x_obrezana_minus = x_center - radij
    x_obrezana_plus = x_center + radij + int(radij*0.2)

if desna == 1:
    y_obrezana_minus = y_center - radij
    y_obrezana_plus = y_center + radij - int(radij*0.2)
    x_obrezana_minus = x_center - radij
    x_obrezana_plus = x_center + radij - int(radij*0.2)

if y_center - radij < 0:
    y_obrezana_minus = 0
if y_center + radij > int(imgG_shape_y):
    y_obrezana_plus
if x_center - radij < 0:
    x_obrezana_minus = 0
if x_center + radij > int(imgG_shape_x):
    x_obrezana_plus
    
imgG = imgG[y_obrezana_minus:y_obrezana_plus,x_obrezana_minus:x_obrezana_plus]
#showImage(imgG,'Povečana, premaknjena')

imgO = imgO[y_obrezana_minus:y_obrezana_plus,x_obrezana_minus:x_obrezana_plus]
#showImage(imgO,'obrezana')

########################

# izberi prag
iThr = int(np.average(imgO))
imgT = 255 * (imgO < iThr).astype('uint8')
imgT = np.array(imgT)
#imgT = imgG
#showImage(imgT, 'Upragovanje')

#Osnova za masko
mask1 = np.ones(shape = imgO.shape, dtype = "uint8")
mask2 = np.zeros(shape = imgO.shape, dtype = "uint8")
#print(imgG_shape*0.5)

imgO_shape = imgO.shape
#print(imgG_shape)
imgO_shape_x = imgO_shape[1]
imgO_shape_y = imgO_shape[0]


# Risanje maske
if leva == 1:
    cv.circle(mask1, (int(imgO_shape_x/2) - int(radij*0.05),(int(imgO_shape_y/2))), radij, (0,0,0), -1)
    cv.rectangle(img = mask2, pt1 = (int(imgO_shape_x*0.4), int(imgO_shape_y)), pt2 = (0, int(imgO_shape_y*0.4)), color = (255,255,255), thickness = -1)

if desna == 1:
    cv.circle(mask1, (int(imgO_shape_x/2) + int(radij*0.1),(int(imgO_shape_y/2))), radij, (0,0,0), -1)
    cv.rectangle(img = mask2, pt1 = (int(imgO_shape_x*0.4), int(imgO_shape_y)), pt2 = (int(imgO_shape_x), int(imgO_shape_y*0.4)), color = (255,255,255), thickness = -1)

#showImage(mask1)
#showImage(mask2)
mask = mask1-mask2
#showImage(mask, 'Maska')
imgT = imgT-mask
#showImage(imgT, 'Slika - Maska')

# dilacija
#imgD = dilation(imgT, disk(5))
#showImage(imgD,'dilation')
imgT = opening(imgT, disk(2))
#showImage(imgT, 'Odpiranje')

imgT = cv.medianBlur(imgT, 11) 
#showImage(imgT, 'Glajenje')

edge = cv.Canny(imgT, prag1, prag2,True)

#showImage(edge,'edge')

corners = cv.cornerHarris(edge, 5, 3, 1/16, True)
corners[corners < 0] = 0 #odrezemo odzive, ki so negativni
corners = (corners - np.min(corners)) / (np.max(corners) - np.min(corners))
#showImage(corners, 'corners')

#showImage(255 * (corners > 0.2).astype('uint8'))


# Test funkcije findLocalMax
oLocalMax = findLocalMax(corners, corners.max() * 0.01) 
#showImage(edge, 'Oslonilne točke')
#plt.plot(oLocalMax[:,0], oLocalMax[:,1], 'o', markersize=5.0)
#print(oLocalMax)



#----------------------------------------------------------------------------
#print(oLocalMax.shape)
oLocalMax_shape = oLocalMax[:,0].shape #dimenzije
oLocalMax_shape = oLocalMax_shape[0] 

x = []
y = []

for i in range(oLocalMax_shape):
    x.append(oLocalMax[i,0])
    y.append(oLocalMax[i,1])
#print("MAx x: ")
#print(max(x))
#print("MAx y: ")
#print(max(y))

#print("Index prve točke: ")
#print(x.index(max(x)))
#print("Index druge točke: ")
#print(y.index(max(y)))

if leva == 1:
    x1 = oLocalMax[x.index(min(x)),0]
    y1 = oLocalMax[x.index(min(x)),1]
    x2 = oLocalMax[y.index(max(y)),0]
    y2 = oLocalMax[y.index(max(y)),1]

if desna == 1:
    x1 = oLocalMax[x.index(max(x)),0]
    y1 = oLocalMax[x.index(max(x)),1]
    x2 = oLocalMax[y.index(max(y)),0]
    y2 = oLocalMax[y.index(max(y)),1]

#Izpis teh dveh točk
print("Točka A: (",x1,",",y1,") (modra)")
print("Točka B: (",x2,",",y2,") (oranžna)")

#Iskanje tretje točke na elipsi
tretja_tocka = 0 #inicializacija indeksa tretje točke
d1 = 0.6
d2 = 0.3
#print(edge.shape[1]*d1,edge.shape[1]*d2,edge.shape[0]*d1,edge.shape[0])

tretja_tocka = []

if leva == 1:
    d1 = 0.3
    d2 = 0.3
    #IF stavki določajo okno, v katerem poiščemo točko na elipsi
    for n in range(oLocalMax_shape):
        if oLocalMax[n,0] > 0:
            if oLocalMax[n,0] < edge.shape[1]*d1:
                if oLocalMax[n,1] > edge.shape[0]*d2:
                    if oLocalMax[n,1] < edge.shape[0]:
                        tretja_tocka.append(n)
if desna == 1:
    d1 = 0.6
    d2 = 0.3
    #IF stavki določajo okno, v katerem poiščemo točko na elipsi
    for n in range(oLocalMax_shape):
        if oLocalMax[n,0] > edge.shape[1]*d1:
            if oLocalMax[n,0] < edge.shape[1]:
                if oLocalMax[n,1] > edge.shape[0]*d2:
                    if oLocalMax[n,1] < edge.shape[0]:
                        tretja_tocka.append(n)


#Zapis točke, ki je bila najdena
x3 = oLocalMax[tretja_tocka[0],0]
y3 = oLocalMax[tretja_tocka[0],1]
print("Točka P: (",x3,",",y3,") (zelena)")



#izračun prvega kota
kot1 = np.arctan((x1 - x2)/(y1 - y2)) 
kot1 = np.degrees(kot1)
kot1 = 90 - np.abs(kot1)


#naklon prve daljice
k1 = (y2 - y1)/(x2 - x1)
n1 = int(y1 - k1*x1)
    
k2 = -1/k1
    
n2 = y3 - k2*x3
a = k1
b = -1
c = n1    
          
x4 = (n2 - n1)/(k1 - k2)
y4 = x4*k2 + n2
x4 = int(x4)
y4 = int(y4)
print("Točka K: (",x4,",",y4,") (rdeča)")


x_sredina = int((x1/2)+(x2/2))
y_sredina = int((y1/2)+(y2/2)) 

#izračun razdalij med točkami
#PK = np.abs(a*x3 + b*y3 + c)/(np.sqrt(a**2 + b**2))    
PK = np.sqrt((x3-x4)**2 + (y3-y4)**2)
AK = np.sqrt((x1-x4)**2 + (y1-y4)**2)
BK = np.sqrt((x2-x4)**2 + (y2-y4)**2) 
KO = np.sqrt((x4-x_sredina)**2 + (y4-y_sredina)**2)
AO = np.sqrt((x1-x_sredina)**2 + (y1-x_sredina)**2)
#print("PK", PK)
#print("AK", AK)
#print("BK", BK)
#Nariši črto med točko A in B
   
    
#izračun drugega kota
kot2 = np.arcsin(PK/(np.sqrt(AK*BK)))
kot2 = np.degrees(kot2)


AB = np.sqrt((x2-x1)**2 + (y2-y1)**2)
AB_polovic = int(AB/2)
R = int(PK*np.sqrt(1/(1-(KO**2/AO**2))))

#Nariši črto, elipso in polkrog
if leva == 1:
    imgO = -imgO
    imgO = imgO-50
    cv.ellipse(imgO,(x_sredina,y_sredina),(AB_polovic,R),int(kot1),0,360,(0,0,0),1)
    cv.ellipse(imgO,(x_sredina,y_sredina),(AB_polovic,AB_polovic),-int(kot1),90,-90,(0,0,0),1)
    cv.line(imgO,(x1,y1),(x2,y2),(0,0,0),1) 
if desna == 1:
    cv.ellipse(imgO,(x_sredina,y_sredina),(AB_polovic,R),-int(kot1),0,360,(255,255,255),1)
    cv.ellipse(imgO,(x_sredina,y_sredina),(AB_polovic,AB_polovic),-int(kot1),0,-180,(255,255,255),1)
    cv.line(imgO,(x1,y1),(x2,y2),(255,255,255),1) 

#Prikaz slike in točk
showImage(imgO, 'Končna slika')
plt.plot(x1,y1, 'o', markersize=10.0)
plt.plot(x2,y2, 'o', markersize=10.0)
plt.plot(x3,y3, 'o', markersize=10.0)   
plt.plot(x4,y4, 'o', markersize=10.0)   
print("Prvi kot je: ",round(kot1,1),"°")
print("Drugi kot je: ",round(kot2,1),"°")

