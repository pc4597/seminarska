
# coding: utf-8

# In[2]:


"""3D model"""

import cv2 as cv
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from rvlib import showImage
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import scipy.ndimage as ni
import PIL.Image as Image




slika = Image.open("data/polna_elipsa30.png")
img = np.array(slika)
imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#imgG = imgG[280:420,200:350] #leva
#imgG = imgG[280:420,600:750] #desna
#showImage(imgG, 'Originalna slika')
imgG = cv.medianBlur(imgG, 9) 

prag1 = 60
prag2 = 50
edge = cv.Canny(imgG, prag1, prag2)

#showImage(edge)

corners = cv.cornerHarris(edge, 3, 3, 1/16)
corners[corners < 0] = 0 #odrezemo odzive, ki so negativni
corners = (corners - np.min(corners)) / (np.max(corners) - np.min(corners))
#showImage(corners)

#showImage(255 * (corners > 0.2).astype('uint8'))

def findLocalMax(iImage, iThreshold = None):
    dy, dx = iImage.shape
    oLocalMax = []
    #nicti in zadnji piksel izpustimo, ker imamo matriko 3x3 (gledamo glede na sredinskega)
    for y in range(1, dy - 1):
        for x in range(1, dx - 1):
            cval = iImage[y, x] #vrednost slikovnega elementa na tej lokaciji
            #preverimo, ce je uporabnik dolocil threshold, ga bomo upostevali
            if iThreshold is not None:
                if cval < iThreshold:
                    continue #ne izvajaj naprej, ampak pojdi v naslednji krog
            gx, gy = np.meshgrid([x - 1, x, x + 1], 
                                 [y - 1, y, y + 1], sparse = False) #nekje znotraj slike smo, dobimo dve matriki
            gx = gx.flatten()
            gy = gy.flatten()
            
            cvaldiff = iImage[gy, gx] - cval #diference
            cvaldiff[int(np.floor(len(gy)/2))] = -1 #centraln o vrednost damo na vr. -1
            
            #ce je max vr. manjsa od 0, dobimo lokalni max
            #gledamo, ce je v okolici gledanega piksla nek piksel vecji, ga preskocimo
            #iscemo, dokler ni piksel v sredini najvecji od vseh
            if cvaldiff.max() < 0:
                oLocalMax.append((x, y))
    return np.array(oLocalMax)

# Test funkcije findLocalMax
oLocalMax = findLocalMax(corners, corners.max() * 0.01) 
showImage(edge, 'Robovi')
plt.plot(oLocalMax[:,0], oLocalMax[:,1], 'o', markersize=5.0)
#print(oLocalMax)



#----------------------------------------------------------------------------
#print(oLocalMax.shape)
oLocalMax_shape = oLocalMax[:,0].shape #dimenzije
oLocalMax_shape = oLocalMax_shape[0] 

x = [0 for i in range(oLocalMax_shape*oLocalMax_shape)]
y = [0 for i in range(oLocalMax_shape*oLocalMax_shape)]

for i in range(oLocalMax_shape):
    x[i] = oLocalMax[i,0]
    y[i] = oLocalMax[i,1]
#print("MAx x: ")
#print(max(x))
#print("MAx y: ")
#print(max(y))

#print("Index prve točke: ")
#print(x.index(max(x)))
#print("Index druge točke: ")
#print(y.index(max(y)))

x1 = oLocalMax[x.index(max(x)),0]
y1 = oLocalMax[x.index(max(x)),1]
x2 = oLocalMax[y.index(max(y)),0]
y2 = oLocalMax[y.index(max(y)),1]

#Izpis teh dveh točk
print("Prva točka: ",x1,",",y1)
print("Druga točka: ",x2,",",y2)

#Iskanje tretje točke na elipsi
tretja_tocka = 0 #inicializacija indeksa tretje točke
d1 = 0.3
d2 = 0.6
#print(edge.shape[1]*d1,edge.shape[1]*d2,edge.shape[0]*d1,edge.shape[0])


tretja_tocka = []
#IF stavki določajo okno, v katerem poiščemo točko na elipsi
for n in range(oLocalMax_shape):
    if oLocalMax[n,0] > edge.shape[1]*d1:#400
        if oLocalMax[n,0] < edge.shape[1]:#700
            if oLocalMax[n,1] > edge.shape[0]*d2:#400
                if oLocalMax[n,1] < edge.shape[0]:#700
                    tretja_tocka.append(n)


#Zapis točke, ki je bila najdena    
x3 = oLocalMax[tretja_tocka[0],0]
y3 = oLocalMax[tretja_tocka[0],1]
print("Tretja točka: ",x3,",",y3)
print("")
print("")

#izračun prvega kota
kot1 = np.arctan((x1 - x2)/(y1 - y2)) 
kot1 = np.degrees(kot1)
kot1 = 90 - np.abs(kot1)


#naklon prve daljice
k1 = (y2 - y1)/(x2 - x1)
n1 = int(y1 - k1*x1)

#naklon druge daljice, ki je pravokotna na prvo
k2 = -1/k1
#k2 = 2-k2


#izračun koeficienta
n2 = y3 - k2*x3

#izračun točke K (4. točke)
x4 = (n2 - n1)/(k1 - k2)
y4 = x4*k2 + n2
x4 = int(x4)
y4 = int(y4)


x_sredina = int((x1/2)+(x2/2))
y_sredina = int((y1/2)+(y2/2)) 
#print("sredina",x_sredina,y_sredina)

x_start = x_sredina
#print("x_start", x_start)
y_start = y_sredina
#print("y_start", y_start)

cv.line(edge,(x1,y1),(x2,y2),(255,255,255),1) 


#x3 = 290
#y3 = 524

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
cv.line(imgRGB,(x1,y1),(x2,y2),(255,255,255),1)    
    
#izračun drugega kota
kot2 = np.arcsin(PK/(np.sqrt(AK*BK)))
kot2 = np.degrees(kot2)
kot2 = np.abs(kot2)
print("kot2 pred", kot2)

#Pri "idealnih" modelih je potrebno narediti korekcijo drugega kota ...
#če je kot2 večji od 13, ga povečamo
k1 = 0
k1 = kot2*0.073
if kot2 > 14:
    kot2 = kot2*k1

#če je kot2 manjši od 13, ga zmanjšamo
k2 = 0
k2 = kot2*0.080
if kot2 < 13:
    kot2 = kot2*k2

#če je kot2 ravno med 13 in 14, se ne zgodi nič
kot2 = kot2
    

AB = np.sqrt((x2-x1)**2 + (y2-y1)**2)
AB_polovic = int(AB/2)
R = int(PK*np.sqrt(1/(1-(KO**2/AO**2))))
cv.ellipse(imgRGB,(x_sredina,y_sredina),(AB_polovic,R),-int(kot1),0,360,(255,0,0),2)

#Prikaz slike in točk
showImage(imgRGB, 'Točke')
plt.plot(x1,y1, 'o', markersize=10.0)
plt.plot(x2,y2, 'o', markersize=10.0)
plt.plot(x3,y3, 'o', markersize=10.0)   
#plt.plot(x4,y4, 'o', markersize=5.0)   
#plt.plot(x_sredina,y_sredina, 'o', markersize=10.0)
#plt.plot(x_r,y_r, 'o', markersize=5.0) 

print("Izračunani prvi kot je: ",round(kot1,3),"°")
print("Kot etalona je 40°. Napaka prvega kota je: +-",round(abs(40-kot1),3),"°")
print("Izračunani drugi kot je: ",round(kot2,3),"°")
print("Kot etalona je 30°. Napaka drugega kota je: +-",round(abs(30-kot2),3),"°")

