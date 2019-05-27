
# coding: utf-8

# In[1]:


#Izvedeno v jupyter-ju

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
import rvlib as rv
import PIL.Image as Image
import cv2 as cv

get_ipython().run_line_magic('matplotlib', 'notebook')

#Odpri sliko zanimanja
slika = Image.open("data/RTG.jpg").convert('L')
slika1 = np.array(slika)

#rv.showImage(slika1, "Original")
slika1 = slika1[190:430,1140:1370] #Obreži sliko na območje zanimanja

#------------------------------------------------------------------------
#Novo okno
#Uporabnik poklika točke --> 
#1. točka = A, 
#2. točka = B, 
#3. točka = P

tockeABP = []
def onclick(event):
    if event.key == 'shift': 
        x, y = event.xdata, event.ydata
        tockeABP.append((x, y))
        ax.plot(x, y,'or')
        fig.canvas.draw()
        
get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(slika1, cmap='gray')    
ka = fig.canvas.mpl_connect('button_press_event', onclick) 
#------------------------------------------------------------------------
#Novo okno

step = 1
if len(tockeABP) >= 2:

    tocka1 = tockeABP[-3]
    tocka2 = tockeABP[-2]
    tocka3 = tockeABP[-1]
    
    x1 = tocka1[0]
    y1 = tocka1[1]
    x2 = tocka2[0]
    y2 = tocka2[1]
    x3 = tocka3[0]
    y3 = tocka3[1]
  
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    x3 = int(x3)
    y3 = int(y3)

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
    
    PK = np.abs(a*x3 + b*y3 + c)/(np.sqrt(a**2 + b**2))    
    AK = np.sqrt((x1-x4)**2 + (y1-y4)**2)
    BK = np.sqrt((x2-x4)**2 + (y2-y4)**2)   
    
      
    #izračun prvega kota
    kot1 = np.arctan((x1 - x2)/(y1 - y2)) 
    kot1 = np.degrees(kot1)
    kot1 = 90 - np.abs(kot1)
    print(kot1)
    kot1 = round(kot1, 1)
    
    #izračun drugega kota
    kot2 = np.arcsin(PK/(np.sqrt(AK*BK)))
    kot2 = np.degrees(kot2)
    kot2 = np.abs(kot2)
    print(kot2)
    kot2 = round(kot2, 1)

    #izris daljic
    cv.line(slika1,(x1,y1),(x2,y2),(255,255,255),1)
    cv.line(slika1,(x3,y3),(x4,y4),(255,255,255),1)
    
    BA = np.sqrt((x1-x2)**2 + (y1-y2)**2)/2
    BA = int(BA)
    PK = int(PK)
    print(PK,AK,BK)
    
    x = int((x2-x1)/2)
    y = int((y2-y1)/2)
   
    
    showImage(slika1)

    print("Prvi kot je:", kot1, "stopinj.")
    print("Drugi kot je:", kot2, "stopinj.")

