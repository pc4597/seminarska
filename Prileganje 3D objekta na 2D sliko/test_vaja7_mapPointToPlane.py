import os
import subprocess
import mrcfile
import numpy as np
import PIL.Image as Image
from matplotlib import pyplot
from os.path import join
import rvlib
from scipy.optimize import fmin, minimize
from scipy.interpolate import interpn
import scipy.ndimage as ni
from mpl_toolkits.mplot3d import Axes3D
from drr import VolumeRenderer

def preprocess_ct(ct):
    ct_temp = dict(ct)
    ct_temp['img'] = (ct_temp['img'] - np.median(ct_temp['img'])).\
           astype('float')
    return ct_temp        

def preprocess_Xray(Xray):
    Xray_temp = dict(Xray)
    Xray_temp['img'] = rvlib.windowImage(Xray_temp['img'], 60.0, 120.0)
    Xray_temp['img'] = Xray_temp['img'].astype('float') / \
             Xray_temp['img'].max() * 255.0
    return Xray_temp

def addHomogCoord(iPts):
    iPts = np.asarray(iPts)
    iPts = np.hstack((iPts, np.ones((iPts.shape[0],1))))
    return iPts


v = np.load('vretenca.npz')
# oblikuj v dict
Xray = {'img': v['xrayImg'], 'TPos': v['xrayTPos'], 'SPos': v['xraySPos']}
ct = {'img': v['ctVol'], 'TPos': v['ctTPos']} #3D polje; TPos je preslikava 4x4 matrika
ct2 = {'img': v['ct2Vol'], 'TPos': v['ct2TPos']} #2D polje; TPos v mm


#-----------------------------------------------------------------------------------------------------
def mapPointToPlane(iPos, Xray):
    #S + (p'-S)*alfa = p; pogoj: (p - p0) * n = 0 
    #--> (S + (p' - S) * alfa - p0) * n = 0
    #--> alfa = ((p0 - S) * n)/((p' - S) * n)
    #--> p = ((p0 - S) * n * (p' - S))/((p0 - S) * n) + S
    
    # tocke so v obliki Nx3/4
    iPos = np.asarray(iPos)
    if iPos.ndim == 1:
        iPos = np.reshape(iPos, (1,iPos.size))    
    iPos = iPos[:,:3]    
    # doloci izhodisce na detektorju
    p0 = np.dot(Xray['TPos'], np.array([0, 0, 0, 1]))
    # doloci normalo na ravnino
    ex = np.dot(Xray['TPos'], np.array([1, 0, 0, 1]))
    ey = np.dot(Xray['TPos'], np.array([0, 1, 0, 1]))
    n = np.cross(ex[:-1] - p0[:-1], ey[:-1] - p0[:-1])    
    #n = np.dot(Xray['TPos'], np.array([0, 0, 1, 0]))    
    n = n / np.sqrt(np.sum(n**2.0))  

    # s = rvlib.addHomo(Xray['SPos'].reshape((1,3)))
    # skrajsaj vse vektorje na dimenzije 1x3
    p0 = p0[:3]
    n = n[:3]
    s = Xray['SPos'].reshape((1,3))
    
    # koeficient za skaliranje smernega vektorja
    alpha = np.dot(p0 - s, n) / np.dot(iPos - s, n)

    # doloci polozaj tock na detektorju
    oPos = s + alpha.reshape((iPos.shape[0],1)) * (iPos - s)
    
    return oPos

#-------------------------------------------------------------------------------------
# Testiranje funkcije mapPointToPlane

# ustvari mrezo tock oglisc 3D slike
sz, sy, sx = ct2['img'].shape
g3x, g3y, g3z = np.meshgrid((0, sx-1), (0, sy-1), (0, sz-1), indexing='xy')
g3 = np.vstack((g3x.flatten(), g3y.flatten(), g3z.flatten())).transpose()
g3 = np.hstack((g3, np.ones((g3.shape[0], 1))))
g3p = np.dot(g3, ct['TPos'].transpose())    

# ustvari mrezo tock oglisc 2D slike
s2y,s2x = Xray['img'].shape
g2x,g2y = np.meshgrid((0, s2x-1),(0, s2y-1),indexing='xy')
# ustvari Mx4 matriko, zadnja koordinata homogena
g2 = np.vstack((g2x.flatten(), g2y.flatten())).transpose()
g2 = np.hstack((g2, np.zeros((g2.shape[0], 1)), np.ones((g2.shape[0], 1))))
g2p = np.dot(g2, Xray['TPos'].transpose())

# koordinata izvora zarkov
xsp = Xray['SPos'].flatten()    

# tocke 3D oglisc na ravnini detektorja
g2proj = mapPointToPlane(g3p[:,:3], Xray)

# narisi geometrijske razmere
s = 1
fig = pyplot.figure()    
ax = fig.add_subplot(111,projection='3d')
ax.scatter(xsp[0], xsp[1], xsp[2], c='m', marker='o')
ax.scatter(g2p[::s,0], g2p[::s,1], g2p[::s,2], c='b', marker='.', linewidths=0)
ax.scatter(g3p[::s,0], g3p[::s,1], g3p[::s,2], c='g', marker='.', linewidths=0)
ax.scatter(g2proj[:,0], g2proj[:,1], g2proj[:,2], c='r', marker='o')
pyplot.show()
# od zgoraj:
# ax.view_init(elev=80, azim=-125)
# iz zornega kota vira
# ax.view_init(elev=-9, azim=147)

#rdece tocke so tiste, ki lezijo na ravnini