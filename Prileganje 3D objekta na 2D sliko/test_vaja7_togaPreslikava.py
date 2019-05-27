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



def project3DTo2D(ct, Xray, iStep):
    '''
    Naloga 4: Funkcija za stozcasto projekcijo 3D slike v 2D ravnino

    Parameters
    ----------
    ct : dict
        Podatki o 3D ct sliki in sivinske vrednosti pod ključi TPos in img
    Xray : dict
        Podatki o 2D Xray sliki in sivinske vrednosti pod ključi TPos, SPos in img
    iStep : float
        Korak vzorčenja v mm vzdolž Xray žarkov od izvora do 2D detektorja
    
    Returns
    ---------
    oDRR : numpy.ndarray
        Izhodna iz 3D v 2D prostor preslikana slika, ki simulira rentgen
    oMask : numpy.ndarray
        Maska področja kamor se preslika 3D slika (1-ospredje, 0-ozadje)
    '''
    # ustvari mrezo tock oglisc 3D slike
    s3z, s3y, s3x = ct['img'].shape
    print(ct['img'].shape)
    g3x, g3y, g3z = np.meshgrid((0,s3x-1),(0,s3y-1),(0,s3z-1),indexing='xy')
    g3 = np.vstack((g3x.flatten(),g3y.flatten(),g3z.flatten())).transpose()
    g3 = np.hstack((g3, np.ones((g3.shape[0],1))))
    g3p = np.dot(g3, ct['TPos'].transpose())    
    
    # tocke 3D oglisc na ravnini detektorja
    g2proj = mapPointToPlane(g3p[:,:3], Xray)       
   
    # preslikaj v 2D ravnino
    g2plane = np.dot(addHomogCoord(g2proj), \
                np.linalg.inv(Xray['TPos']).transpose()) 
    
    # poisci najmanjsi ocrtan pravokotnik
    xmin = np.floor(np.min(g2plane[:,0]))
    xmax = np.ceil(np.max(g2plane[:,0]))
    ymin = np.floor(np.min(g2plane[:,1]))
    ymax = np.ceil(np.max(g2plane[:,1]))
    
    # preveri ali so tocke znotraj 2D slike
    s2y, s2x = Xray['img'].shape
    xmin = np.max((0,xmin)); xmin = int(np.min((s2x,xmin)))
    xmax = np.max((0,xmax)); xmax = int(np.min((s2x,xmax)))
    ymin = np.max((0,ymin)); ymin = int(np.min((s2y,ymin)))
    ymax = np.max((0,ymax)); ymax = int(np.min((s2y,ymax)))
    
    # definiraj mrezo tock v 2D ravnini
    g2x, g2y = np.meshgrid(range(xmin,xmax), range(ymin,ymax), indexing='xy')
    rectShape = g2x.shape
    
    # ustvari Mx4 matriko, zadnja koordinata homogena
    g2 = np.vstack((g2x.flatten(),g2y.flatten())).transpose()
    g2 = np.hstack((g2, np.zeros((g2.shape[0],1)), np.ones((g2.shape[0],1))))
    g2 = np.dot(g2, Xray['TPos'].transpose())
    
    # preberi pozicijo izvora zarkov
    xsp = Xray['SPos'].flatten().reshape((1,3))
    
    # doloci minimalno in maksimalno razdaljo za vzorcenje
#    d = np.sqrt(np.sum((g3p[:,:3]  - g2proj[:,:3])**2.0, axis=0))
    d = np.sqrt(np.sum((g3p[:,:3]  - xsp)**2.0, axis=1))
    dmin = np.min(d); dmax = np.max(d)
  
    # doloci vzornce tocke vzdolz zarkov    
    ds = np.arange(dmin,dmax,iStep)
    ds = np.reshape(ds, (1,ds.size))    

    # definiraj vzorcne tocke v 3d prostoru
    vs = g2[:,:3] - xsp
    vs = vs / np.sqrt(np.sum(vs**2.0, axis=1)).reshape((vs.shape[0],1))
    
    Nx1 = (vs.shape[0],1)
    px = xsp[0,0] + vs[:,0].reshape(Nx1) * ds
    py = xsp[0,1] + vs[:,1].reshape(Nx1) * ds
    pz = xsp[0,2] + vs[:,2].reshape(Nx1) * ds
    
    # preslikava koordinat v prostor 3D slike   
    Tmat = np.linalg.inv(ct['TPos'])
    pxn = Tmat[0,0]*px + Tmat[0,1]*py + Tmat[0,2]*pz + Tmat[0,3]
    pyn = Tmat[1,0]*px + Tmat[1,1]*py + Tmat[1,2]*pz + Tmat[1,3]
    pzn = Tmat[2,0]*px + Tmat[2,1]*py + Tmat[2,2]*pz + Tmat[2,3]

    # preveri katere koordinate so znotraj 3D slike
    idx = np.where((pxn>=0) & (pxn<s3x) & \
                    (pyn>=0) & (pyn<s3y) & \
                    (pzn>=0) & (pzn<s3z))  

    oRayInterp = np.zeros_like(pxn)    
    pxn = pxn[idx[0],idx[1]]
    pyn = pyn[idx[0],idx[1]]
    pzn = pzn[idx[0],idx[1]]

    # izvedi trilinearno interpolacijo      
    oRayInterp_i = interpn((np.arange(s3z),np.arange(s3y),np.arange(s3x)), \
                  ct['img'].astype('float'), \
                  np.dstack((pzn,pyn,pxn)), \
                  method='linear', bounds_error=False, fill_value=0) 
                      
    oRayInterp[idx[0],idx[1]] = oRayInterp_i               

    # izvedi dejansko operacijo vzdolz zarkov                  
    oRayInterp = np.mean(oRayInterp, axis=1).reshape(rectShape)
    
    # ustvari izhodne spremenljivke        
    oDRR = np.zeros_like(Xray['img']).astype('float')
    oMask = np.zeros_like(Xray['img'])

    oDRR[ymin:ymax, xmin:xmax] = oRayInterp
    oMask[ymin:ymax, xmin:xmax] = 255
    
    return oDRR, oMask

def fast_project3DTo2D(ct, Xray, iStep):
    # rendering for all 2d image pixels
    drr = vr.render(ct['TPos'])
    return drr, drr != 0

def rigidTransMatrix(ct, iPar): # iPar = [tx, ty, tz, alpha, beta, gamma]
    s3z,s3y,s3x = ct['img'].shape
    oRot = rvlib.transAffine3D(iTrans=(0,0,0), iRot=(iPar[3],iPar[4],iPar[5]))
    oTrans = rvlib.transAffine3D(iTrans=(iPar[0],iPar[1],iPar[2]), iRot=(0,0,0))
    oCenter = rvlib.transAffine3D(iTrans=(-s3x/2, -s3y/2, -s3z/2), iRot=(0,0,0))
    oInvCenter = rvlib.transAffine3D(iTrans=(s3x/2, s3y/2, s3z/2), iRot=(0,0,0))
    return np.dot(oTrans, np.dot(oInvCenter, np.dot(oRot, oCenter)))

def project3DTo2D(ct, Xray, iStep, iPar=[0,0,0,0,0,0]):
    TPos = np.dot(ct['TPos'], rigidTransMatrix(ct, iPar))    
    newCt = {'img':ct['img'], 'TPos':TPos}
    return rvlib.project3DTo2D(newCt, Xray, iStep)

#-------------------------------------------------------------------------------------
# test funkcije
# poklici funkcijo
iPar = [0, 0, 0, 10, 0, 0]
iStep = 1

# predobdelava 3D in 2D slik    
ct_temp = preprocess_ct(ct)
Xray_temp = preprocess_Xray(Xray)


oDRR, oDRRMask = project3DTo2D(ct_temp, Xray_temp, iStep, iPar)    


# šest osi v enem prikaznem oknu
f, ax = pyplot.subplots(1, 3, figsize=(9,3))
ax[0].imshow(oDRR, cmap='gray')
ax[0].set_title('DRR')
ax[1].imshow(Xray_temp['img'], cmap='gray')
ax[1].set_title('Xray')      
ax[2].imshow(2*rvlib.normalizeImage(oDRR, iType='range') + 
  rvlib.normalizeImage(Xray_temp['img'], iType='range'), cmap='gray')
ax[2].set_title('Superpozicija')
pyplot.show()