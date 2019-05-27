import numpy as np
from scipy.interpolate import interpn
import scipy.ndimage as ni 

def normalizeImage(iImage, iType='whitening'):
    '''
    Normalizacija vrednosti
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iType : str
        Nacin normalizacije slike (whitening, range)
        
    Returns
    --------
    oImage : numpy.ndarray
        Normalizirana slika
    '''
    if iType=='whitening':
        oImage = (iImage - np.mean(iImage)) / np.std(iImage)
    elif iType=='range':
        oImage = (iImage - np.min(iImage)) / (np.max(iImage) - np.min(iImage))
    return oImage


def scaleImage(iImage, iSlopeA, iIntersectionB):
    '''
    Linearna sivinska preslikava y = a*x + b
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iSlopeA : float
        Linearni koeficient (a) v sivinski preslikavi
        
    iIntersectionB : float
        Konstantna vrednost (b) v sivinski preslikavi
        
    Returns
    --------
    oImage : numpy.ndarray
        Linearno preslikava sivinska slika
    '''    
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype='float')
    oImage = iSlopeA * iImage + iIntersectionB
    # zaokrozevanje vrednosti
    if iImageType.kind in ('u','i'):
        oImage[oImage<np.iinfo(iImageType).min] = np.iinfo(iImageType).min
        oImage[oImage>np.iinfo(iImageType).max] = np.iinfo(iImageType).max
    return np.array(oImage, dtype=iImageType)
    
    
def windowImage(iImage, iCenter, iWidth):
    '''
    Linearno oknjenje y = (Ls-1)/w*(x-(c-w/2)
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iCenter : float
        Sivinska vrednost, ki določa položaj centra okna
        
    iWidth : float
        Širina okna, ki določa razpon linearno preslikavnih vrednosti
        
    Returns
    --------
    oImage : numpy.ndarray
        Oknjena sivinska slika
    '''     
    iImageType = iImage.dtype
    if iImageType.kind in ('u','i'):
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
        iRange = iMaxValue - iMinValue
    else:
        iMaxValue = np.max(iImage)
        iMinValue = np.max(iImage)
        iRange = iMaxValue - iMinValue
    
    iSlopeA = iRange / float(iWidth)
    iInterceptB = - iSlopeA * (float(iCenter) - iWidth / 2.0)
    
    return scaleImage(iImage, iSlopeA, iInterceptB)


def addHomogCoord(iPts):
    '''
    Seznamu 2D koordinat dodaj homogeno koordinato

    Parameters
    ----------
    iPts : numpy.ndarray
        Polje Nx3 koordinat x, y, z
        
    Returns
    --------
    oPts : numpy.ndarray
        Polje Nx4 homogenih koordinat x, y, z

    '''
    iPts = np.asarray(iPts)
    iPts = np.hstack((iPts, np.ones((iPts.shape[0],1))))
    return iPts


def mapPointToPlane(iPos, Xray):
    '''
    Naloga 3: Funkcija za preslikavo tocke na ravnino detektorja
    
    Parameters
    ----------
    iPos : numpy.ndarray
        X,Y,Z koordinate 3D točk v obliki Nx3/4 (nehomogena ali homogena oblika)
    Xray : dict
        Podatki o 2D Xray sliki in sivinske vrednosti pod ključi TPos, SPos in img
    
    Returns
    ---------
    oPos : numpy.ndarray
        X,Y,Z koordinate v 2D ravnino detektorja preslikanihg 3D točk
    '''
    # iPos = np.asarray(iPos[:,:3])
    iPos = np.asarray(iPos)
    if iPos.ndim == 1:
        iPos = np.reshape(iPos, (1,iPos.size))    
    iPos = iPos[:,:3]    
    # doloci izhodisce na detektorji
    p0 = np.dot(Xray['TPos'], np.array([0, 0, 0, 1]))
    # doloci normalo na ravnino
    ex = np.dot(Xray['TPos'], np.array([1, 0, 0, 1]))
    ey = np.dot(Xray['TPos'], np.array([0, 1, 0, 1]))
    n = np.cross(ex[:-1]-p0[:-1], ey[:-1]-p0[:-1])    
#    n = np.dot(Xray['TPos'], np.array([0, 0, 1, 0]))    
    n = n / np.sqrt(np.sum(n**2.0))  

    # skrajsaj vse vektorje na dimenzije 1x3
    p0 = p0[:3]; n = n[:3]
    s = Xray['SPos'].reshape((1,3))
    
    # koeficient za skaliranje smernega vektorja
    alpha = np.dot(p0 - s, n) / np.dot(iPos - s, n)

    # doloci polozaj tock na detektorju
    oPos = s + alpha.reshape((iPos.shape[0],1)) * (iPos - s)
    
    return oPos

    
def transAffine2D(iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0)):
    '''
    Ustvari poljubno 2D afino preslikavo v obliki 3x3 homogene matrike

    Parameters
    ----------
    iScale : tuple, list
        Skaliranje vzdolž x in y (kx,ky)

    iTrans : tuple, list
        Translacija vzdolž x in y (tx,ty)

    iRot : float
        Kot rotacije (alfa)

    iShear : tuple, list
        Strig vzdolž x in y (gxy,gyx)
        
    Returns
    --------
    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika
    '''
    iRot *= np.pi/180.0
    oMatScale = np.array(((iScale[0], 0 ,           0),
                          (0,         iScale[1],    0),
                          (0,         0 ,           1)))
    oMatTrans = np.array(((1,         0 ,    iTrans[0]),
                          (0,         1,     iTrans[1]),
                          (0,         0 ,           1)))
    oMatShear = np.array(((1,         iShear[0] ,   0),
                          (iShear[1], 1,            0),
                          (0,         0 ,           1)))
    oMatRot = np.array(((np.cos(iRot),-np.sin(iRot),0),
                          (np.sin(iRot),np.cos(iRot), 0),
                          (0,           0 ,           1)))
    oMat2D= np.dot(oMatTrans, np.dot(oMatShear, np.dot(oMatRot, oMatScale)))
    return oMat2D
    
    
def transAffine3D(iScale=(1, 1, 1), iTrans=(0, 0, 0), iRot=(0,0,0), iShear=(0, 0, 0)):
    '''
    Ustvari poljubno 3D afino preslikavo v obliki 4x4 homogene matrike

    Parameters
    ----------
    iScale : tuple, list
        Skaliranje vzdolž x, y in z (kx,ky,kz)

    iTrans : tuple, list
        Translacija vzdolž x, y in z (tx,ty,tz)

    iRot : float
        Koti rotacije okoli x, y in z (alfa, beta, gama)

    iShear : tuple, list
        Strig vzdolž x, y in z (gxy, gxz, gyz)
        
    Returns
    --------
    oMat3D : numpy.ndarray
        Homogena 4x4 transformacijska matrika
    '''    
    iRot = np.array(iRot)*np.pi/180.0
    oMatScale = np.array(((iScale[0], 0, 0, 0),
                          (0, iScale[1], 0, 0),
                          (0, 0, iScale[2], 0),
                          (0, 0, 0, 1)))
    oMatTrans = np.array(((1, 0, 0, iTrans[0]),
                          (0, 1, 0, iTrans[1]),
                          (0, 0, 1, iTrans[2]),
                          (0, 0, 0, 1)))
    oMatShear = np.array(((1, iShear[0], iShear[1], 0),
                          (iShear[0], 1, iShear[2], 0),
                          (iShear[1], iShear[2], 1, 0),
                          (0, 0, 0, 1)))
    oMatRotX = np.array(((1, 0, 0, 0),
                         (0, np.cos(iRot[0]), -np.sin(iRot[0]), 0),
                         (0, np.sin(iRot[0]), np.cos(iRot[0]), 0),
                         (0, 0, 0, 1)))
    oMatRotY = np.array(((np.cos(iRot[1]), 0, np.sin(iRot[1]), 0,),
                         (0, 1, 0, 0),
                         (-np.sin(iRot[1]), 0, np.cos(iRot[1]), 0),
                         (0, 0, 0, 1)))
    oMatRotZ = np.array(((np.cos(iRot[2]), -np.sin(iRot[2]), 0, 0),
                         (np.sin(iRot[2]), np.cos(iRot[2]), 0, 0),
                         (0, 0, 1, 0),
                         (0, 0, 0, 1)))
    oMatRot = np.dot(oMatRotX, np.dot(oMatRotY, oMatRotZ))
    oMat3D= np.dot(oMatTrans, np.dot(oMatShear, np.dot(oMatRot, oMatScale)))
    return oMat3D


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


def mutualInformation(iImageI, iImageJ, iBins, iSpan=(None, None)):
    '''
    Izracunaj medsebojno informacijo med 2D slikama
    
    Parameters
    ----------
    iImageI : numpy.ndarray
        Sivinska informacija slike I
    iImageJ : numpy.ndarray
        Sivinska informacija slike J
    iBins : int
        Stevilo predalov v histogramu
    iSpan : tuple | list
        Obmocje vrednosti (min, max)
    
    Returns
    ---------
    oDRR : numpy.ndarray
        Izhodna iz 3D v 2D prostor preslikana slika, ki simulira rentgen
    oMask : numpy.ndarray
        Maska področja kamor se preslika 3D slika (1-ospredje, 0-ozadje)    
    '''
    iImageI = np.asarray(iImageI)
    iImageJ = np.asarray(iImageJ)
    
    # funkcija za pretvorbo sivinskih vrednosti v indekse
    def getIndices(iData, iBins, iSpan):
        # doloci obmocje sivinskih vrednosti  
        minVal, maxVal = iSpan
        maxVal = (np.max(iData)+1e-7 if maxVal is None else maxVal)
        minVal = (np.min(iData) if minVal is None else minVal)
        # pretvori v indeks polja
        idx = np.round((iData - minVal) / (maxVal - minVal) * (iBins-1))
        idx[idx < 0] = 0
        idx[idx >= iBins] = iBins-1
        # vrni indekse
        return idx.astype('uint32')
        
    # funkcija za izracun 1D histograma 
    def hist1D(iData, iBins, iSpan):
        # pretvorba sivinskih vrednosti v indekse
        idx = getIndices(iData, iBins, iSpan)
        # izracunaj histogram
        histData = np.zeros((iBins,))
        for i in idx:
            histData[i] += 1
        # vrni histogram
        # return histData
        # vrni glajeni histogram
        return ni.convolve(histData, np.array([0.2, 0.6, 0.2]))
        
    # funkcija za izracun 2D histograma 
    def hist2D(iData1, iData2, iBins, iSpan):
        # pretvorba sivinskih vrednosti v indekse
        idx1 = getIndices(iData1, iBins, iSpan)
        idx2 = getIndices(iData2, iBins, iSpan)        
        # izracunaj histogram
        histData = np.zeros((iBins, iBins))    
        for (i, j) in zip(idx1, idx2):
            histData[i,j] += 1
        # vrni histogram
        # return histData
        # vrni glajeni histogram
        return ni.convolve(histData, np.array([
                [1, 2, 1], [2, 8, 2], [1, 2, 1]])/20.0)    
        
    # izracunaj histograme slik
    hI = hist1D(iImageI, iBins, iSpan)
    hJ = hist1D(iImageJ, iBins, iSpan)
    hIJ = hist2D(iImageI, iImageJ, iBins, iSpan)

    # normaliziraj histograme v gostote verjetnosti    
    pI = hI / (np.sum(hI) + 1e-7)
    pJ = hJ / (np.sum(hJ) + 1e-7)
    pIJ = hIJ / (np.sum(hIJ) + 1e-7)
    
    # izracunaj entropije    
    HI = np.sum(- pI[pI>0] * np.log(pI[pI>0]))
    HJ = np.sum(- pJ[pJ>0] * np.log(pJ[pJ>0]))
    HIJ = np.sum(- pIJ[pIJ>0] * np.log(pIJ[pIJ>0]))
    
    # izracunaj medsebojno informacijo
    oMI = HI + HJ - HIJ
    
    return oMI