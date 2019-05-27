# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 08:47:50 2015

@author: Žiga Špiclin

RVLIB: knjižnica funkcij iz laboratorijskih vaj
       pri predmetu Robotski vid
"""
import numpy as np
import PIL.Image as im
import matplotlib.pyplot as plt
import matplotlib.cm as cm # uvozi barvne lestvice

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


def loadImageRaw(iPath, iSize, iFormat):
    '''
    Naloži sliko iz raw datoteke
    
    Parameters
    ----------
    iPath : str 
        Pot do datoteke
    iSize : tuple 
        Velikost slike
    iFormat : str
        Tip vhodnih podatkov
    
    Returns
    ---------
    oImage : numpy array
        Izhodna slika
    
    
    '''
    
    oImage = np.fromfile(iPath, dtype=iFormat) # nalozi raw datoteko
    oImage = np.reshape(oImage, iSize) # uredi v matriko
    
    return oImage


def showImage(iImage, iTitle=''):
    plt.figure() # odpri novo prikazno okno
    
    if iImage.ndim == 3 and iImage.shape[0] == 3:
        iImage = np.transpose(iImage,[1,2,0])

    plt.imshow(iImage, cmap = cm.Greys_r) # prikazi sliko v novem oknu
    plt.suptitle(iTitle) # nastavi naslov slike
    plt.xlabel('x')
    plt.ylabel('y')


def saveImageRaw(iImage, iPath, iFormat):
    '''
    Shrani sliko na disk
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika za shranjevanje
    iPath : str
        Pot in ime datoteke, v katero želimo sliko shraniti
    iFormat : str
        Tip podatkov v matriki slike
    
    Returns
    ---------
    Nothing
    '''
    iImage = iImage.astype(iFormat)
    iImage.tofile(iPath) # zapisi v datoteko


def loadImage(iPath):
    '''
    Naloži sliko v standardnih formatih (bmp, jpg, png, tif, gif, idr.)
    in jo vrni kot matriko
    
    Parameters
    ----------
    iPath - str
        Pot do slike skupaj z imenom
        
    Returns
    ----------
    oImage - numpy.ndarray
        Vrnjena matrična predstavitev slike
    '''
    oImage = np.array(im.open(iPath))
    if oImage.ndim == 3:
        oImage = np.transpose(oImage,[2,0,1])
    elif oImage.ndim == 2:
        oImage = np.transpose(oImage,[1,0])   
    return oImage


def saveImage(iPath, iImage, iFormat):
    '''
    Shrani sliko v standardnem formatu (bmp, jpg, png, tif, gif, idr.)
    
    Parameters
    ----------
    iPath : str
        Pot do slike z željenim imenom slike
    iImage : numpy.ndarray
        Matrična predstavitev slike
    iFormat : str
        Željena končnica za sliko (npr. 'bmp')
    
    Returns
    ---------
    Nothing

    '''
    if iImage.ndim == 3:
        iImage = np.transpose(iImage,[1,2,0])
    elif iImage.ndim ==2:
        iImage = np.transpose(iImage,[1,0])     
    img = im.fromarray(iImage) # ustvari slikovni objekt iz matrike
    img.save(iPath.split('.')[0] + '.' + iFormat)

    
def windowImage(iImage, iCenter, iWidth):
    """Lin. oknjenje: (Ls-1)/w*(I-(c-w/2)) ---> iSlopeA = (Ls-1)/w, iInterceptB = -(Ls-1)/w*(c-w/2)"""
    iImageType = iImage.dtype
    if iImageType.kind in ('u', 'i'):
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
        iRange = iMaxValue - iMinValue
    else:
        iMaxValue = np.max(iImage)
        iMinValue = np.min(iImage)
        iRange = iMaxValue - iMinValue
        
    iSlopeA = iRange / float(iWidth)
    iInterceptB = - iSlopeA * (float(iCenter) - iWidth / 2.0)
    
    return scaleImage(iImage, iSlopeA, iInterceptB)


def scaleImage(iImage, iSlopeA, iIntersectionB):
    """Linearna sivinska preslikava: a*I+b"""
    
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype = 'float')
    oImage = iSlopeA * iImage + iIntersectionB
    #zaokrozevanje vrednosti, pretvorba nazaj v originalna števila (8-bitno -> 0-255)
    
    if iImageType.kind in ('u', 'i'): # ali je uint ali int (prevelike in premajhne odrezati)
        oImage[oImage < np.iinfo(iImageType).min] = np. iinfo(iImageType).min #odrezemo premajhna stevila; np.iinfo(iImageType).min dobimo min vr., ki jo ta tip podpira
        oImage[oImage > np.iinfo(iImageType).max] = np. iinfo(iImageType).max #odrezemo prevelika stevila in jih postavimo na max mozne vr.;
    return np.array(oImage, dtype = iImageType)


def drawLine(iImage, iValue, x1, y1, x2, y2):
    ''' Narisi digitalno daljico v sliko

        Parameters
        ----------
        iImage : numpy.ndarray
            Vhodna slika
        iValue : tuple, int
            Vrednost za vrisavanje (barva daljice).
            Uporabi tuple treh elementov za barvno sliko in int za sivinsko sliko
        x1 : int
            Začetna x koordinata daljice
        y1 : int
            Začetna y koordinata daljice
        x2 : int
            Končna x koordinata daljice
        y2 : int
            Končna y koordinata daljice
    '''    
    
    oImage = iImage    
    
    if iImage.ndim == 3:
        assert type(iValue) == tuple, 'Za barvno sliko bi paramter iValue moral biti tuple treh elementov'
        for rgb in range(3):
            drawLine(iImage[rgb,:,:], iValue[rgb], x1, y1, x2, y2)
    
    elif iImage.ndim == 2:
        assert type(iValue) == int, 'Za sivinsko sliko bi paramter iValue moral biti int'
    
        dx = np.abs(x2 - x1)
        dy = np.abs(y2 - y1)
        if x1 < x2:
            sx = 1
        else:
            sx = -1
        if y1 < y2:
            sy = 1
        else:
            sy = -1
        napaka = dx - dy
     
        x = x1
        y = y1
        
        while True:
            oImage[y-1, x-1] = iValue
            if x == x2 and y == y2:
                break
            e2 = 2*napaka
            if e2 > -dy:
                napaka = napaka - dy
                x = x + sx
            if e2 < dx:
                napaka = napaka + dx
                y = y + sy
    
    return oImage
    
    
def colorToGray(iImage):
    '''
    Pretvori barvno sliko v sivinsko.
    
    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna barvna slika
        
    Returns
    -------
    oImage : numpy.ndarray
        Sivinska slika
    '''
    dtype = iImage.dtype
    r = iImage[0,:,:].astype('float')
    g = iImage[1,:,:].astype('float')
    b = iImage[2,:,:].astype('float')
    
    return (r*0.299 + g*0.587 + b*0.114).astype(dtype)
    
    
def computeHistogram(iImage, iNumBins, iRange=[], iDisplay=False, iTitle=''):
    '''
    Izracunaj histogram sivinske slike
    
    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna slika, katere histogram želimo izračunati

    iNumBins : int
        Število predalov histograma
        
    iRange : tuple, list
        Minimalna in maksimalna sivinska vrednost 

    iDisplay : bool
        Vklopi/izklopi prikaz histograma v novem oknu

    iTitle : str
        Naslov prikaznega okna
        
    Returns
    -------
    oHist : numpy.ndarray
        Histogram sivinske slike
    oEdges: numpy.ndarray
        Robovi predalov histograma
    '''    
    iImage = np.asarray(iImage)
    iRange = np.asarray(iRange)
    if iRange.size == 2:
        iMin, iMax = iRange
    else:
        iMin, iMax = np.min(iImage), np.max(iImage)
    oEdges = np.linspace(iMin, iMax+1, iNumBins+1)
    oHist = np.zeros([iNumBins,])
    for i in range(iNumBins):
        idx = np.where((iImage >= oEdges[i]) * (iImage < oEdges[i+1]))
        if idx[0].size > 0:
            oHist[i] = idx[0].size
    if iDisplay:
        plt.figure()
        plt.bar(oEdges[:-1], oHist)
        plt.suptitle(iTitle)

    return oHist, oEdges
    
    
def computeContrast(iImages):
    '''
    Izracunaj kontrast slik
    
    Parameters
    ---------
    iImages : list of numpy.ndarray
        Vhodne slike, na katerih želimo izračunati kontrast
        
    Returns : list
        Seznam kontrastov za vsako vhodno sliko
    '''
    oM = np.zeros((len(iImages),))
    for i in range(len(iImages)):
        fmin = np.percentile(iImages[i].flatten(),5)
        fmax = np.percentile(iImages[i].flatten(),95)
        oM[i] = (fmax - fmin)/(fmax + fmin)
    return oM
    
    
def computeEffDynRange(iImages):
    '''
    Izracunaj efektivno dinamicno obmocje
    
    Parameters
    ----------
    iImages : numpy.ndarray
        Vhodne slike
        
    Returns
    --------
    oEDR : float
        Vrednost efektivnega dinamicnega obmocja
    '''
    L = np.zeros((len(iImages,)))
    sig = np.zeros((len(iImages),))
    for i in range(len(iImages)):
        L[i] = np.mean(iImages[i].flatten())
        sig[i] = np.std(iImages[i].flatten())
    oEDR = np.log2((L.max() - L.min())/sig.mean())
    return oEDR
    

def computeSNR(iImage1, iImage2):
    '''
    Vrne razmerje signal/sum
    
    Paramters
    ---------
    iImage1, iImage2 : np.ndarray
        Sliki področij zanimanja, med katerima računamo SNR
        
    Returns
    ---------
    oSNR : float
        Vrednost razmerja signal/sum
    '''
    mu1 = np.mean(iImage1.flatten())
    mu2 = np.mean(iImage2.flatten())
    
    sig1 = np.std(iImage1.flatten())
    sig2 = np.std(iImage2.flatten())
    
    oSNR = np.abs(mu1 - mu2)/np.sqrt(sig1**2 + sig2**2)
            
    return oSNR



def alignICP(iPtsRef, iPtsMov, iEps=1e-6, iMaxIter=50, plotProgress=False):
    """Postopek iterativno najblizje tocke"""
    # YOUR CODE HERE
    curMat = []; oErr = []; iCurIter = 0
    if plotProgress:
        iPtsMov0 = np.matrix(iPtsMov)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    #zacni iterativni postopek
    while True:
        # poisci korespondencne pare tock
        iPtsRef_t, iPtsMov_t = findCorrespondingPoints(iPtsRef, iPtsMov)
        #doloci afino aproksimacijo preslikave
        oMat2D = mapAffineApprox2D(iPtsRef_t, iPtsMov_t)
        #posodobi premicne tocke
        iPtsMov = np.dot(addHomCoord2D(iPtsMov), oMat2D.transpose())
        #izracunaj napako
        curMat.append(oMat2D)
        oErr.append(np.sqrt(np.sum((iPtsRef_t[:,:2] - iPtsMov_t[:,:2])**2)))
        iCurIter = iCurIter + 1
        dMat = np.abs(oMat2D - transAffine2D())
        if iCurIter > iMaxIter or np.all(dMat < iEps):
            break
            
    #doloci kompozitum preslikav
    oMat2D = transAffine2D()
    for i in range(len(curMat)):
        
        """if plotProgress:
            iPtsMov_t = np.dot(addHomCoord2D(iPtsMov0), oMat2D.transpose())
            ax.clear()
            ax.plot(iPtsRef[:, 0], iPtsRef[:, 1], 'ob')
            ax.plot(iPtsMov_t[:, 0], iPtsMov_t[:, 1], 'om')
            fig.canvas.draw()
            #plt.pause(1)"""
            
        oMat2D = np.dot(curMat[i], oMat2D)
    
    return oMat2D, oErr

def gammaImage( iImage, iGamma ):
    """Gama preslikava: (Ls-1)(I/(Lr-1))^gama"""
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype = 'float')
    
    if iImageType.kind in ('u', 'i'):
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
        iRange = iMaxValue - iMinValue
    else:
        iMaxValue = np.max(iImage)
        iMinValue = np.min(iImage)
        iRange = iMaxValue - iMinValue
     
    #gama preslikava
    iImage = (iImage - iMinValue) / float(iRange) #skaliranje
    oImage = iImage ** float(iGamma) #potenciranje
    oImage = float(iRange) * oImage + iMinValue
    #zaokrozevanje vrednosti
    if iImageType.kind in ('u', 'i'): # ali je uint ali int (prevelike in premajhne odrezati)
        oImage[oImage < np.iinfo(iImageType).min] = np. iinfo(iImageType).min #odrezemo premajhna stevila; np.iinfo(iImageType).min dobimo min vr., ki jo ta tip podpira
        oImage[oImage > np.iinfo(iImageType).max] = np. iinfo(iImageType).max #odrezemo prevelika stevila in jih postavimo na max mozne vr.;
    #Vrni sliko v originalnem formatu
    return np.array(oImage, dtype = iImageType) 


def transAffine2D(iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0)):
    """Funkcija za poljubno 2D afino preslikavo"""
    # YOUR CODE HERE
    
    iRot = iRot * np.pi / 180
    oMatScale = np.array(((iScale[0], 0, 0), (0, iScale[1], 0), (0, 0, 1)))
    oMatTrans = np.array(((1, 0, iTrans[0]), (0, 1, iTrans[1]), (0, 0, 1)))
    oMatRot = np.array(((np.cos(iRot), -np.sin(iRot), 0), \
                        (np.sin(iRot), np.cos(iRot), 0), 
                    (0, 0, 1)))
    oMatShear = np.array(((1, iShear[0], 0), (iShear[1], 1, 0), (0, 0, 1)))
    #ustvari izhodno matriko
    oMat2D = np.dot(oMatTrans, np.dot(oMatShear, np.dot(oMatRot, oMatScale)))
    
    return oMat2D


def addHomCoord2D(iPts):
    if iPts.shape[-1] == 3:
        return iPts
    iPts = np.hstack((iPts, np.ones((iPts.shape[0], 1))))
    return iPts

def findCorrespondingPoints(iPtsRef, iPtsMov):
    """Poisci korespondence kot najblizje tocke"""
    # YOUR CODE HERE
    iPtsMov = np.array(iPtsMov)
    iPtsRef = np.array(iPtsRef)
    
    idxPair = -np.ones((iPtsRef.shape[0], 1), dtype ='int32')
    idxDist = np.ones((iPtsRef.shape[0], iPtsMov.shape[0]))
    for i in range(iPtsRef.shape[0]):
        for j in range(iPtsMov.shape[0]):
            idxDist[i,j] = np.sum((iPtsRef[i,:2] - iPtsMov[j,:2])**2)
        
        #doloci bijektivno preslikavo
        while not np.all(idxDist == np.inf):
            i, j = np.where(idxDist == np.min(idxDist))
            idxPair[i[0]] = j[0]
            idxDist[i[0], :] = np.inf
            idxDist[: ,j[0]] = np.inf
        #doloci pare
        idxValid, idxNotValid = np.where(idxPair >= 0)
        idxValid = np.array(idxValid)
        iPtsRef_t = iPtsRef[idxValid, :]
        iPtsMov_t = iPtsMov[idxPair[idxValid].flatten(), :]
    
    
    
    
    return iPtsRef_t, iPtsMov_t

def mapAffineApprox2D(iPtsRef, iPtsMov):
    """Afina aproksimacijska poravnava"""
    # YOUR CODE HERE
    iPtsRef = np.matrix(iPtsRef)
    iPtsMov = np.matrix(iPtsMov)
    #po potrebi dodaj homogeno koordinato
    iPtsRef = addHomCoord2D(iPtsRef)
    iPtsMov = addHomCoord2D(iPtsMov)
    #afina aproksimacija (s psefvdoinverzom)
    iPtsRef = iPtsRef.transpose()
    iPtsMov = iPtsMov.transpose()
    
    #psevdoinverz
    oMat2D = np.dot(iPtsRef, np.linalg.pinv(iPtsMov))
    
    #psevdoinverz na dolgo in siroko
    #oMat2D = iPtsRef * iPtsMov.transpose() * \
    #np.linalg.inv(iPtsMov * iPtsMov.transpose())
    
    return oMat2D
