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
import cv2 as cv
# nalozi knjiznico za morfoloske operacije 
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage.measure import label


def preprocess_ct(ct):
    ct_temp = dict(ct)
    ct_temp['img'] = (ct_temp['img'] - np.median(ct_temp['img'])).\
          astype('float')
    ct_temp['spac'] = [1,1,1]
    return ct_temp        

def preprocess_Xray(Xray):
    Xray_temp = dict(Xray)
    Xray_temp['img'] = rvlib.windowImage(Xray_temp['img'], 100.0, 180.0)
    Xray_temp['img'] = Xray_temp['img'].astype('float') / \
             Xray_temp['img'].max() * 255.0
    Xray_temp['spac'] = [1,1]
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


def rigidTransMatrix(ct, iPar): # iPar = [tx, ty, tz, alpha, beta, gamma]
    s3z,s3y,s3x = ct['img'].shape
    oRot = rvlib.transAffine3D(iTrans=(0,0,0), iRot=(iPar[3],iPar[4],iPar[5]))
    oTrans = rvlib.transAffine3D(iTrans=(iPar[0],iPar[1],iPar[2]), iRot=(0,0,0))
    oCenter = rvlib.transAffine3D(iTrans=(-s3x/2, -s3y/2, -s3z/2), iRot=(0,0,0))
    oInvCenter = rvlib.transAffine3D(iTrans=(s3x/2, s3y/2, s3z/2), iRot=(0,0,0))
    return np.dot(oTrans, np.dot(oInvCenter, np.dot(oRot, oCenter)))



def fast_project3DTo2D(ct, Xray, iStep):
    # rendering for all 2d image pixels
    drr = vr.render(ct['TPos'])
    return drr, drr != 0

def project3DTo2D(ct, Xray, iStep, iPar=[0,0,0,0,0,0]):
    TPos = np.dot(ct['TPos'], rigidTransMatrix(ct, iPar))    
    newCt = {'img':ct['img'], 'TPos':TPos}
    #return rvlib.project3DTo2D(newCt, Xray, iStep)
    return fast_project3DTo2D(newCt, Xray, iStep)


def correlationCoefficient(iImageI, iImageJ):
    '''
    Izracunaj korelacijski koeficient med 2D slikama
    
    Parameters
    ----------
    iImageI : numpy.ndarray
        Sivinska informacija slike I
    iImageJ : numpy.ndarray
        Sivinska informacija slike J
    
    Returns
    ---------
    oCC : float
    '''
    iImageI = np.asarray(iImageI)
    iImageJ = np.asarray(iImageJ)
    
    return np.mean(
        ((iImageI - iImageI.mean()) * (iImageJ - iImageJ.mean())) / iImageJ.std() / iImageI.std())


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
    oMI : float   
    '''
    iImageI = np.asarray(iImageI)
    iImageJ = np.asarray(iImageJ)
    
    # funkcija za pretvorbo sivinskih vrednosti v indekse
    def getIndices(iData, iBins, iSpan):
        # doloci obmocje sivinskih vrednosti  
        minVal, maxVal = iSpan
        maxVal = (np.max(iData)+1e-6 if maxVal is None else maxVal)
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


def criterionFcn(iPar, ct, Xray, cf_type='mi'):
    iStep = 1
    iBins = 8
    iSpan = [0.0, 255.0]
#     imin = -0.80841250322681901
#     imax = 13.620590454314147

    oDRR, oDRRMask = project3DTo2D(ct, Xray, iStep, iPar)

    if cf_type == 'mi':
        ids = np.where(oDRRMask>0)
        xmin, xmax = np.min(ids[1]), np.max(ids[1])
        ymin, ymax = np.min(ids[0]), np.max(ids[0])
        iImageI = Xray['img'][ymin:ymax,xmin:xmax]
        iImageJ = (oDRR[ymin:ymax,xmin:xmax] - imin) / (imax - imin) * 255.0
        return -mutualInformation(iImageI, iImageJ, iBins, iSpan)
    elif cf_type == 'cc':
        iImageI = Xray['img']
        iImageJ = oDRR        
        return correlationCoefficient(iImageI, iImageJ)


#-------------------------------------------------------------------------------------
#nalaganje podatkov
POLYMENDER_EXE = r'./PolyMender_1_7_1_exe_64/PolyMender.exe'
SOF2MRC_EXE = r'./mrc/sof2mrc.exe'


def convert_stl_to_sof(input_file, output_file, tree_depth=8, sampling_mm=0.9):
    cmd = [
        POLYMENDER_EXE, 
        input_file, 
        str(tree_depth), 
        str(sampling_mm), 
        output_file
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        cwd=os.path.split(POLYMENDER_EXE)[0]
    )
    stdout, stderr = process.communicate()
    print ("cat returned code = %d" % process.returncode)
    print ("cat output:\n\n%s\n\n" % stdout)
    print ("cat errors:\n\n%s\n\n" % stderr)

    
def convert_sof_to_mrc(input_file, output_file, smoothing_kernel=10):
    cmd = [
        SOF2MRC_EXE, 
        input_file, 
        output_file,
        str(smoothing_kernel)
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        cwd=os.path.split(SOF2MRC_EXE)[0]
    )
    stdout, stderr = process.communicate()
    print ("cat returned code = %d" % process.returncode)
    print ("cat output:\n\n%s\n\n" % stdout)
    print ("cat errors:\n\n%s\n\n" % stderr)    


if __name__ == "__main__":
    PATH = os.getcwd()    
    
    # Convert from STL to SOF
    convert_stl_to_sof(join(PATH, 'ponvica.stl'), join(PATH, 'ponvica.sof'))
    
    # Convert from SOF to MRC
    convert_sof_to_mrc(join(PATH, 'ponvica.sof'), join(PATH, 'ponvica.mrc'))
    
    # Open the MRC file and correct header
    with mrcfile.open('ponvica.mrc', mode='r+', permissive=True) as mrc:
        mrc.header.map = mrcfile.constants.MAP_ID
    # Reopen the MRC file and read volume info
    with mrcfile.open('ponvica.mrc', permissive=True) as mrc:
        volume = mrc.data
        print(volume)

def convertToGray(image):
    dtype = image.dtype
    rgb = np.array(image, dtype = 'float')
    gray = rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.114
    return gray.astype(dtype)

slika = Image.open("RTG3.png")
arr_slika = np.array(slika)
arr_slika_g = convertToGray(arr_slika)
arr_slika = arr_slika_g[190:430,1140:1370]
iThr = int(np.average(arr_slika))
imgT = 255 * (arr_slika < iThr).astype('uint8')
arr_slika = np.array(imgT)
#fig, ax = pyplot.subplots(1, 1, figsize=(5, 15))
#ax.imshow((arr_slika), cmap='gray')
#pyplot.show()



# Create the basic black image 
mask1 = np.ones(shape = arr_slika.shape, dtype = "uint8")
mask2 = np.zeros(shape = arr_slika.shape, dtype = "uint8")
imgG_shape = arr_slika.shape
imgG_shape_x = imgG_shape[0]
imgG_shape_y = imgG_shape[1]
# Draw a white, filled rectangle on the mask image
cv.circle(mask1, (int(imgG_shape_x/2),int(imgG_shape_x/2)), int(125), (0,0,0), -1)
cv.rectangle(img = mask2, pt1 = (int(imgG_shape_y*0.65), int(imgG_shape_x)), pt2 = (int(imgG_shape_y), int(imgG_shape_x*0.65)), color = (255,255,255), thickness = -1)
#showImage(mask1)
#showImage(mask2)
mask = mask1-mask2
#opening
#arr_slika = opening(arr_slika, disk(8))
arr_slika = arr_slika-mask
#showImage(imgT, 'obdelana')
arr_slika = cv.medianBlur(arr_slika, 9) 
#showImage(imgT, 'BLUR')
prag1 = 200
prag2 = 50
arr_slika = cv.Canny(arr_slika, prag1, prag2,True)



# oblikuj v dict
Xray = {'img': arr_slika, 'TPos': v['xrayTPos'], 'SPos': v['xraySPos']}
ct = {'img': volume, 'TPos': v['ctTPos']}
#spreminjanje pozicije 3D modela
ct['TPos'][0,-1] = 550 #povečava
ct['TPos'][1,-1] = -265  #-gor/+dol
ct['TPos'][2,-1] = -208 #+desno/-levo
#ct['TPos'] =  np.matrix([[-0.74661,0.1559,0.64673,-290.01552246],[-0.27278,-0.72442,-0.08387,-90.00116692785],[0.60676,-0.23903,0.75809,-39.99004559293],[0,0,0,1]])
#spreminjanje pozicije izvora
Xray['SPos'][0,0] = -500
Xray['SPos'][0,1] = 0
Xray['SPos'][0,2] = 0
#spreminjanje pozicije slike
Xray['TPos'] = np.matrix([[0,0,-1,700],[0,1,0,-250],[1,0,0,-200],[0,0,0,1]])




# predobdelava 3D in 2D slik    
ct_temp = preprocess_ct(ct)

Xray_temp = preprocess_Xray(Xray)    

vr = VolumeRenderer(
    vol=ct_temp, 
    img=Xray_temp,
    ray_step_mm=1, 
    render_op='maxip'
)

#-------------------------------------------------------------------------------------
# test funkcije
cf_type = 'cc'  # mi, cc
opt_type = 'Nelder-Mead'  # Powell, Nelder-Mead
disp_iter = True



# Optimizacija enega parametra toge preslikave: alpha
#iParStart = -15
#def get_full_par(iPar):
    #return [0, 0, 0, iPar, 0, 0]

# # Optimizacija vseh 6 parametrov toge preslikave: tx, ty, tz, alpha, beta, gamma
#iParStart = [-5, 0, 5, -5, 5, 0]
iParStart = [0, 0, 0, 197.500603, 142.644328, -20.0699134] 
def get_full_par(iPar):
    return [iPar[0], iPar[1], iPar[2], iPar[3], iPar[4], iPar[5]]

# definicija funkcije MP(p)
oCF = lambda iPar : criterionFcn(get_full_par(iPar), ct_temp, Xray_temp, cf_type=cf_type)

neval = 1
def callbackF(iPar):
    global neval
    fval = oCF(iPar)
    print('{0:4d}   {1: 3.6f}'.format(neval, fval))
    neval += 1        
if disp_iter:
    print('{0:4s}   {1:9s}'.format('Iter', ' CF(p)'))

# klic optimizacije

# method = Nelder-Mead (simpleksna optimizacija), Powell
res = minimize(fun=oCF, x0=iParStart, method=opt_type, tol=1e-6, 
                options={'maxiter':1000, 'maxfev':5000, 'xtol':1e-6, 
                'ftol':1e-6, 'disp': disp_iter}, callback=(callbackF if disp_iter else None))
iParEnd = res.x
  
print('iPar pred poravnavo: {}\niPar po poravnavi: {}'.format(
      iParStart, iParEnd))

iParStart = get_full_par(iParStart)
iParEnd = get_full_par(iParEnd)

# prikaz rezultatov
oDRR_start, oDRRMask = project3DTo2D(ct_temp, Xray_temp, 3, iParStart)
oDRR_end, oDRRMask = project3DTo2D(ct_temp, Xray_temp, 3, iParEnd)
oDRR_optim, oDRRMask = project3DTo2D(ct_temp, Xray_temp, 3, np.zeros((6,1)))

# Prikaz
f1, ax1 = pyplot.subplots(1, 2, figsize=(9, 3))
ax1[0].imshow(oDRR_start, cmap='gray')
ax1[0].set_title('DRR pred poravnavo')

ax1[1].imshow(oDRR_end, cmap='gray')
ax1[1].set_title('DRR po poravnavi')

f2, ax2 = pyplot.subplots(1, 2, figsize=(9, 3))
ax2[0].imshow(2*rvlib.normalizeImage(oDRR_start, iType='range') + 
  rvlib.normalizeImage(Xray_temp['img'], iType='range'), cmap='gray')
ax2[0].set_title('Superpozicija pred poravnavo')

ax2[1].imshow(2*rvlib.normalizeImage(oDRR_end, iType='range') + 
  rvlib.normalizeImage(Xray_temp['img'], iType='range'), cmap='gray')
ax2[1].set_title('Superpozicija po poravnavi')

pyplot.show()


#------------------------------------------------------------------------------------------------

print(oDRR_end.shape)
oDRR_end = np.array(oDRR_end)

oDRR_endRGB = cv.cvtColor(oDRR_end, cv.COLOR_BGR2RGB)
print(oDRR_endRGB)

oDRR_endG = cv.cvtColor(oDRR_endRGB, cv.COLOR_BGR2GRAY)
print(oDRR_endG)

oDRR_endG8=np.uint8(oDRR_endG*255)

edge = cv.Canny(oDRR_endG8, prag1, prag2) #TULE MI NIKAKOR NOČE DELAT CANNY! -.- ZGORNI DVE VRSTICI STA BILI ZA TEST


corners = cv.cornerHarris(edge, 5, 3, 1/16)
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

oLocalMax = findLocalMax(corners, corners.max() * 0.01) 

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


"ORIGINAL!!!!!!!!!!!!!!"
#Iskanje tretje točke na elipsi
tretja_tocka = 0 #inicializacija indeksa tretje točke
d1 = 0.4
d2 = 0.7
#print(edge.shape[1]*d1,edge.shape[1]*d2,edge.shape[0]*d1,edge.shape[0])


tretja_tocka = []
#IF stavki določajo okno, v katerem poiščemo točko na elipsi
for n in range(oLocalMax_shape):
    if oLocalMax[n,0] > edge.shape[1]*d1:#400
        if oLocalMax[n,0] < edge.shape[1]:#700
            if oLocalMax[n,1] > edge.shape[0]*d1:#400
                if oLocalMax[n,1] < edge.shape[0]:#700
                    tretja_tocka.append(n)

                    
#Zapis točke, ki je bila najdena
x3 = oLocalMax[tretja_tocka[-1],0]
y3 = oLocalMax[tretja_tocka[-1],1]
print("Tretja točka: ",x3,",",y3)

#Prikaz slike in točk
pyplot.imshow(oDRR_endG8, cmap='gray') #Tu sem spremenila vhod

pyplot.plot(x1,y1, 'o', markersize=10.0)
pyplot.plot(x2,y2, 'o', markersize=10.0)
pyplot.plot(x3,y3, 'o', markersize=10.0) 
pyplot.show()       


#izračun prvega kota
kot1 = np.arctan((x1 - x2)/(y1 - y2)) 
kot1 = np.degrees(kot1)
kot1 = 90 - np.abs(kot1)
print("Izračunani prvi kot je: ",round(kot1,3),"°")
print("")

"ORIGINAL!!!!!!!!!!!!!!"

#naklon prve daljice
k1 = (y2 - y1)/(x2 - x1)
n1 = int(y1 - k1*x1)

#naklon druge daljice, ki je pravokotna na prvo
k2 = -1/k1
k2 = 2-k2

#izračun koeficienta
n2 = y3 - k2*x3

#izračun točke K (4. točke)
x4 = (n2 - n1)/(k1 - k2)
y4 = x4*k2 + n2
x4 = int(x4)
y4 = int(y4)

#izračun razdalij med točkami
PK = np.sqrt((x3-x4)**2 + (y3-y4)**2)
AK = np.sqrt((x1-x4)**2 + (y1-y4)**2)
BK = np.sqrt((x2-x4)**2 + (y2-y4)**2)   
    
    
#izračun drugega kota
kot2 = np.arcsin(PK/(np.sqrt(AK*BK)))
kot2 = np.degrees(kot2)
kot2 = np.abs(kot2)
if kot2 > 16:
    kot2 = kot2 + kot2*0.25

print("Izračunani drugi kot je: ",round(kot2,3),"°")