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
   # print ("cat returned code = %d" % process.returncode)
   # print ("cat output:\n\n%s\n\n" % stdout)
   # print ("cat errors:\n\n%s\n\n" % stderr)

    
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
   # print ("cat returned code = %d" % process.returncode)
   # print ("cat output:\n\n%s\n\n" % stdout)
   # print ("cat errors:\n\n%s\n\n" % stderr)    


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
        #print(volume)  

    """   
    # Display volume info and cross-sections
    print('Volume size: {}'.format(volume.shape))
    fig, ax = pyplot.subplots(1, 3, figsize=(5, 15))
    ax[0].imshow(np.squeeze(volume[:, :, volume.shape[-1]//2]), cmap='gray')
    ax[1].imshow(np.squeeze(volume[:, volume.shape[1]//2, :]), cmap='gray')
    ax[2].imshow(np.squeeze(volume[volume.shape[0]//2, :, :]), cmap='gray')
    #pyplot.show()
    """

v = np.load('vretenca.npz')

def convertToGray(image):
    dtype = image.dtype
    rgb = np.array(image, dtype = 'float')
    gray = rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.114
    return gray.astype(dtype)

slika = Image.open("RTG3.png")
arr_slika = np.array(slika)
arr_slika = convertToGray(arr_slika)
arr_slika = arr_slika[190:430,1140:1370]
iThr = int(np.average(arr_slika))
imgT = 255 * (arr_slika < iThr).astype('uint8')
arr_slika = np.array(imgT)

# oblikuj v dict
Xray = {'img': arr_slika, 'TPos': v['xrayTPos'], 'SPos': v['xraySPos']}
ct = {'img': volume, 'TPos': v['ctTPos']}
#spreminjanje pozicije 3D modela
ct['TPos'][0,-1] = 500 #povečava
ct['TPos'][1,-1] = -255  #-gor/+dol
ct['TPos'][2,-1] = -208 #+desno/-levo
#ct['TPos'] =  np.matrix([[-0.74661,0.1559,0.64673,-290.01552246],[-0.27278,-0.72442,-0.08387,-90.00116692785],[0.60676,-0.23903,0.75809,-39.99004559293],[0,0,0,1]])
#spreminjanje pozicije izvora
Xray['SPos'][0,0] = -500
Xray['SPos'][0,1] = 0
Xray['SPos'][0,2] = 0
#spreminjanje pozicije slike
Xray['TPos'] = np.matrix([[0,0,-1,700],[0,1,0,-250],[1,0,0,-200],[0,0,0,1]])



#-------------------------------------------------------------------------------------
# test funkcije
# poklici funkcijo
iPar = [0, 0, 0, 197.500603, 142.644328, -20.0699134]
iStep = 5

# predobdelava 3D in 2D slik    
ct_temp = preprocess_ct(ct)
Xray_temp = preprocess_Xray(Xray)

vr = VolumeRenderer(
    vol=ct_temp, 
    img=Xray_temp,
    ray_step_mm=1, 
    render_op='maxip'
)

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