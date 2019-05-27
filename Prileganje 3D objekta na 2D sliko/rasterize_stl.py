import os
import subprocess
import mrcfile
import numpy as np
from matplotlib import pyplot
from os.path import join

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
       
    # Display volume info and cross-sections
    print('Volume size: {}'.format(volume.shape))
    fig, ax = pyplot.subplots(1, 3, figsize=(5, 15))
    ax[0].imshow(np.squeeze(volume[:, :, volume.shape[-1]//2]), cmap='gray')
    ax[1].imshow(np.squeeze(volume[:, volume.shape[1]//2, :]), cmap='gray')
    ax[2].imshow(np.squeeze(volume[volume.shape[0]//2, :, :]), cmap='gray')
    pyplot.show()

