import os
import time
from collections import namedtuple

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.tools as tools


import numpy as np
import SimpleITK as sitk
import six

def preprocess_ct(ct):
    """
    Preprocess input image by subtracting the median value and changing type to float.

    :param ct: Input image.
    :type ct: dict
    :return: Preprocessed input image.
    :rtype: dict
    """
    ct_temp = dict(ct)
    ct_temp['img'] = (ct_temp['img'] - np.median(ct_temp['img'])). \
        astype('float')
    return ct_temp


def preprocess_xray(xray):
    """
    Preprocess input image by changing type to float and scaling to range [0.0, 255.0].

    :param xray: Input image.
    :type xray: dict
    :return: Preprocessed input image.
    :rtype: dict
    """
    xray_temp = dict(xray)
    xray_temp['img'] = xray_temp['img'].astype('float') / \
                       xray_temp['img'].max() * 255.0
    return xray_temp


def mat_affine_3d(scale=(1, 1, 1), trans=(0, 0, 0), rot=(0, 0, 0), shear=(0, 0, 0)):
    """
    Create 3D affine mapping in form of 4x4 homogeneous matrix.

    :param scale: Scaling along x, y in z (kx,ky,kz).
    :param trans: Translation along x, y in z (tx,ty,tz).
    :param rot: Rotation about x, y in z (alfa, beta, gama).
    :param shear: Shear along x, y in z (gxy, gxz, gyz).
    :type scale: tuple[float]
    :type trans: tuple[float]
    :type rot: tuple[float]
    :type shear: tuple[float]
    :return: Transformation in form of 4x4 homogeneous matrix.
    :rtype: np.ndarray
    """
    rot = np.array(rot) * np.pi / 180.0
    mat_scale = np.array(((scale[0], 0, 0, 0),
                          (0, scale[1], 0, 0),
                          (0, 0, scale[2], 0),
                          (0, 0, 0, 1)))
    mat_trans = np.array(((1, 0, 0, trans[0]),
                          (0, 1, 0, trans[1]),
                          (0, 0, 1, trans[2]),
                          (0, 0, 0, 1)))
    mat_shear = np.array(((1, shear[0], shear[1], 0),
                          (shear[0], 1, shear[2], 0),
                          (shear[1], shear[2], 1, 0),
                          (0, 0, 0, 1)))
    mat_rot_x = np.array(((1, 0, 0, 0),
                          (0, np.cos(rot[0]), -np.sin(rot[0]), 0),
                          (0, np.sin(rot[0]), np.cos(rot[0]), 0),
                          (0, 0, 0, 1)))
    mat_rot_y = np.array(((np.cos(rot[1]), 0, np.sin(rot[1]), 0,),
                          (0, 1, 0, 0),
                          (-np.sin(rot[1]), 0, np.cos(rot[1]), 0),
                          (0, 0, 0, 1)))
    mat_rot_z = np.array(((np.cos(rot[2]), -np.sin(rot[2]), 0, 0),
                          (np.sin(rot[2]), np.cos(rot[2]), 0, 0),
                          (0, 0, 1, 0),
                          (0, 0, 0, 1)))
    mat_rot = np.dot(mat_rot_x, np.dot(mat_rot_y, mat_rot_z))
    return np.dot(mat_trans, np.dot(mat_shear, np.dot(mat_rot, mat_scale)))


def rigid_volume_trans(volume, rigid_body_par):
    """
    Rigid pose transformation of input volume in world-coordinate space.

    :param volume: Volume dict.
    :param rigid_body_par: Rigid-body parameters [tx, ty, tz, alpha, beta, gamma].
    :type volume: dict
    :type rigid_body_par: tuple
    :return: Transformation in form of 4x4 homogeneous matrix.
    :rtype: np.ndarray
    """
    s3z, s3y, s3x = volume['img'].shape
    oRot = mat_affine_3d(rot=rigid_body_par[3:6])
    oTrans = mat_affine_3d(trans=rigid_body_par[0:3])
    oCenter = mat_affine_3d(trans=(-s3x / 2, -s3y / 2, -s3z / 2))
    return np.dot(volume['TPos'], np.dot(oTrans, np.dot(np.linalg.inv(oCenter), np.dot(oRot, oCenter))))


class RenderParams(object):
    Attributes = namedtuple('Attributes',
                            ['sizes2d', 'steps2d', 'sizes3d', 'steps3d', 'boxmin', 'boxmax', 'ray_org', 'ray_step',
                             'trans_2d'])

    attr_info = Attributes(
        sizes2d=np.ndarray((2,), dtype='uint32'),
        steps2d=np.ndarray((2,), dtype='float32'),
        sizes3d=np.ndarray((4,), dtype='uint32'),  # actual size 3
        steps3d=np.ndarray((4,), dtype='float32'),  # actual size 3
        boxmin=np.ndarray((4,), dtype='float32'),  # actual size 3
        boxmax=np.ndarray((4,), dtype='float32'),  # actual size 3
        ray_org=np.ndarray((4,), dtype='float32'),  # actual size 3
        ray_step=np.ndarray((2,), dtype='float32'),  # actual size 1
        trans_2d=np.ndarray((16,), dtype='float32')
    )

    attr_in_size = Attributes(
        sizes2d=2,
        steps2d=2,
        sizes3d=3,
        steps3d=3,
        boxmin=3,
        boxmax=3,
        ray_org=3,
        ray_step=1,
        trans_2d=16
    )

    attr_mem_size = Attributes(
        *(attr.size * attr[0].nbytes for attr in attr_info)
    )

    mem_size = sum(attr_mem_size)

    _params = None
    _struct_arr_ptr = None

    @property
    def values(self):
        if self._params is not None:
            return self._params
        else:
            return None

    def __init__(self, params, struct_arr_ptr):
        self._params = params
        self._struct_arr_ptr = struct_arr_ptr

        # check input sizes
        for (name, in_size) in six.iteritems(self.attr_in_size._asdict()):
            if np.asarray(params._asdict()[name]).size != in_size:
                raise ValueError('Parameter "{}" should have {} elements, but has {}.'.format(
                    name, in_size, params._asdict()[name].size)
                )
        # force type conversion and 64-bit alignment
        params_typed = self.Attributes(
            *(np.resize(np.array(p, dtype=r.dtype), (r.size,)) for (p, r) in zip(params, self.attr_info))
        )
        # copy from host to allocated structure array on device
        offset_bytes = 0
        for (p, p_mem_size) in zip(params_typed, self.attr_mem_size):
            cuda.memcpy_htod(int(struct_arr_ptr) + offset_bytes, memoryview(p))
            offset_bytes += p_mem_size

    def __str__(self):
        if self._struct_arr_ptr is None:
            return ''
        values_to_print = []

        # copy from allocated structure array on device to host
        offset_bytes = 0
        for (name, value) in six.iteritems(self.attr_info._asdict()):
            mem_size = value.size * value[0].nbytes
            result_out_bytes = bytearray(mem_size)
            cuda.memcpy_dtoh(result_out_bytes, int(self._struct_arr_ptr) + offset_bytes)
            out_value = np.frombuffer(result_out_bytes, dtype=value.dtype, count=value.size)
            offset_bytes += mem_size

            values_to_print.append(
                '{key}={value}'.format(
                    key=name,
                    value=out_value
                )
            )
        return '\n'.join(values_to_print)


class VolumeRenderer(object):
    VALID_RENDER_OPERATION = ['maxip', 'minip', 'drr']
    BLOCK_SIZE_1D = 1024
    BLOCK_SIZE_2D = 32

    def __init__(self, vol, img, ray_step_mm=None, render_op='drr'):
        source = """
        #include "cuda_math.h"

        //------------------------------ DATA STRUCTURES -------------------------------	
        struct sRenderParams
        {
            // 2D detector data
            uint2 sizes2D;
            float2 steps2D;
            
            // 3D image data
            uint3 sizes3D; float1 __padding1;
            float3 steps3D; float1 __padding2;
            float3 boxmin; float1 __padding3;
            float3 boxmax; float1 __padding4;
            
            // Source position
            float3 ray_org; float1 __padding5;
        
            // Step along rays
            float1 ray_step, __padding6;
            
            // Transformation from 2D image to 2D plane in WCS 
            float T2D[16];
        };
            
        //-------------------------------- DEVICE CODE ---------------------------------	
        // Device variables
        extern "C" {
        texture<float, cudaTextureType3D, cudaReadModeElementType> d_tex;
        }
            
        // Intersect ray with a 3D volume:
        // see https://wiki.aalto.fi/download/attachments/40023967/ 
        // gpgpu.pdf?version=1&modificationDate=1265652539000
        __device__
        int intersectBox(
            float3 ray_org, float3 raydir, 
            sRenderParams *d_params, 
            float *tnear, float *tfar )
        {							    
            // Compute intersection of ray with all six bbox planes
            float3 invR = make_float3(1.0f) / raydir;
            float3 tbot = invR * (d_params->boxmin - ray_org);
            float3 ttop = invR * (d_params->boxmax - ray_org);	
        
            // Re-order intersections to find smallest and largest on each axis
            float3 tmin = fminf(ttop, tbot);
            float3 tmax = fmaxf(ttop, tbot);
        
            // Find the largest tmin and the smallest tmax
            float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
            float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
        
            *tnear = largest_tmin;
            *tfar = smallest_tmax;
        
            return smallest_tmax > largest_tmin;	
        }
        
        // Define DRR operator
        struct drr_operator
        {        
            static __inline__ __host__ __device__ 
            void compute ( 
                float &in, float &acc )
            {        
                acc += in;
            }
        };
        
        // Define MIP operator
        struct mip_operator
        {        
            static __inline__ __host__ __device__ 
            void compute ( 
                float &in, float &acc )
            {        	
                if(in > acc)
                    acc = in;
            }
        };
        
        // Define MINIP operator
        struct minip_operator
        {        
            static __inline__ __host__ __device__ 
            void compute ( 
                float &in, float &acc )
            {        	
                if(in < acc)
                    acc = in;
            }
        };
        
        // Homogeneous transformation: 
        // multiplication of a point w homog. transf. matrix
        /*static __inline__ __host__ __device__ 
        float3 hom_trans(float*& Tx, float3& pos)
        {
            float xw = Tx[0]*pos.x + Tx[4]*pos.y +  Tx[8]*pos.z + Tx[12];
            float yw = Tx[1]*pos.x + Tx[5]*pos.y +  Tx[9]*pos.z + Tx[13];
            float zw = Tx[2]*pos.x + Tx[6]*pos.y + Tx[10]*pos.z + Tx[14];
            
            return make_float3( xw, yw, zw );
        }*/
        
        static __inline__ __host__ __device__ 
        float3 hom_trans(float*& Tx, float3& pos)
        {
            float xw = Tx[0]*pos.x + Tx[1]*pos.y +  Tx[2]*pos.z + Tx[3];
            float yw = Tx[4]*pos.x + Tx[5]*pos.y +  Tx[6]*pos.z + Tx[7];
            float zw = Tx[8]*pos.x + Tx[9]*pos.y + Tx[10]*pos.z + Tx[11];
            
            return make_float3( xw, yw, zw );
        }
        
        // Rendering kernel: 
        // traverses the volume and performs linear interpolation
        extern "C" {
        __global__ 
        void render_kernel( 
            float* d_image, 
            float* d_Tx, float* d_TxT2D, 
            sRenderParams *d_params )	
        { 
            // Resolve 2D image index
            float x = blockIdx.x*blockDim.x + threadIdx.x;
            float y = blockIdx.y*blockDim.y + threadIdx.y;
            
            if ( (uint(x) >= d_params->sizes2D.x) || 
                    (uint(y) >= d_params->sizes2D.y) ) 
                return;		
            
            float3 ray_org, pos2D;
            
            // Transform source position to volume space
            ray_org = hom_trans( d_Tx, d_params->ray_org );
            
            // Create a point in 2D detector space
            pos2D = make_float3( x*d_params->steps2D.x, y*d_params->steps2D.y, 0.0f );
            
            // Inline homogeneous transformation to volume space
            // ie., (x,y) pixel in 3D volume coordinate system
            pos2D = hom_trans( d_TxT2D, pos2D );
                
            // Find eye ray in world space that points from the X-ray source 
            // to the current pixel on the detector plane:
            // - ray origin is in the X-ray source (xs,ys,zs)
            // - unit vector points to the point in detector plane (xw-xs,yw-ys,zw-zs)		
            float3 ray_dir = normalize( pos2D - ray_org ); 
                    
            // Find intersection with 3D volume
            float tnear, tfar;
            if ( ! intersectBox(ray_org, ray_dir, d_params, &tnear, &tfar) )
                return;
            
            // March along ray from front to back		
            float dt = d_params->ray_step.x;
                    
            float3 pos = make_float3(
                (ray_org.x + ray_dir.x*tnear) / d_params->steps3D.x, 
                (ray_org.y + ray_dir.y*tnear) / d_params->steps3D.y, 
                (ray_org.z + ray_dir.z*tnear) / d_params->steps3D.z);
        
            float3 step = make_float3(
                ray_dir.x * dt / d_params->steps3D.x, 
                ray_dir.y * dt / d_params->steps3D.y, 
                ray_dir.z * dt / d_params->steps3D.z);
                    
            #ifdef RENDER_MINIP
            float acc = 1e+7;
            #else
            float acc = 0;
            #endif
            for( ; tnear<=tfar; tnear+=dt )
            {		
                // resample the volume
                float sample = tex3D( d_tex, pos.x+0.5f, pos.y+0.5f, pos.z+0.5f );
                
                #ifdef RENDER_MAXIP
                mip_operator::compute( sample, acc );
                #elif RENDER_MINIP
                minip_operator::compute( sample, acc );
                #elif RENDER_DRR
                drr_operator::compute( sample, acc );
                #endif   
        
                // update position
                pos += step;
            }
        
            // Write to the output buffer
            uint idx = uint(x) + uint(y) * d_params->sizes2D.x;
            d_image[idx] = acc;	
        }
        }
        
        // Rendering kernel: 
        // traverses the volume and performs linear interpolation
        // for selected points in the 2d image
        extern "C" {
        __global__ 
        void render_kernel_idx( 
            float* d_image, uint* d_idx, uint max_idx,
            float* d_Tx, float* d_TxT2D, 
            sRenderParams *d_params )	
        { 
            // Resolve 1D index
            uint idx = blockIdx.x*blockDim.x + threadIdx.x;
                
            if ( idx > max_idx )				
                return;
                
            uint idx_t = d_idx[ idx ];
            
            // Resolve 2D image index
            uint y = idx_t / d_params->sizes2D.x;
            uint x = idx_t - y * d_params->sizes2D.x;
            
            //if ( (uint(x) >= d_params->sizes2D.x) || 
            //		(uint(y) >= d_params->sizes2D.y) ) 
            //	return;		
            
            float3 ray_org, pos2D;
            
            // Transform souce position to volume space
            ray_org = hom_trans( d_Tx, d_params->ray_org );
            
            // Create a point in 2D detector space
            pos2D = make_float3(float(x)*d_params->steps2D.x, 
                float(y)*d_params->steps2D.y, 0.0f);
            
            // Inline homogeneous transformation to volume space
            // ie., (x,y) pixel in 3D volume coordinate system
            pos2D = hom_trans( d_TxT2D, pos2D );
                
            // Find eye ray in world space that points from the X-ray source 
            // to the current pixel on the detector plane:
            // - ray origin is in the X-ray source (xs,ys,zs)
            // - unit vector points to the point in detector plane (xw-xs,yw-ys,zw-zs)		
            float3 ray_dir = normalize( pos2D - ray_org ); 
                    
            // Find intersection with 3D volume
            float tnear, tfar;
            if ( ! intersectBox(ray_org, ray_dir, d_params, &tnear, &tfar) )
                return;
            
            // March along ray from front to back		
            float dt = d_params->ray_step.x;
                    
            float3 pos = make_float3(
                (ray_org.x + ray_dir.x*tnear)/d_params->steps3D.x, 
                (ray_org.y + ray_dir.y*tnear)/d_params->steps3D.y, 
                (ray_org.z + ray_dir.z*tnear)/d_params->steps3D.z);
        
            float3 step = make_float3(
                ray_dir.x*dt/d_params->steps3D.x, 
                ray_dir.y*dt/d_params->steps3D.y, 
                ray_dir.z*dt/d_params->steps3D.z);
                    
            #ifdef RENDER_MINIP
            float acc = 1e+7;
            #else
            float acc = 0;
            #endif
            for( ; tnear<=tfar; tnear+=dt )
            {		
                // resample the volume
                float sample = tex3D(d_tex, pos.x+0.5f, pos.y+0.5f, pos.z+0.5f);
                
                #ifdef RENDER_MAXIP
                mip_operator::compute( sample, acc );
                #elif RENDER_MINIP
                minip_operator::compute( sample, acc );
                #elif RENDER_DRR
                drr_operator::compute( sample, acc );
                #endif                              
        
                // update position
                pos += step;
            }
        
            // Write to the output buffer
            d_image[idx] = acc;
        }
        }        
        """
        if render_op not in self.VALID_RENDER_OPERATION:
            raise ValueError('Rendering operation "{}" is not valid.'.format(render_op))
        cmodule = pycuda.compiler.SourceModule(
            source,
            options=['-DRENDER_{}'.format(render_op.upper())],
            include_dirs=["C:\\Users\\Ana\\Documents\\ROBOTSKI VID SEMINAR\\include"],          
            no_extern_c=True
        )        
        # include_dirs=[os.path.join(os.getcwd(), 'include')],

        self._texture = cmodule.get_texref('d_tex')
        self._renderer = cmodule.get_function('render_kernel')
        self._renderer_idx = cmodule.get_function('render_kernel_idx')

        if ray_step_mm is None:
            ray_step_mm = float(np.linalg.norm(vol['spac']) / 2.0)

        self._params_d = cuda.mem_alloc(RenderParams.mem_size)
        self.params = RenderParams(
            RenderParams.Attributes(
                sizes2d=img['img'].shape[::-1],
                steps2d=img['spac'],
                sizes3d=vol['img'].shape[::-1],
                steps3d=vol['spac'],
                boxmin=np.array((0, 0, 0), dtype='float32').flatten(),
                boxmax=(np.array(vol['img'].shape[::-1]) - 1.0) * np.array(vol['spac']).astype('float32').flatten(),
                ray_org=img['SPos'].flatten(),
                ray_step=ray_step_mm,
                trans_2d=np.array(img['TPos'], dtype='float32').flatten()
            ),
            self._params_d
        )

        # Copy array to texture memory
        self._texture.set_array(
            cuda.np_to_array(vol['img'].astype('float32'), order='C')
        )
        # We could set the next if we wanted to address the image
        # in normalized coordinates ( 0 <= coordinate < 1.)
        # self._texture.set_flags(cuda.TRSF_READ_AS_INTEGER)
        # self._texture.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        # Perform linear interpolation
        self._texture.set_filter_mode(cuda.filter_mode.LINEAR)
        self._texture.set_address_mode(0, cuda.address_mode.CLAMP)
        self._texture.set_address_mode(1, cuda.address_mode.CLAMP)
        self._texture.set_address_mode(2, cuda.address_mode.CLAMP)

        # check max threads per block for current GPU
        max_threads_per_block = tools.DeviceData().max_threads

        if self.BLOCK_SIZE_1D is None:
            self.BLOCK_SIZE_1D = max_threads_per_block
        elif self.BLOCK_SIZE_1D > max_threads_per_block:
            raise ValueError('Parameter BLOCK_SIZE_1D={} exceeds maximal pool of threads per block '
                             '(current GPU has maximum of {} threads per block).'.format(
                self.BLOCK_SIZE_1D, max_threads_per_block)
            )

        if self.BLOCK_SIZE_2D is None:
            self.BLOCK_SIZE_2D = 2
            while self.BLOCK_SIZE_2D**2 < max_threads_per_block:
                self.BLOCK_SIZE_2D *= 2
        elif self.BLOCK_SIZE_2D**2 > max_threads_per_block:
            raise ValueError('Parameter BLOCK_SIZE_2D={} (squared=) exceeds maximal pool of threads per block '
                             '(current GPU has maximum of {} threads per block).'.format(
                self.BLOCK_SIZE_2D, self.BLOCK_SIZE_2D**2, max_threads_per_block)
            )

        # threads per block
        nx, ny = self.params.values.sizes2d
        self._blocksize_2d = (
            nx if nx < self.BLOCK_SIZE_2D else self.BLOCK_SIZE_2D,
            ny if ny < self.BLOCK_SIZE_2D else self.BLOCK_SIZE_2D,
            1
        )
        # blocks per grid
        self._gridsize_2d = (
            int(nx / self._blocksize_2d[0]),
            int(ny / self._blocksize_2d[1]),
            1
        )

    def render(self, t_3d_to_wcs, idx=None):
        t_3d_to_wcs = np.asarray(t_3d_to_wcs).astype('float32')
        if t_3d_to_wcs.shape != (4, 4):
            raise ValueError('Input should be a 4x4 homogeneous matrix!')

        t_wcs_to_3d = np.linalg.inv(t_3d_to_wcs)
        t_2d_to_wcs = np.reshape(self.params.values.trans_2d, (4, 4))
        t_x_t_2d = np.dot(t_wcs_to_3d, t_2d_to_wcs)

        img_out = np.ndarray(self.params.values.sizes2d[::-1], dtype='float32')
        img_out[:, :] = 0.0

        if idx is None:
            self._renderer(
                cuda.Out(img_out),
                cuda.In(t_wcs_to_3d),
                cuda.In(t_x_t_2d),
                self._params_d,
                texrefs=[self._texture],
                block=self._blocksize_2d,
                grid=self._gridsize_2d
            )
        else:
            idx = np.asarray(idx).astype('uint32')

            # threads per block
            nx = idx.size
            blocksize_1d = (nx if nx < self.BLOCK_SIZE_1D else self.BLOCK_SIZE_1D, 1, 1)
            # blocks per grid
            gridsize_1d = (int(nx / blocksize_1d[0]), 1, 1)

            self._renderer_idx(
                cuda.Out(img_out),
                cuda.In(idx),
                np.uint32(idx.size),
                cuda.In(t_wcs_to_3d),
                cuda.In(t_x_t_2d),
                self._params_d,
                texrefs=[self._texture],
                block=blocksize_1d,
                grid=gridsize_1d
            )

        return img_out


if __name__ == '__main__':
    os.environ["PATH"] += os.pathsep + '/usr/local/cuda/bin'
    os.environ["PATH"] += "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin"
    os.environ["PATH"] += "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\libnvvp"
    os.environ["PATH"] += "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\lib\\x64"
    os.environ["PATH"] += "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\extras\\CUPTI\\libx64"
    os.environ["PATH"] += "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\VC\\bin"
    os.environ["PATH"] += "C:\\Users\\Ana\\Documents\\ROBOTSKI VID SEMINAR\\include"
    os.environ["CUDA_PATH"] += "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0"

    print('IMPORTANT NOTE:\n'
          'Before using this code you should ensure that project is deployed to path'
          ' "/tmp/drr-cuda-project/" and that environment variable "PATH" is updated'
          ' by adding "/usr/local/cuda/bin".\n\nIf this is not set, the cuda code will'
          ' return compilation errors.\n\nAnd always remember to have fun!')

    #----------------------------------------------------------------------------------
    # Load the data
    ct = sitk.ReadImage("ct.nrrd")
    xray = sitk.ReadImage("xray.nrrd")

    xrayTPos = np.array([
        [-0.2925, -0.0510, -0.9549, 397.8680],
        [-0.9542, 0.0809, 0.2879, 192.7720],
        [0.0625, 0.9954, -0.0723, -107.8180],
        [0, 0, 0, 1]
    ]).astype(np.float32)

    ctTPos = np.array([
        [1, 0, 0, 19.961],
        [0, 1, 0, 23.7891],
        [0, 0, 1, 164.0],
        [0, 0, 0, 1]
    ]).astype(np.float32)

    xraySPos = np.array(
        [-648.471, 285.4830, 117.6120]
    ).astype(np.float32)

    xray = {'img': sitk.GetArrayFromImage(xray), 'TPos': xrayTPos, 'SPos': xraySPos, 'spac': xray.GetSpacing()[::-1]}
    ct = {'img': sitk.GetArrayFromImage(ct), 'TPos': ctTPos, 'spac': ct.GetSpacing()[::-1]}

    xray = preprocess_xray(xray)
    ct = preprocess_ct(ct)

    rigid_body_params = [0, 0, 0, 0, 0, 0]  # iPar = [0, 0, 0, 0, 0, 0] is the goldstandard position
    ct['TPos'] = rigid_volume_trans(ct, rigid_body_params)

    #----------------------------------------------------------------------------------
    # Demo usage
    vr = VolumeRenderer(vol=ct, img=xray, ray_step_mm=1, render_op='drr')

    # rendering for all 2d image pixels
    drr = vr.render(ct['TPos'])

    # rendering for the indexed 2d image pixels
    drr_idx = vr.render(ct['TPos'], idx=range(xray['img'].size))

    if os.path.exists('drr.nrrd'):
        os.remove('drr.nrrd')
    sitk.WriteImage(sitk.GetImageFromArray(drr), 'drr.nrrd', True)

    if os.path.exists('drr_idx.nrrd'):
        os.remove('drr_idx.nrrd')
    sitk.WriteImage(sitk.GetImageFromArray(drr_idx), 'drr_idx.nrrd', True)

    #----------------------------------------------------------------------------------
    # Perform timing tests
    def perform_timing_test(vol, img, n_repetitions, ray_step_mm, render_op):
        print('-' * 80)
        print('Starting timing test with {} repeats.'.format(n_repetitions))
        rigid_body_params = np.random.randint(low=-100, high=100, size=(n_repetitions, 6))

        vr = VolumeRenderer(vol, img, ray_step_mm, render_op)

        start = time.time()

        for i in range(n_repetitions):
            _ = vr.render(rigid_volume_trans(vol, rigid_body_params[i, :]))

        end = time.time()

        time_diff = end - start
        print('Test finished in {:.4f} seconds.'.format(time_diff))
        print('Each projection took {:.4f} miliseconds.'.format(1000.0 * time_diff / n_repetitions))


    print('-' * 80)
    print('Test settings: \n\tray_step_mm=1mm\n\trender_op=drr')
    perform_timing_test(ct, xray, 1000, 1, 'drr')

    print('-' * 80)
    print('Test settings: \n\tray_step_mm=5mm\n\trender_op=drr')
    perform_timing_test(ct, xray, 1000, 5, 'drr')

    print('-' * 80)
    print('Test settings: \n\tray_step_mm=1mm\n\trender_op=maxip')
    perform_timing_test(ct, xray, 1000, 1, 'maxip')
