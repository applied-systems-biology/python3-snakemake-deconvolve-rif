import numpy as np
import skimage.external.tifffile as tifffile
from skimage import img_as_float32, img_as_float
from skimage import transform


def deconvolve(input_data_file, data_voxel_size, input_psf_file, psf_voxel_size, output_file):

    # Parameter for regularized inverse filter
    rif_lambda = 0.001

    psf = tifffile.imread(input_psf_file)[0,:,:,:]
    img = tifffile.imread(input_data_file)

    # Resize PSF to match the image voxel size
    psf_new_shape = np.array(img.shape) * (psf_voxel_size / data_voxel_size)
    psf_new_shape = tuple(int(x) for x in psf_new_shape)
    psf = transform.resize(psf, psf_new_shape)

    # Transform into Fourier space
    img = img_as_float(img)
    img_fft = np.fft.fftn(img)

    psf = img_as_float(psf)
    psf_fft = np.fft.fftn(psf, s=img_fft.shape)

    # Apply RIF
    # Adapted from DeconvolutionLab2 code
    # See https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RegularizedInverseFilter.java
    Y = img_fft
    H = psf_fft

    ## Generate laplacian
    l = np.ones((3, 3, 3))
    l[1, 1, 1] = -(3 * 3 * 3 - 1)
    l /= np.max(l)

    L = np.fft.fftn(l, img_fft.shape)
    H2 = H * H
    L2 = L * rif_lambda * L
    FA = H2 + L2
    FT = H / FA
    X = Y * FT
    deconv = np.fft.ifftn(X).real
    deconv -= np.min(deconv)
    deconv /= np.max(deconv)

    tifffile.imsave(output_file, img_as_float32(deconv))
