"""
Adapted from:
https://github.com/saopicc/DDFacet/raw/refs/heads/master/DDFacet/ToolsDir/ModToolBox.py

Returns
-------
_type_
    _description_
"""

import numpy as np

def GiveFFTFastSizes(Odd=True,NLim=100000):
    """
    Computes list of optimal FFT sizes. From http://www.fftw.org/doc/Real_002ddata-DFTs.html: 
      "FFTW is best at handling sizes of the form 2^a.3^b.5^c.7^d.11^e.13^f,
       where e+f is either 0 or 1, and the other exponents are arbitrary."
    Returns array of such integer numbers, up to NLim.
    If Odd=True, this does not include factors of 2.
    """
    sizes = np.array([1])
    for base, powers in [
             (2,[0] if Odd else range(1,20)),
             (3,range(15)), (5,range(15)), (7,range(15)) ]:
        sizes = (sizes[np.newaxis,:] * base**np.array(powers)[:,np.newaxis]).ravel()
    sizes = sizes[np.newaxis,:] * np.array([1,11,13])[:,np.newaxis]

    # no need to take set(), since sizes are unique by construction (from prime factors...)
    return np.array(sorted(sizes[(sizes<NLim)&(sizes>64)]))    
    # return np.array(sorted(set(sizes[(sizes<NLim)&(sizes>64)])))    

FFTOddSizes  = GiveFFTFastSizes(True,200000)
FFTEvenSizes = GiveFFTFastSizes(False,200000)

def GiveClosestFastSize(n,Odd=True):
    #ind=np.argmin(np.abs(n-FFTOddSizes))
    if Odd:
        ind=np.argmin(np.abs(n-FFTOddSizes))
        return FFTOddSizes[ind]
    else:
        ind=np.argmin(np.abs(n-FFTEvenSizes))
        return FFTEvenSizes[ind]

def EstimateNpix(Npix,
                 Padding=1.0,
                 min_size_fft=513):
    """ Picks image size from the list of fast FFT sizes.
        To avoid spectral leakage the number of taps in the FFT
        must not be too small.
    """
    Npix=int(round(Npix))
    Odd=True

    NpixOrig=Npix
    #if Npix%2!=0: Npix+=1
    #if Npix%2==0: Npix+=1
    Npix=GiveClosestFastSize(Npix,Odd=Odd)
    NpixOpt=Npix


    Npix *= Padding
    if Npix < min_size_fft:
        Npix = min_size_fft
    Npix=int(round(Npix))
    #if Npix%2!=0: Npix+=1
    #if Npix%2==0: Npix+=1
    Npix=GiveClosestFastSize(Npix,Odd=Odd)
    return NpixOpt,Npix