import xml.etree.ElementTree as et

import os
import base64
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def smooth_and_scale(rts, intensities, scans_per_second, window_length, polyorder):

    scans_per_second = int(scans_per_second)
    window_length = int(window_length)
    polyorder = int(polyorder)
    
    rts = np.array(rts)
    intensities = np.array(intensities)

    interpolate_func = interp1d(rts,intensities,kind='linear')
    rt_range = np.linspace(0,9999,int(scans_per_second * 9999)).round(3)
    rts = rt_range[(rt_range >= rts.min()) & (rt_range <= rts.max())]
    intensities = savgol_filter(
        interpolate_func(rts), 
        window_length=window_length, 
        polyorder=polyorder, 
        mode='interp'
    )                

    return rts, intensities

def df_from_mzml(mzml_file:Path, scans_per_second:int=8, window_length:int=5, polyorder:int=3) -> pd.DataFrame:   

    namespace = '{http://psi.hupo.org/ms/mzml}'

    # prepare dicts for storing mzml data (in a matrix-like format)
    data_table = {}
    data_table['transition'] = []
    data_table['rt'] = []
    data_table['intensity'] = []

    try:

        tree = et.parse(mzml_file)
        xml_root = tree.getroot()

        for chromatogram in xml_root.iter("{}chromatogram".format(namespace)):

            pmz = False
            pcmz = False
            
            if "SRM" in f"{chromatogram.attrib['id']}":
                pcmz = f"{chromatogram.attrib['id']}".split("Q1=")[1].split(" ")[0]
                pmz = f"{chromatogram.attrib['id']}".split("Q3=")[1].split(" ")[0]

            polarity = False
            data = {}
            data[0] = []
            data[1] = []

            for node in chromatogram.iter():

                # find out if positive or negative
                if node.tag == '{}cvParam'.format(namespace):
                    if node.attrib["name"] == 'negative scan':                            
                        polarity = 'neg'
                    if node.attrib["name"] == 'positive scan':
                        polarity = 'pos'                      

                # get rt and ints                        
                if node.tag == '{}binaryDataArrayList'.format(namespace):                            
                    for binaryDataArray in node.iter('{}binaryDataArray'.format(namespace)):                                
                        data_type = False
                        data_values = []
                        for data_node in binaryDataArray.iter():
                            if data_node.tag == '{}cvParam'.format(namespace):                             
                                if data_node.attrib["name"] in ['time array', 'intensity array']:
                                    data_type = data_node.attrib["name"].replace("array", "").strip()
                            if data_node.tag == '{}binary'.format(namespace):
                                try:
                                    decoded_node_data = base64.b64decode(data_node.text)
                                    data_values = np.frombuffer(decoded_node_data, np.float64)                                        
                                except Exception as ex:
                                    print(ex)
                                    pass
                        
                        if data_type and len(data_values) > 0:
                            if data_type == 'time':
                                data[0] = data_values
                            if data_type == 'intensity':
                                data[1] = data_values
            
            if polarity and pcmz and pmz:

                # as np array
                rts = np.array([t*60 for t in data[0]])
                intensities = np.array(data[1])

                # scale and smooth
                rts, intensities = smooth_and_scale(
                    rts=rts,
                    intensities=intensities,
                    scans_per_second=scans_per_second,
                    window_length=window_length,
                    polyorder=polyorder
                )

                # make negative values 0
                intensities[intensities < 0] = 0
                    
                data_table['transition'].append(np.full(len(rts), f"{pcmz}_{pmz}_{polarity}"))
                data_table['rt'].append(rts)
                data_table['intensity'].append(intensities)                    

        # (re)shape
        data_table['transition'] = np.concatenate(tuple(data_table['transition']))
        data_table['rt'] = np.concatenate(tuple(data_table['rt']))
        data_table['intensity'] = np.concatenate(tuple(data_table['intensity']))

        # create dataframe
        data_table = pd.DataFrame(data_table)     
        data_table['transition'] = data_table['transition'].astype('category')
        data_table['rt'] = data_table['rt'].astype(float).round(decimals=5)
        data_table['intensity'] = data_table['intensity'].astype(int)

        # include file
        data_table['file'] = str(mzml_file)

        # include sample
        data_table['sample'] = data_table['file'].apply(lambda x: x.split(os.path.sep)[-1].replace(".mzML", ""))

    except Exception as ex:
        print(ex)

    return data_table
from scipy.ndimage import shift    

def phase_align(reference, target, roi=None, res=10):
    '''
    Cross-correlate data within region of interest at a precision of 1./res
    if data is cross-correlated at native resolution (i.e. res=1) this function
    can only achieve integer precision 

    Args:
        reference (1d array/list): signal that won't be shifted
        target (1d array/list): signal to be shifted to reference
        roi (tuple): region of interest to compute chi-squared
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    # convert to int to avoid indexing issues
    if roi==None: roi = [0,len(reference)-1]

    ROI = slice(int(roi[0]), int(roi[1]), 1)

    # interpolate data onto a higher resolution grid 
    x,r1 = highres(reference[ROI],kind='cubic',res=res)
    x,r2 = highres(target[ROI],kind='cubic',res=res)

    # subtract off mean 
    r1 -= r1.mean()
    r1 -= r2.mean()

    # compute the phase-only correlation function
    product = np.fft.fft(r1) * np.fft.fft(r2).conj()
    cc = np.fft.fftshift(np.fft.ifft(product))

    # manipulate the output from np.fft
    l = reference[ROI].shape[0]
    shifts = np.linspace(-0.5*l,0.5*l,l*res)

    # plt.plot(shifts,cc,'k-'); plt.show()
    return shifts[np.argmax(cc.real)]


def highres(y,kind='cubic',res=1):
    '''
    Interpolate data onto a higher resolution grid by a factor of *res*

    Args:
        y (1d array/list): signal to be interpolated
        kind (str): order of interpolation (see docs for scipy.interpolate.interp1d)
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    y = np.array(y)
    x = np.arange(0, y.shape[0])
    f = interp1d(x, y,kind='cubic')
    xnew = np.linspace(0, x.shape[0]-1, x.shape[0]*res)
    ynew = f(xnew)
    return xnew,ynew