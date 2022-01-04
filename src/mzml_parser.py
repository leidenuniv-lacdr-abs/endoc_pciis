from typing import Tuple
from nptyping import NDArray

import xml.etree.ElementTree as et

import os
import base64
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def smooth_and_scale(rts:NDArray, intensities:NDArray, scans_per_second:int=8, window_length:int=5, polyorder:int=3) -> Tuple[NDArray,NDArray]:
    """[summary]

    Args:
        rts (NDArray): Retention times
        intensities (NDArray): Intensities
        scans_per_second (int, optional): Number of scans per second to up/down scale to. Defaults to 8.
        window_length (int, optional): Window length for scipy.signal.savgol_filter function. Defaults to 5.
        polyorder (int, optional): Polyorder for scipy.signal.savgol_filter function. Defaults to 3.

    Returns:
        Tuple[NDArray,NDArray]: Both the new retention times and intensities are returned.
    """

    # prepare interpolate function based on input data
    interpolate_func = interp1d(rts,intensities,kind='linear')
    
    # determine new rt range
    rt_range = np.linspace(0,9999,int(scans_per_second * 9999)).round(3)
    rts = rt_range[(rt_range >= rts.min()) & (rt_range <= rts.max())]
    
    # apply up/down scaling and include smooting
    intensities = savgol_filter(
        interpolate_func(rts), 
        window_length=window_length, 
        polyorder=polyorder, 
        mode='interp'
    )                

    # return the new rt and intensities arrays as Tuple
    return rts, intensities

def df_from_mzml(mzml_file:Path, scans_per_second:int=8, window_length:int=5, polyorder:int=3) -> pd.DataFrame:
    """Converts and mzML file to a Pandas DataFrame

    Args:
        mzml_file (Path): location of mzML file to convert
        scans_per_second (int, optional): Number of scans per second to up/down scale to. Defaults to 8.
        window_length (int, optional): Window length for scipy.signal.savgol_filter function. Defaults to 5.
        polyorder (int, optional): Polyorder for scipy.signal.savgol_filter function. Defaults to 3.

    Returns:
        pd.DataFrame: A data frame with transition, rt, intensity, file, and a sample column
    """

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