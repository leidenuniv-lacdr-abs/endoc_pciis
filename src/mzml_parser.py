from typing import Tuple
from nptyping import NDArray
from typing import Dict

import xml.etree.cElementTree as et
import os
import base64
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def smooth_and_scale(data: Dict, scans_per_second:int=8, window_length:int=5, polyorder:int=3) -> Tuple[NDArray,NDArray]:
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

    rts = np.array([t*60 for t in data["time"]])
    intensities = np.array(data["intensity"])

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
    # make negative values 0
    intensities[intensities < 0] = 0          

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
    data_table = {"transition": [], "rt": [], "intensity": []}

    try:

        tree = et.parse(mzml_file)
        xml_root = tree.getroot()

        for chromatogram in xml_root.iter(f"{namespace}chromatogram"):
            data_table = parse_chromatogram(chromatogram, data_table, namespace, scans_per_second, 
                                window_length, polyorder)

        data_table = format_datatable(data_table, mzml_file)

    except Exception as ex:
        print(ex)

    return data_table


def parse_chromatogram(chromatogram, data_table, namespace, scans_per_second, window_length, polyorder):
    pmz = False
    pcmz = False
    polarity = False
    data = {"time": [], "intensity": []}

    if "SRM" in f"{chromatogram.attrib['id']}":
        pcmz = f"{chromatogram.attrib['id']}".split("Q1=")[1].split(" ")[0]
        pmz = f"{chromatogram.attrib['id']}".split("Q3=")[1].split(" ")[0]

    for node in chromatogram.iter():
        data, polarity = parse_node(node, namespace, data, polarity)
        
    if all([polarity, pcmz, pmz]):
        # scale and smooth
        rts, intensities = smooth_and_scale(
            data = data, scans_per_second = scans_per_second,
            window_length = window_length, polyorder = polyorder
        )

        data_table['transition'].append(np.full(len(rts), f"{pcmz}_{pmz}_{polarity}"))
        data_table['rt'].append(rts)
        data_table['intensity'].append(intensities) 
        
    return data_table                   

def parse_node(node, namespace, data, polarity):
    # find out if positive or negative
    if node.tag == f'{namespace}cvParam':
        if node.attrib["name"] == 'negative scan':                            
            polarity = 'neg'
        elif node.attrib["name"] == 'positive scan':
            polarity = 'pos'                      

    # get rt and ints                        
    elif node.tag == f'{namespace}binaryDataArrayList':                            
        for binaryDataArray in node.iter(f'{namespace}binaryDataArray'):                                
            data = parse_binary(binaryDataArray, data, namespace)

    return data, polarity


def format_datatable(data_table, mzml_file):
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
    return data_table


def parse_binary(binaryDataArray, data, namespace):
    data_type = False
    data_values = []
    for data_node in binaryDataArray.iter():
        if data_node.tag == f'{namespace}cvParam':                             
            if data_node.attrib["name"] in ['time array', 'intensity array']:
                data_type = data_node.attrib["name"].replace(" array", "")
        elif data_node.tag == f'{namespace}binary':
            try:
                decoded_node_data = base64.b64decode(data_node.text)
                data_values = np.frombuffer(decoded_node_data, np.float64)                                        
            except Exception as ex:
                print(ex)
                pass
    
    if data_type and len(data_values) > 0:
        data[data_type] = data_values

    return data