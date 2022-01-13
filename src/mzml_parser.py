from typing import List, Tuple
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

# global variables (local)
_NAMESPACE = '{http://psi.hupo.org/ms/mzml}'

def smooth_and_scale(rts:List=[], intensities:List=[],scans_per_second:int=8, window_length:int=5, polyorder:int=3) -> Tuple[NDArray,NDArray]:
    """[summary]

    Args:
        rts (List): Retention times
        intensities (List): Intensities
        scans_per_second (int, optional): Number of scans per second to up/down scale to. Defaults to 8.
        window_length (int, optional): Window length for scipy.signal.savgol_filter function. Defaults to 5.
        polyorder (int, optional): Polyorder for scipy.signal.savgol_filter function. Defaults to 3.

    Returns:
        Tuple[NDArray,NDArray]: Both the new retention times and intensities are returned.
    """

    rts = np.array([t*60 for t in rts])
    intensities = np.array(intensities)

    # prepare interpolate function based on input data
    interpolate_func = interp1d(rts,intensities,kind='linear')
    
    # determine new rt range
    rt_range = np.linspace(0,9999,int(scans_per_second * 9999)).round(3)
    rts = rt_range[(rt_range >= rts.min()) & (rt_range <= rts.max())]
    
    # apply up/down scaling and include smoothing
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

    df = pd.DataFrame()

    try:
        chrom_dfs = []

        tree = et.parse(mzml_file)
        xml_root = tree.getroot()

        for chromatogram in xml_root.iter(f"{_NAMESPACE}chromatogram"):
            
            # retrieve the transition, rt, and intensity information
            chrom_dict = parse_chromatogram(chromatogram)

            # scale and smooth when the chromatogram contains actual data
            if chrom_dict.keys():
                chrom_dict['rts'], chrom_dict['intensities'] = smooth_and_scale( 
                    rts=chrom_dict['rts'],
                    intensities=chrom_dict['intensities'],
                    scans_per_second = scans_per_second,
                    window_length = window_length,
                    polyorder = polyorder
                )

                # convert it to a dataframe, and append it to the list
                chrom_dfs.append(chromatogram_as_df(chrom_dict))
                
        # concat the chromatogram data frames
        df = pd.concat(chrom_dfs, ignore_index=True)         
                
        # include file
        df['file'] = str(mzml_file)

        # include sample
        df['sample'] = df['file'].apply(lambda x: x.split(os.path.sep)[-1].replace(".mzML", ""))

    except Exception as ex:
        print(ex)      

    return df


def parse_chromatogram(chromatogram):
    """[summary]

    Args:
        chromatogram ([type]): [description]

    Returns:
        [type]: [description]
    """

    if not "SRM" in f"{chromatogram.attrib['id']}":            
        return {}

    precursor_mz = f"{chromatogram.attrib['id']}".split("Q1=")[1].split(" ")[0]
    product_mz = f"{chromatogram.attrib['id']}".split("Q3=")[1].split(" ")[0]

    for node in chromatogram.iter():
        
        # parameter node
        if node.tag == f'{_NAMESPACE}cvParam':
            if node.attrib["name"] == 'negative scan':                            
                polarity = 'neg'
            elif node.attrib["name"] == 'positive scan':
                polarity = 'pos'                      

        # signal node                        
        elif node.tag == f'{_NAMESPACE}binaryDataArrayList':                                                                            
            rts, intensities = get_signal(node)
    
    # construct transition
    transition = f"{precursor_mz}_{product_mz}_{polarity}"
        
    return {'transition':transition, 'rts':rts, 'intensities':intensities}

def chromatogram_as_df(chrom_dict:dict):
    """[summary]

    Args:
        chrom_dict (dict): [description]

    Returns:
        [type]: [description]
    """

    df = pd.DataFrame({
            'transition':np.full(len(chrom_dict['rts']), chrom_dict['transition']),
            'rt':chrom_dict['rts'], 
            'intensity':chrom_dict['intensities']
        }
    )     

    df['transition'] = df['transition'].astype('category')
    df['rt'] = df['rt'].astype(float).round(decimals=5)
    df['intensity'] = df['intensity'].astype(int)    
    
    return df

def get_signal(node):
    """[summary]

    Args:
        node ([type]): [description]

    Returns:
        [type]: [description]
    """

    signal_data = {}
    for binaryDataArray in node.iter(f'{_NAMESPACE}binaryDataArray'):
        for data_node in binaryDataArray.iter():
            if data_node.tag == f'{_NAMESPACE}cvParam':                             
                if data_node.attrib["name"] in ['time array', 'intensity array']:
                    data_type = data_node.attrib["name"].replace(" array", "")
            elif data_node.tag == f'{_NAMESPACE}binary':
                try:
                    decoded_node_data = base64.b64decode(data_node.text)
                    signal_data[data_type] = np.frombuffer(decoded_node_data, np.float64)                                        
                except Exception as ex:
                    print(ex)
                    pass
    
    return signal_data['time'], signal_data['intensity']