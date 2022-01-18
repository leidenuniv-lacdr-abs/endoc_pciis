import os
from glob import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.mzml_parser import df_from_mzml

# Now we are ready to determine the area of a peak
# We do this with the help of the peak_widths function
# from the scipy package, and the detect_peaks function
# for the package detecta.
from scipy.signal import peak_widths
import warnings

# Duarte, M. (2020) detecta: A Python module to detect events in data. 
# GitHub repository, https://github.com/demotu/detecta.
from detecta import detect_peaks

# The phase align code has been used from 
# GitHub: https://github.com/pearsonkyle/Signal-Alignment
# Citation: http://iopscience.iop.org/article/10.3847/1538-3881/aaf1ae/meta
from scipy.ndimage import shift
from src.signal_alignment import phase_align

# Altair for visualization
import altair as alt
from altair_saver import save

class PciisMsmsExperiment:
    
    def __init__(self, path:Path, name:str="") -> None:

        if name == "": # get name from path
            self.name = path.name
        else:
            self.name = name

        self.path = path
                
    def __str__(self) -> str:
        return self.name

    def set_targets(self, targets_file:Path):

        # Define the targets to apply PCI_IS correction
        # by reading them in from a csv file
        self.targets = pd.read_csv(
            targets_file, sep="\t",
            converters = {'precursor': str, 'product': str}
        )

        # Add column with unique transition information
        self.targets['transition'] = self.targets.apply(
            lambda x: f"{x['precursor']}_{x['product']}_{x['polarity']}", axis=1
        )

    def set_pciis(self, pciis_file:Path):

        self.pciis = pd.read_csv(
            pciis_file, sep="\t",
            converters = {'precursor': str, 'product': str}
        )

        # Add column with unique transition information
        self.pciis['transition'] = self.pciis.apply(
            lambda x: f"{x['precursor']}_{x['product']}_{x['polarity']}", axis=1
        )

        # TODO: build support for multiple pciis
        self.pciis = self.pciis.iloc[0] # use pciis from first line

    def load_data(self, mzml_file_path:Path, scans_per_second:int=8, window_length:int=5, polyorder:int=3):

        # Read in raw signal data from mzml files
        self.mzml_files = glob(os.path.join(mzml_file_path,'*.mzML'))

        # Collect all individual DataFrames as a list
        self.data = pd.concat(
            [df_from_mzml(mzml_file, scans_per_second, window_length, polyorder) for mzml_file in self.mzml_files]
        )

    def apply_pciis(self):
        
        # Extract PCI_IS signal
        pciis_filter = self.data['transition'] == self.pciis['transition']
        pciis_data = self.data[pciis_filter][['file', 'rt','intensity']].copy()

        # Change the column name of the intensity to pciis and 
        # set the index of the DataFrame to file/rt to be able 
        # to stitch the column to the target signal
        pciis_data.rename(columns={'intensity': 'pciis'}, inplace=True)
        pciis_data = pciis_data.set_index(['file', 'rt'])

        # Combine signal from all targets into a single DataFrame
        dfs = []
        for tIdx, target in self.targets.iterrows():
            
            # Extract target signal
            target_filter = self.data['transition'] == target['transition']
            target_data = self.data[target_filter].copy()
            
            # Keep index, in case we have multiple targets with the same transition
            target_data['target'] = tIdx
            
            # Prepare index for join with pciis signal using file/rt
            target_data = target_data.set_index(['file', 'rt'])    
            
            # Stitch the PCI_IS signal to the dataframe as a column
            dfs.append(target_data.join(pciis_data['pciis']).reset_index())
            
        # Combine the DataFrames again    
        target_data = pd.concat(dfs)

        # Extract concentration information from filename
        target_data['concentration'] = target_data['file'].apply(
            lambda x: x.split("_")[-1].replace(".mzML","")
        )

        # Extract sample information from filename
        target_data['sample'] = target_data['file'].apply(
            lambda x: x.split("_")[-2]
        )

        # Calculate ratio between intensity and pciis
        target_data['ratio'] = target_data['intensity']/target_data['pciis']

        self.data = target_data

    def apply_rt_correction(self, reference_file:Path):

        df = self.data.copy()

        # Extract ratio by target/rt from the reference file
        reference__filter = df['file'] == str(reference_file)
        reference_ratio = df[reference__filter][['target','rt','ratio']].copy()
        reference_ratio = reference_ratio.set_index(['target','rt'])

        # Change the column name
        reference_ratio.rename(columns={'ratio': 'ratio_reference'}, inplace=True)

        # Append a column with the ratio by rt from the reference file
        df_aligned = df.set_index(['target','rt']).join(reference_ratio).reset_index().dropna()

        # Apply alignement by target/file
        aligned_dfs = []
        for gIdx, df_grouped in df_aligned.groupby(['target','file']):
                        
            df_grouped['ratio_aligned'] = shift( # apply phase shift
                df_grouped['ratio'], 
                float(phase_align( # calculate phase shift
                    df_grouped['ratio_reference'].values, df_grouped['ratio'].values, res=1
                )), 
                mode='constant', 
                cval=0.0
            )
            
            aligned_dfs.append(df_grouped)
            
        # Merge them back together
        df_aligned = pd.concat(aligned_dfs, ignore_index=True)

        # Make all negative values = 0. Negative values could appear
        # due to the Savitzky-Golay filter that has been applied
        df_aligned['ratio_aligned'] = df_aligned['ratio_aligned'].clip(0) 

        self.data = df_aligned

    def find_peaks(self):
        # Initialize an empty list to collect all peak DataFrames found with
        # the required meta-data such as apex, width, start, end, etc.
        peak_dfs = []
        for tIdx, df_target in self.data.groupby('target'):
            
            # Get target details
            t = self.targets.loc[tIdx]
            
            for fIdx, df_target_file in df_target.groupby('file'):
            
                # By signal type (intensity or ratio)
                for signal_type in ['intensity','ratio','ratio_aligned']:
                    peak_index = detect_peaks(
                        df_target_file[signal_type],
                        mph=0,
                        mpd=8,
                        threshold=0,
                        edge='both',
                        kpsh=True
                    )

                    # Suppress the warnings of the peak_width function
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        peak_width_index = peak_widths(
                            df_target_file[signal_type],
                            peak_index,
                            rel_height=0.98
                        )

                    # Collect peak meta-data
                    peak_rt_apex = df_target_file.iloc[peak_index,]['rt']
                    peak_rt_start = df_target_file.iloc[peak_width_index[2],]['rt']
                    peak_rt_end = df_target_file.iloc[peak_width_index[3],]['rt']
                    peak_width = [ pw[0] - pw[1] for pw in zip(peak_rt_end,peak_rt_start) ]

                    peaks_df = pd.DataFrame({
                        'peak_rt_apex':peak_rt_apex.tolist(),
                        'peak_rt_start':peak_rt_start.tolist(),
                        'peak_rt_end':peak_rt_end.tolist(),
                        'peak_width':peak_width,
                    })                   

                    # Calculate the absolute error between the retention time found of the 
                    # apex and the expected retention time as predefined in the targets file.
                    peaks_df['abs_rt_error'] = peaks_df['peak_rt_apex'].apply(
                        lambda x: abs(x - t.rt)
                    )

                    # We are only interested in the peak with the smallest retention time error.
                    peaks_df = peaks_df.nsmallest(1, 'abs_rt_error')
                    
                    # Keep track of where the peaks where found, in what 
                    # sample, of what target, and based on what signal.
                    peaks_df['signal_type'] = signal_type
                    peaks_df['target'] = tIdx
                    peaks_df['file'] = fIdx            

                    # Calculate the peak area
                    peaks_df['area'] = peaks_df.apply(
                        lambda row:
                            df_target_file[
                                (df_target_file['rt'] >= row['peak_rt_start']) & 
                                (df_target_file['rt'] <= row['peak_rt_end'])
                            ][signal_type].sum()
                        ,axis=1
                    )
            
                    # Keep track of the number of scans used to determine the peak area
                    peaks_df['scans'] = peaks_df.apply(
                        lambda row:
                            len(df_target_file[
                                (df_target_file['rt'] >= row['peak_rt_start']) & 
                                (df_target_file['rt'] <= row['peak_rt_end'])
                            ][signal_type])
                        ,axis=1
                    )
                    
                    # And add it to the list to combine later
                    peak_dfs.append(peaks_df)
                    
        # Merge them back together
        self.peaks = pd.concat(peak_dfs, ignore_index=True)  

        # Add concentration
        self.peaks['concentration'] = self.peaks['file'].apply(
            lambda x: x.split("_")[-1].replace(".mzML","")
        )

        # Add sample
        self.peaks['sample'] = self.peaks['file'].apply(
            lambda x: x.split("_")[-2]
        )

        # Save peaks as csv file
        self.peaks.to_csv('peaks_found.csv', index=False)

    def plot_results(self, rt_window:int=12, width:int=210, height:int=180):

        # Plot signals of all targets to visualize and compare the impact
        # of PCI_IS correction on the signal

        for tIdx, target in self.targets.iterrows():

            plots = alt.vconcat()        
            
            # Show the plots for each concentration ('LOW','MEDIUM','HIGH')
            for concentration in self.data['concentration'].unique():
                
                # Data frame with peaks of target, grouped by concentration
                df_concentration_peaks = self.peaks[
                    (self.peaks['target'] == tIdx) &
                    (self.peaks['concentration'] == concentration)
                ].copy()

                # Data frame with signals of target, grouped by concentration
                df_concentration_signal = self.data[   
                    (self.data['target'] == tIdx) &                
                    (self.data['rt'] >= (target['rt'] - rt_window/2)) &
                    (self.data['rt'] <= (target['rt'] + rt_window/2)) &
                    (self.data['concentration'] == concentration)
                ].copy()

                # Original signal
                intensity_plot = alt.Chart(df_concentration_signal).mark_line().encode(
                    x='rt', y='intensity', color='sample'
                ).properties(
                    width=width, height=height, 
                    title=f"intensity ({concentration}) {target['name']}"
                )
                
                # Pciis signal
                intensity_plot = intensity_plot + alt.Chart(df_concentration_signal).mark_line(
                    strokeWidth=0.5
                ).encode(
                    x='rt', y='pciis', color='sample'
                ).properties(
                    width=width, height=height
                )        

                # Add rt of target (red verticle line)
                intensity_plot = intensity_plot + alt.Chart(
                    df_concentration_peaks[df_concentration_peaks['signal_type'] == 'intensity']).mark_rule(
                        color='red', strokeWidth=2
                    ).encode(
                        alt.X('mean(peak_rt_apex)',
                        title='rt')        
                    )

                # Add area window (start) of target (red verticle line)
                intensity_plot = intensity_plot + alt.Chart(
                    df_concentration_peaks[df_concentration_peaks['signal_type'] == 'intensity']).mark_rule(
                        color='grey', strokeWidth=2
                    ).encode(
                        x='min(peak_rt_start)'
                    )

                # Add area window (end) of target (red verticle line)
                intensity_plot = intensity_plot + alt.Chart(
                    df_concentration_peaks[df_concentration_peaks['signal_type'] == 'intensity']).mark_rule(
                        color='grey', strokeWidth=2
                    ).encode(
                        x='min(peak_rt_end)'
                    )   

                # Ratio
                unaligned_plot = alt.Chart(df_concentration_signal).mark_line().encode(
                    x='rt', y='ratio', color='sample'
                ).properties(
                    width=width, height=height, 
                    title=f"ratio ({concentration}) {target['name']}"
                )

                # Add rt of target (red verticle line)
                unaligned_plot = unaligned_plot + alt.Chart(
                    df_concentration_peaks[df_concentration_peaks['signal_type'] == 'ratio']).mark_rule(
                        color='red', strokeWidth=2
                    ).encode(
                        alt.X('mean(peak_rt_apex)',
                        title='rt')        
                    )

                # Add area window (start) of target (red verticle line)
                unaligned_plot = unaligned_plot + alt.Chart(
                    df_concentration_peaks[df_concentration_peaks['signal_type'] == 'ratio']).mark_rule(
                        color='grey', strokeWidth=2
                    ).encode(
                        x='min(peak_rt_start)'
                    )

                # Add area window (end) of target (red verticle line)
                unaligned_plot = unaligned_plot + alt.Chart(
                    df_concentration_peaks[df_concentration_peaks['signal_type'] == 'ratio']).mark_rule(
                        color='grey', strokeWidth=2
                    ).encode(
                        x='min(peak_rt_end)'
                    )    

                # Ratio aligned + target rt
                aligned_plot = alt.Chart(df_concentration_signal).mark_line().encode(
                    x='rt', y='ratio_aligned', color='sample'
                ).properties(
                    width=width, height=height, 
                    title=f"ratio aligned ({concentration}) {target['name']}"
                )

                # Add rt of target (red verticle line)
                aligned_plot = aligned_plot + alt.Chart(
                    df_concentration_peaks[df_concentration_peaks['signal_type'] == 'ratio_aligned']).mark_rule(
                        color='red', strokeWidth=2
                    ).encode(
                        alt.X('mean(peak_rt_apex)',
                        title='rt')        
                    )

                # Add area window (start) of target (red verticle line)
                aligned_plot = aligned_plot + alt.Chart(
                    df_concentration_peaks[df_concentration_peaks['signal_type'] == 'ratio_aligned']).mark_rule(
                        color='grey', strokeWidth=2
                    ).encode(
                        x='min(peak_rt_start)'
                    )

                # Add area window (end) of target (red verticle line)
                aligned_plot = aligned_plot + alt.Chart(
                    df_concentration_peaks[df_concentration_peaks['signal_type'] == 'ratio_aligned']).mark_rule(
                        color='grey', strokeWidth=2
                    ).encode(
                        x='min(peak_rt_end)'
                    )
                
                plots = alt.vconcat(plots, ((intensity_plot | unaligned_plot | aligned_plot)))

            area_plots = []
            for signal_type in ['intensity','ratio_aligned']:

                # Filtered data source
                peaks_filter = f"target == {tIdx} and signal_type == '{signal_type}'"    
                source = self.peaks.query(peaks_filter)

                error_points = alt.Chart(source).mark_point(filled=True).encode(
                    x=alt.X('area:Q', aggregate='mean', scale=alt.Scale(domain=[source['area'].min(),source['area'].max()])),
                    y=alt.Y('concentration:N', sort=alt.SortField('sample')),
                    color=alt.Color("concentration", legend=None)
                ).properties(width=width * 1.5, height=height/2, title=f"{target['name']}: mean area + error bars (based on: {signal_type})")    

                # Mean area by concentration error bars
                error_bars = alt.Chart(source).mark_errorbar(extent='ci').encode(
                    x=alt.X('area:Q', axis=alt.Axis(labels=False), scale=alt.Scale(domain=[source['area'].min(),source['area'].max()])),
                    y=alt.Y('concentration:N', sort=alt.SortField('sample')),
                    color=alt.Color("concentration", legend=None)
                ).properties(width=width * 1.5, height=height/2)

                area_plots.append((error_points + error_bars))                        

            plots = alt.vconcat(plots, (area_plots[0] | area_plots[1]))
                        
            # Display the plots        
            plots.display()

            del plots