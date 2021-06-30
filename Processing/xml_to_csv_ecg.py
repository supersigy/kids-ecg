"""
Created on Mon April 05 10:10:06 2021
@author: Gabor Asztalos

Helper function to convert a GEMuse XML based ECG event into two csv files: 
one for median, one for rhythm readings
Requires: infile(path to XML), outfile(desired path and name for conversion)
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import base64

def xml_to_csv_ecg(infile, outfile):
	ecg = ET.parse(infile) 
	root = ecg.getroot()
	
	reqno = ecg.find('Order').find('RequisitionNumber').text
	
	for waveform in ecg.iter('Waveform'):
	    col_names = []
	    wave_form_type = waveform.find('WaveformType').text
	    col_names = ['Wave_Form_Type']
	    wave_form_data = []
	    for lead in waveform.iter('LeadData'): 
	        sample_rate = lead.find('LeadSampleCountTotal').text
	        if sample_rate == '5000':
	            lead_id = lead.find('LeadID').text
	            col_names.append(lead_id)
	            wave_form_data.append(lead.find('WaveFormData').text)
	    if wave_form_type == 'Median':
	        median_df = pd.DataFrame(columns = col_names)
	        row = [wave_form_type]
	        row.extend(wave_form_data)
	        #rhythm_df.loc[len(rhythm_df)] = row
	        median_df = median_df.append(pd.DataFrame([row], columns = col_names),ignore_index=True)
	    else:
	        rhythm_df = pd.DataFrame(columns = col_names)
	        row = [wave_form_type]
	        row.extend(wave_form_data)
	        #rhythm_df.loc[len(rhythm_df)] = row
        	rhythm_df = rhythm_df.append(pd.DataFrame([row], columns = col_names),ignore_index=True)
        	
        median_df.to_csv(outfile+'_m.csv')
        rhythm_df.to_sv(outfile+'_r.csv')