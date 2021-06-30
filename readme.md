Children's 12 Lead ECG Analysis

While a good few machine learning models have been studied on adult ECGs over the past decade, there is far less happening in the realm of child cardiology. With kids, normal ranges are much harder to define making modeling tricky and while heart disease prevelance in children is much lower being able to diagnos complications from inhereted heart issues would extend lives and quality of these lives greatly. With ~170K ECGs from a large children's hospital this is an attempt to see what insight can be derived.

Project Folders:

Processing  
	- xml_to_csv_ecg.py
		- helper function to convert ECG event from GEMuse XML outfile to csv of waveform data
	- data_load_setup.py
		- creates a folder of patient waveforms linked to labels that can be utilized by pytorch data 	 loader class and models
	- dx_labeler.py
		- takes doctor free text from dx1-dx10 fields and applies language logic to bin into   predetermined diagnosis buckets for 4 categories(rhythm, morphology, hypertrophy, long QT)

Modeling
	- dx_kmeans.py
		- NLP clustering on doctor diagnosis descriptions as reference to help in creating labels for 	 future models
	- MuseRandFor.py
		- Takes tabular data on patient demographics and signals calculated by GEMuse ECG tool and runs random forest to set baseline for future models for diagnosis categorization. 
	- muse_rhythm_classifier.py
		- Takes tabular data on patient demographics and signals calculated by GEMuse ECG tool and runs neural network for rhythm diagnosis classification. 
	- cnn_waveforms.py
		- Loads full 12 lead waveform timeseries data(12*5000) and feeds into convolution model to learn
		  how to calculate QTC interval and predict diagnosis.

