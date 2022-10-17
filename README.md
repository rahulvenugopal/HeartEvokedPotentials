# HeartEvokedPotentials
![](https://github.com/rahulvenugopal/HeartEvokedPotentials/blob/main/HEP_Sleep.png)

# Sleep HEP pipeline
- Dataset: Overnight polysomnography data in .edf format and the same data scored using Polyman, available as a .edf file
- Neurokit2 to pre-process ECG and identify R peaks
- Adding R peak as a STIM channel and cleaned ECG back to MNE object
- Scroll through data and sanity check ECG data quality and R peaks detected
- Read the hypnogram file and do some dataframe cleanups
- YASA to expand and label each datapoint with corresponding sleep stage
- Epoch the sleep data based on R peak (STIM channel)
- -200 0 +1000 ms
- For each R peak, add the sleep stage, add this sleep stage dataframe as metadata to MNE object
- Minimal epochs rejection based on amplitude and flat criteria
- The above step needs some more additions to chuck out bad epochs
- Saving a log of bad epochs rejected
- Segregating cut epochs based on five sleep stages
- Average the respective epochs to generate sleep HEPs for each stage (W, N1, N2, N3 and R)
- Visualise and save HEPs across all EEG channels
- Pickling the averaged HEPs for cluster based stats

# Stats using TFCE
- Load all the pickled data and select one sleep stage (say REM state)
- As per tutorial (Learn Decoding repo), we need two datasets
- One for meditators and one for controls
- `subjects` * `time` * `frequency` should be the format
- Once done, pass through the function `spatio_temporal_cluster_test`

# To Do
- Surrogate R peaks which are shifted randomly away from actual R peak location
- The above step is to validate the specific effects of R peak comapred to other part of ECG
- Threshold free cluster enhancement (TFCE) stats which run across time and electrodes

# References
1. [Systematic review and meta-analysis of the relationship between the heartbeat-evoked potential and interoception](https://pubmed.ncbi.nlm.nih.gov/33450331/)
2. [Brain–Heart Interactions Underlying Traditional Tibetan Buddhist Meditation](https://academic.oup.com/cercor/article/30/2/439/5510041)
3. [Cortical monitoring of cardiac activity during rapid eye movement sleep: the heartbeat evoked potential in phasic and tonic rapid-eye-movement microstates](https://pubmed.ncbi.nlm.nih.gov/33870427/)
