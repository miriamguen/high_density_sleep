This is the data related to the following publication:

ANPHY-Sleep: Open Sleep Database from Healthy Adults using High-Density Scalp Electroencephalogram

Xiaoyan Wei, Tamir Avigdor, Alyssa Ho, Erica Minato, Alfonso Garcia-Asensi, Jessica Royer, Yingqi Wang, 
Katharina Schiller, Boris C. Bernhardt, Birgit Frauscher 


The ANPHY-Sleep database contains:

1/EPCTL01, EPCTL02,... EPCTL29 folders contain combined polysomnographic and HD-EEG signals and the corresponding sleep annotations.
The signals are stored using the European Data Format (EDF). The sleep scoring annotation is saved in the TXT file format
which includes the sleep stage label, the start time of the corresponding sleep stage, and the duration of each sleep stage.
   Notes: the "L" represents the experiment light on/off state, "N1" represents the non-rapid eye movement stage 1, "N2" represents
    the non-rapid eye movement stage 2, "N3" represents the non-rapid eye movement stage 3, and "R" represents the rapid eye movement stage.

2/Average_digitized position.txt contains the average digitized electrode co-registered locations from all subjects

3/Deail information of subjects.xlsx contains the general information about the sex, age, and sleep properties of each subject

4/ Code.zip contains tutorials written in Python for the data repository for basic handling and analysis of the files, such as file opening, sleep stage file conversion, demographics, and sleep parameters analysis. The details of the scripts are as follows:
       The script named  " scoring convert.py"  could extract the sleep stage label from the HTML report exported Polysmith 12 version.
       The script " demographics.py" plots the subjects' gender and age distribution.
       The script " se.py" plots the subjects' sleep efficiency distribution.
       The code " sleep parameters.py" plots the subjects' sleep parameters boxplot.
       The code " hypnogram.py" plots the hypnogram of subject 16.
