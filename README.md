# Installation

For installation process there are two scripts: one for UNIX systems, the other one for Windows.

**Note that Python must be installed on the machine.**

The install script is named **Install.sh**. The install script will install virtualenv (Note: you can edit the name of created virtualenv editing the Install.sh file), create a
virtual environment named venv, and install into the created virtual environment packages
from requirement.txt.

 - First place yourself in the folder where DS3 is stored.
 - We need to give executable permissions to our Install.sh file.

---
    $> chmod +x Install.sh
---
 
 - Then execute the Install.sh file.

---
    $> ./Install.sh
---


## Windows (only)

The installation file is **Install.bat**. For Windows just execute it the file.

## Execution

Files Start.sh and Start.bat take care of execution. Of course .sh file is for UNIX and
.bat file is for Windows. As the installation process in UNIX we need to give executable
permissions to our file and then run it from the console.

---
    $> chmod +x Start.sh
	$>./Start.sh
---
**Windows only**
For Windows instead we just need to double-click the .bat file.
Now we take a look of the two "parts" of DS3.

## Dataset
The dataset used for this experiment is available at the following repository:

  - [https://github.com/penguincwarrior/CNN-WPDP-APPSCI2019](https://github.com/penguincwarrior/CNN-WPDP-APPSCI2019)




## Baseline
As the name suggests, *Baseline* refers to a classic approach to software defect prediction.
It gives the possibility to use the following
machine learning techniques to train and predict bug modules:
 
 - LogistiRegression
 - RandomForest
 - SVC (Support Vector Classifier)

It just needs to have the **Folder Path** where .csv of different versions of a project are
stored.
Other important choice is using or not DS3. The tool offers more options related to dissimilarity distances. 
Rather than using only chi-square, the tool provides the following metrics: Euclidean and Manhattan, as Figure 8.4 shows.
Others choices are:

 - Verbose (y/N): Print or not logs.
 - Plot (y/N): Save or not results plot in a dedicated plot folder inside the tool.

## CNN

Differently from Baseline the
CNN process requires more input from the user: not only **Folder Path** where *.csv* files
are stored, but also the **Folder Path** where mapped files are stored and the **Folder Path** where
embedded files are stored.
The rest of process for the user is similar to Baseline, even if only chi-square is chosen for
DS3 and only LogistiRegression or RandomForest is chosen as classification model.

## The results

Both the processes will save the predictions in a .csv file in the Results folder.
The header of those files will be the following:

 - project
 - version
 - name
 - bug predicted

The name of results file will follow this paradigm: "Prediction for *{SoftwareName}* +
version + *{version of the software tested}* + *{process used}* + *{use or not of DS3}*.csv", for example: "Prediction for ant version 1.4 Logistic DS3.csv". 
Also plot’s name will have the same paradigm name.
