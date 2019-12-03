# DATA1030-Final-Project

Lauren To, Brown University. Under the guidance of Andras Zsom.

This is a machine learning classification project that aims to predict whether or not an individual was able to succcessfully complete a substance use disorder treatment program, given a number of demographic and substance use features. I use a subset of the Treatment Episode Data Set - Discharges, 2017 (TEDS-D 2017) data set, made available by the SAMHSA, specified to cases pertaining to opioid dependency and outpatient services. In this project, I use a logistic model and a random forest classification model with different tuned parameters to measure optimal performance. 

I use the following versions of Python/various Python packages: Python version 3.7.2; Anaconda version 4.2.0; sklearn version 0.21.3; pandas version 0.24.1; and numpy version 1.16.2. 

*data*: In the form of zip files. Contains the opioid dependency-outpatient subset of the original data set, as well as a pre-processed permutation for EDA, a permutation to be used for training, and a codebook that explains the data set coding methodologies and categories. 

*figures*: Contains all of the figures that were used in my final report analysis.

*results*: Contains a save file with the predictions made by both models after 5 training rounds with different random states, as well as a clean table with the final results. The zip file for the CV grids themselves were too large to store in GitHub.

*reports*: A proposal and a final report.

*src*: All of the source code for this project.
- proposal.py: Contains all preliminary recoding, feature dropping, and MCAR testing, as well as encoding and imputation for EDA.
- preprocessing.py: Contains f-score and mutual information classification code.
- mlcv.ipynb: Contains the construction of the models and training, and storage of parameters/grids. 
- feature_results.ipynb: Contains global and local feature analysis of the results of the grid. 
