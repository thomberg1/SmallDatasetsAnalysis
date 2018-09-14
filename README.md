<b>Experiments on Small Datasets with Deep Learning</b>
<br><br>
This repository contains experiments to use neural networks to learn classifiers using small datasets. 
<br><br>
The datasets are:
<br><br>
•	<b>Breast Cancer</b> Wisconsin Data Set (reference to UCI ML Repository <a href="http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29">description</a>)<br>
•	<b>Diabetes</b> Data Set (reference to UCI Machine Learning Repository <a href="https://archive.ics.uci.edu/ml/datasets/diabetes">description</a>)<br>
•	<b>Ionosphere</b> Data Set (reference to UCI Machine Learning Repository <a href="https://archive.ics.uci.edu/ml/datasets/ionosphere">description</a>)<br>
•	<b>Mushroom</b> Data Set (reference to UCI Machine Learning Repository <a href="https://archive.ics.uci.edu/ml/datasets/mushroom">description</a>)
<br><br>
For each dataset there are three groups of notebooks:
<br><br>
•	<b>Data Analysis</b> performs basic data analysis, visualizations of the data sets, outlier detection and feature engineering.<br>
•	<b>ML model search</b> uses traditional sklearn Grid Search to select a traditional sklearn estimator (e.g. Linear Regression) and allows tuning hyper-parameters.<br>
•	<b>ML classifier</b> trains the final estimator and validates its performance using metrics and visualizations.<br>
•	<b>DL model search</b> uses traditional sklearn Grid Search to select a MLP implemented in pytorch and allows tuning hyper-parameters.<br>
•	<b>DL classifier</b> trains the final estimator and validates its performance using metrics and visualizations.
<br><br>
The notebooks implement the following data analysis, model selection and tuning flow for both traditional ML estimators and NNs:
<br><br>

![Alt text](images/AnalysisSelectionTuning.jpg?raw=true "")

<br/>
<b>Notes:</b>
•	All process steps have been configured for a 12 core CPU and a 12Mb GPU running on OSX and need to be re-configured if executed on significantly different HW (especially the n_jobs parameter of sklearn functions).<br/>
•	All functions make heavy use of python multi-processing and might hang is a sub-process fails. Use the verbose parameter > 0 to get log information from sklearn for debugging purposes.<br/>
•	Sklearn GridSearchCV is used but RandomizedSearchCV and BayesSearchCV can be plugged in but did not perform in this experiment.<br/>
•	A wrapper called Classifier is used to plug pytorch NNs into the sklearn functions. The code is in the lib directory. A supported version of similar functionality can be found <a href="https://github.com/dnouri/skorch</a>.<br/>
•	The focus of the notebooks is to understand the behavior of traditional ML models versus neural networks if one uses small datasets (i.e. overtraining, overfitting, runtime, metrics). Only minimal feature engineering and individual tuning has been done.<br/>
•	If code has been included form other places the URL of the source blog, repository, etc. has been included above the code.<br/>
