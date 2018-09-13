<b>Experiments on Small Datasets with Deep Learning</b>
<br>
This repository contains experiments to use neural networks to learn classifiers using small datasets. 
<br>
The datasets are:
<br>
•	<b>Breast Cancer</b> Wisconsin Data Set (reference to UCI ML Repository description)<br>
•	<b>Diabetes</b> Data Set (reference to UCI Machine Learning Repository description)<br>
•	<b>Ionosphere</b> Data Set (reference to UCI Machine Learning Repository description)<br>
•	<b>Mushroom</b> Data Set (reference to UCI Machine Learning Repository description)
<br>
For each dataset there are three groups of notebooks:
<br>
•	<b>Data Analysis</b> performs basic data analysis, visualizations of the data sets, outlier detection and feature engineering.<br>
•	<b>ML model search</b> uses traditional sklearn Grid Search to select a traditional sklearn estimator (e.g. Linear Regression) and allows tuning hyper-parameters.<br>
•	<b>ML classifier</b> trains the final estimator and validates its performance using metrics and visualizations.<br>
•	<b>DL model search</b> uses traditional sklearn Grid Search to select a MLP implemented in pytorch and allows tuning hyper-parameters.<br>
•	<b>ML classifier</b> trains the final estimator and validates its performance using metrics and visualizations.
<br>
The notebooks implement the following data analysis, model selection and tuning flow for both traditional ML estimators and NNs:
<br>

![Alt text](images/AnalysisSelectionTuning.jpg?raw=true "")
