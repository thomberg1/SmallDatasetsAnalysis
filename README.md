<p/>
This repository contains experiments to use neural networks to learn classifiers using small datasets. 
<p/>
The datasets are:
<p/>
•	<b>Breast Cancer</b> Wisconsin Data Set (reference to UCI ML Repository description)<p/>
•	<b>Diabetes</b> Data Set (reference to UCI Machine Learning Repository description)<p/>
•	<b>Ionosphere</b> Data Set (reference to UCI Machine Learning Repository description)<p/>
•	<b>Mushroom</b> Data Set (reference to UCI Machine Learning Repository description)
<p/>
For each dataset there are three groups of notebooks:
<p/>
•	<b>Data Analysis</b> performs basic data analysis, visualizations of the data sets, outlier detection and feature engineering.<p/>
•	<b>ML model search</b> uses traditional sklearn Grid Search to select a traditional sklearn estimator (e.g. Linear Regression) and allows tuning hyper-parameters.<p/>
•	<b>ML classifier</b> trains the final estimator and validates its performance using metrics and visualizations.<p/>
•	<b>DL model search</b> uses traditional sklearn Grid Search to select a MLP implemented in pytorch and allows tuning hyper-parameters.<p/>
•	<b>ML classifier</b> trains the final estimator and validates its performance using metrics and visualizations.
<p/>
The notebooks implement the following data analysis, model selection and tuning flow for both traditional ML estimators and NNs:<p/>

