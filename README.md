# CS349 - AmazonProductEvaluation

CS349 Machine Learning Spring 2023 @ Northwestern University

*Final Model Predictions of the individual project part*

### Dataset

Dataset can be found [here](https://urldefense.com/v3/__https:/drive.google.com/file/d/16lrMrD3w0bnr_rzqEC7qwF2dRv2ulybv/view?usp=share_link__;!!Dq0X2DkFhyF93HkjWTBQKhk!RDEdHaAs_Vesk88fGJJBFe2xssf3I-qSUH-KVFXN-avAS4hM8M27VyhgOQJR14yUHO7Jh56gZ6DfdcIt9qOIEb69zBLFBLwL5Qg$). Dataset used: *Toys and Games*.
<br>
Download dataset and copy (or replace) folder ```Toys_and_Games/``` into ```source/``` to be able to read missing parts of dataset.
<br>
Final Model Predictions on dataset made on ```test3```.

Python Version used: ```Python 3.9.6```.
<br>
Running ```main.py``` reads feature vectors and trains feature vector generated from ```train3``` on a pre-trained AdaBoost random forest classifier.

### .py scripts
- **main.py**: Python file used for PCA, hyperparameter optimization, training and predictions. Instead of training a model, a saved model is loaded here and predictions are made instantly.
- **preprocessing.py**: Python file used for feature generation.
- **correlation.py**: Python file used to examine correlation between features and awesomeness. Used to iteratively tune feature generation in order to maximize correlation of fetaures with awesomeness.
- **overfitting.py**: Python file used to evaluate the f1 score of a model trained on ```train``` and predictions on ```test1```and ```test2```. Goal is to check for overfitting and get a sense of model performance on untrained data.


### Evaluation
Look at the following [Powerpoint](slides/presentation_JC.pptx) to read the full evaluation of this project. The presentation includes visual material on 
- **Feature Selection** (Correlation, PCA)
- **Classification** (MLP, AdaBoost, Random Forest)
- **Prediction Results**

This project is based on a group project deliverable in this class. The presentation may reference previously trained classifiers, which are not accessible here, however.


