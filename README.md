# CS349 - AmazonProductEvaluation
## Individual Project - Final Model Predictions ##

developed by: pqb0185

Dataset used: Toys and Games

Final Model Predictions on dataset ```test3```.

Dataset can be found [here](https://urldefense.com/v3/__https:/drive.google.com/file/d/16lrMrD3w0bnr_rzqEC7qwF2dRv2ulybv/view?usp=share_link__;!!Dq0X2DkFhyF93HkjWTBQKhk!RDEdHaAs_Vesk88fGJJBFe2xssf3I-qSUH-KVFXN-avAS4hM8M27VyhgOQJR14yUHO7Jh56gZ6DfdcIt9qOIEb69zBLFBLwL5Qg$)

Download Dataset and copy folder Toys_and_Games/ into source/ to be able to read missing parts of dataset.

Python Version used: ```Python 3.9.6```

Running ```main.py``` reads feature vectors and trains feature vector generated from ```train3``` on a pre-trained AdaBoost random forest classifier (similar to the one handed in in the final team deliverable).

### .py scripts
- **main.py**: Python file used for PCA, hyperparameter optimization, training and predictions. Instead of training a model, a saved model is loaded here and predictions are made instantly.
- **preprocessing.py**: Python file used for feature generation.
- **correlation.py**: Python file used to examine correlation between features and awesomeness. Used to iteratively tune feature generation in order to maximize correlation of fetaures with awesomeness.
- **overfitting.py**: Python file used to evaluate the f1 score of a model trained on ```train``` and predictions on ```test1```and ```test2```. Goal is to check for overfitting and get a sense of model performance on untrained data.
