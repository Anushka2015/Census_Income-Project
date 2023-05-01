import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

def save_object(file1_path, obj):
    try:
        dir1_path = os.path.dirname(file1_path)

        os.makedirs(dir1_path, exist_ok=True)

        with open(file1_path, "wb") as file1_obj:
            pickle.dump(obj, file1_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        report={}
        for i in range(len(model)):
            model=list(model.values())[i]
            #train model
            model.fit(X_train,y_train)
        
    
            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get accuracy_score for train and test data
            #train_model_score = accuracy_score(y_train,y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred)
            report[list(model.keys())[i]]=test_model_score
        return report
    
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

def load_object(file1_path):
    try:
        with open(file1_path,'rb') as file1_obj:
            return pickle.load(file1_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

