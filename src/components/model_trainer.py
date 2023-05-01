#Basic Import
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file1_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model={"LinearRegression":LogisticRegression()}
            
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,model)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get accuracy_score from dictionary 
            accuracy_score = max(sorted(accuracy_score.values()))

            model_name = list(model_report.keys())[
                list(model_report.values()).index(accuracy_score)
            ]
            
            best_model = model[model_name]

            print(f'Model Name : {model_name} , accuracy_score : {accuracy_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {model_name} , accuracy_score : {accuracy_score}')

            save_object(
                 file1_path=self.model_trainer_config.trained_model_file1_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
