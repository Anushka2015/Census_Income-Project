import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 Age :int,
                 workclass:str,
                 fnlwgt :int,
                 education :str,
                 educationnum:int,
                 relationship:str,
                 maritalstatus:str,
                 occupation :str,
                 race:str,
                 sex:str,
                 capitalgain:int,
                 capitalloss:int,
                 hoursperweek  :int,
                 nativecountry:str,
                 makesover :int):
        
        
        self.Age=Age
        self.workclass=workclass
        self.fnlwgt=fnlwgt
        self.education =education 
        self.educationnum=educationnum
        self.relationship=relationship
        self.maritalstatus = maritalstatus
        self.occupation= occupation
        self.race = race
        self.sex = sex
        self.capitalgain = capitalgain
        self.capitalloss = capitalloss
        self.hoursperweek= hoursperweek
        self.nativecountry= nativecountry
        self.makesover= makesover

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age':[self.Age],
                'workclass':[self.workclass],
                'fnlwgt':[self.fnlwgt],
                'education ':[self.education ],
                'education-num':[self.educationnum],
                'relationship':[self.relationship],
                'marital-status':[self.maritalstatus],
                'occupation':[self.occupation],
                'race':[self.race],
                'sex':[self.sex],
                'capital-gain':[self.capitalgain],
                'self.capital-loss':[self.capitalloss ],
                'self.hours-per-week':[self.hoursperweek],
                'native-country':[self.nativecountry],
                'makes over':[self.makesover]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
        
        
