import os
import sys
from dataclasses import dataclass

#from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,

)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
                
            )
        
            models = {
                "Ranfom Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K - Neighbors Regression": KNeighborsRegressor(),
                "XGB Regression": XGBRegressor(),
                "Adaboost Regression": AdaBoostRegressor()
            }

            params = {
                "Ranfom Forest": {
                    "n_estimators": [10,100,500],
                    #"criterion": ['gini', 'entropy', 'log_loss'],
                    "max_features": ['sqrt','log2',None]              
                },
                "Decision Tree": {
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 4, 6]

                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.1, 0.01, 0.001],
                    'max_depth': [3, 5, 7]
                },
                "Linear Regression": {
                    'fit_intercept': [True, False],
                    'normalize': [True, False]
                },
                "K - Neighbors Regression": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                },
                "XGB Regression": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.1, 0.01, 0.001],
                    'max_depth': [3, 5, 7]
                },
                "Adaboost Regression": {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.1, 0.01, 0.001]
                }
            }

            model_report = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)

            ]
            best_model = models[best_model_name]
            print(best_model_score[0])
            if best_model_score[0]<0.8:
                raise CustomException("No best model found")
            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        
        

            predicted = best_model.predict(X_test)

            score = r2_score(y_test, predicted)

            return score
        
    

        except Exception as e:
            raise CustomException(e,sys)
        