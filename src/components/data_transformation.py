# here we will apply the data transformation techniques using the data from the data ingestion component
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import save_object
import numpy as np
from dataclasses import dataclass
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    '''
        this function is responsible for transforming the data
        it will apply the necessary transformations to the data
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features =[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            logging.info("Numerical features: %s", numerical_features)
            logging.info("Categorical features: %s", categorical_features)

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), #handling missing values for numerical features
                ('scaler', StandardScaler()) #standardizing numerical features
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # handling missing values
                ('onehot', OneHotEncoder(handle_unknown='ignore')),     # encoding categorical features
                ('scaler', StandardScaler(with_mean=False))             # scaling encoded features
            ])


            logging.info('categorical coding completed')
            logging.info('numerical coding completed')

            # Creating a preprocessor object that applies the transformations to the respective features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numerical_pipeline, numerical_features),
                    ('cat_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_features = ['writing_score', 'reading_score']
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)] 

            logging.info("Preprocessing completed")


            # Saving the preprocessor object to a file which is written in utils
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys) from e