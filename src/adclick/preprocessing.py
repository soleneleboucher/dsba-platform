import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


class DataPreprocessor:
    def __init__(self,
                 categorical_vars = ['Region', 'Carrier', 'Weekday', 'Social_Network', 'Restaurant_Type'],
                 numerical_vars = [	'Time_On_Previous_Website',	'Number_of_Previous_Orders', 'Daytime'],
                 target_col="Clicks_Conversion", 
                 test_size=0.2, 
                 smote_threshold=0.3,
                 scaler_path="scaler.pkl"):
        """
        Initialize with list of categorical and numerical features
        Parameters:
        - Categorical variables: List of categorical variables to encode (Default: 'Region', 'Carrier', 'Weekday', 'Social_Network', 'Restaurant_Type')
        - Numerical variables: List of numerical columns to scale (Default: 'Time_On_Previous_Website',	'Number_of_Previous_Orders', 'Daytime')
        - target_col: Target column name (default: "Clicks_Conversion")
        
        """

        self.categorical_vars = categorical_vars
        self.numerical_vars = numerical_vars
        self.target_col = target_col
        self.test_size = test_size
        self.smote_threshold = smote_threshold
        self.scaler_path = scaler_path
        self.scaler = MinMaxScaler()


    def handle_missing_values(self, df):
        """Handles missing values"""
        df['Restaurant_Type'] = df['Restaurant_Type'].replace(["nan", np.nan], "Unknown")
        return df


    def encode_categorical(self, df):
        """Applies one-hot encoding to categorical features."""
        return pd.get_dummies(df, columns=self.categorical_vars, drop_first=True, dtype=int)


    def feature_engineering(self, df):

        # Interaction terms - combined effect of variables
        df['DaytimeXOrders'] = df['Daytime'] * df['Number_of_Previous_Orders']

        # Recency Transformation (Inverse Time)
        df['InvTimeonPrevSite'] = 1 / (1 + df['Time_On_Previous_Website'])

        # Weekend Indicator (Binary)
        df['Is_Weekend'] = ((df['Weekday_Saturday'] == 1) | (df['Weekday_Sunday'] == 1)).astype(int)

        return df


    def preprocess(self, df):
        """Runs all preprocessing steps sequentially"""
        """
        Prepares data for machine learning by applying preprocessing steps sequentially:
        - Handles missing values.
        - One-hot encodes categorical variables.
        - Generates new feature interactions.

        Parameters:
        - df (pd.DataFrame): Raw input dataset.

        Returns:
        - df (pd.DataFrame): Processed dataframe
        """
        
        # 1. Handle missing values
        df = self.handle_missing_values(df)
        # 2. Encode categorical variables
        df = self.encode_categorical(df)
        # 3. Feature engineering
        df = self.feature_engineering(df)
        
        # 4. Return train and test processed data
        return df
