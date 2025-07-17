import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import os
import kagglehub
import warnings
warnings.filterwarnings("ignore")

class CreditCardFraudPreprocessor:
    """
    A class to preprocess credit card fraud data including:
    - Loading data from Kaggle or local file
    - Scaling features
    - Preparing for 5-fold cross-validation
    """
    
    def __init__(self, download_data=True, file_path=None):
        """
        Initialize the preprocessor with configuration options.
        
        Parameters:
        -----------
        download_data : bool, default=True
            Whether to download the data from Kaggle using kagglehub
        file_path : str, default=None
            Path to the creditcard.csv file (used if download_data=False)
        """
        self.download_data = download_data
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        
    def load_data(self):
        """
        Load the credit card fraud data from Kaggle or local file.
        
        Returns:
        --------
        pd.DataFrame
            The loaded dataframe
        """
        if self.download_data:
            print("Downloading credit card fraud dataset from Kaggle...")
            try:
                # Download the dataset
                path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
                print("Dataset downloaded to:", path)
                
                # List files in the directory
                files = os.listdir(path)
                print("Files in the directory:", files)
                
                # Find the CSV file
                csv_file = next((f for f in files if f.endswith('.csv')), None)
                if csv_file:
                    csv_path = os.path.join(path, csv_file)
                    print(f"Found CSV file: {csv_path}")
                else:
                    csv_path = os.path.join(path, 'creditcard.csv')
                    if not os.path.exists(csv_path):
                        raise FileNotFoundError(f"Could not find CSV file in {path}")
                    
                # Load the data
                self.df = pd.read_csv(csv_path)
                print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            except Exception as e:
                print(f"Error downloading or loading data: {e}")
                if self.file_path:
                    self.df = pd.read_csv(self.file_path)
                else:
                    raise Exception("Could not download data and no file_path provided")
        else:
            if self.file_path is None:
                raise ValueError("file_path must be provided when download_data=False")
            self.df = pd.read_csv(self.file_path)
            
        return self.df
    
    def display_info(self):
        """
        Display basic information about the dataset.
        """
        if self.df is None:
            self.load_data()
            
        print('Dataset shape:', self.df.shape)
        print('No Frauds:', round(self.df['Class'].value_counts()[0]/len(self.df) * 100, 2), '% of the dataset')
        print('Frauds:', round(self.df['Class'].value_counts()[1]/len(self.df) * 100, 2), '% of the dataset')
        print('Missing values:', self.df.isnull().sum().max())
        
    def scale_features(self):
        """
        Scale the Amount and Time features using RobustScaler.
        
        Returns:
        --------
        pd.DataFrame
            The dataframe with scaled features
        """
        if self.df is None:
            self.load_data()
            
        rob_scaler = RobustScaler()
        self.df['scaled_amount'] = rob_scaler.fit_transform(self.df['Amount'].values.reshape(-1,1))
        self.df['scaled_time'] = rob_scaler.fit_transform(self.df['Time'].values.reshape(-1,1))
        self.df.drop(['Time','Amount'], axis=1, inplace=True)
        
        # Rearrange columns
        scaled_amount = self.df['scaled_amount']
        scaled_time = self.df['scaled_time']
        self.df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        self.df.insert(0, 'scaled_amount', scaled_amount)
        self.df.insert(1, 'scaled_time', scaled_time)
        
        return self.df
    
    def split_data(self):
        """
        Prepare the full preprocessed dataset for cross-validation.
        
        Returns:
        --------
        X, y : numpy.ndarray
            Full features and labels for cross-validation
        """
        if self.df is None:
            self.scale_features()
        
        # Prepare data for modeling
        self.X = self.df.drop('Class', axis=1).values
        self.y = self.df['Class'].values
        
        print("Data preparation completed.")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
        
        return self.X, self.y
    
    def preprocess(self):
        """
        Run the full preprocessing pipeline.
        
        Returns:
        --------
        X, y : numpy.ndarray
            Preprocessed features and labels ready for cross-validation
        """
        self.load_data()
        self.display_info()
        self.scale_features()
        return self.split_data()
