import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.preprocessing import StandardScaler


class DataHolder:

    def __init__(self, data_path):
        assert os.path.exists(data_path), f"File path {data_path} does not exist."
        self.data_path = data_path

        file_path = os.path.join(data_path, "X_train_df.pkl")
        assert os.path.exists(file_path), f"X_train_df DataFrame was not found for path {file_path}"
        self.X_train_df = pd.read_pickle(file_path)

        file_path = os.path.join(data_path, "y_train_df.pkl")
        assert os.path.exists(file_path), f"y_train_df DataFrame was not found for path {file_path}"
        self.y_train_df = pd.read_pickle(file_path)
        self.y_train = self.y_train_df.to_numpy().ravel()

        file_path = os.path.join(data_path, "X_test_df.pkl")
        assert os.path.exists(file_path), f"X_test_df was not found for path {file_path}"
        self.X_test_df = pd.read_pickle(file_path)

        file_path = os.path.join(data_path, "y_test_df.pkl")
        assert os.path.exists(file_path), f"y_test_df DataFrame was not found for path {file_path}"
        self.y_test_df = pd.read_pickle(file_path)
        self.y_test = self.y_test_df.to_numpy().ravel()

        self.__is_scaled__ = False
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def get_data_as_df(self):
        return self.X_train_df, self.y_train_df, self.X_test_df, self.y_test_df

    def get_data_as_numpy(self):
        X_train = self.X_train_df.to_numpy()
        y_train = self.y_train
        X_test = self.X_test_df.to_numpy()
        y_test = self.y_test
        return X_train, y_train, X_test, y_test

    def get_data_as_numpy_scaled(self):
        assert self.scaler is not None, f"No scaler in initialized object. Call 'scale_data' before."
        if self.__is_scaled__:
            return self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test
        else:
            logging.warning("Data was not yet scaled. Call scale_data before.")

    def scale_data(self, scaler=StandardScaler()):
        X_train, y_train, X_test, y_test = self.get_data_as_numpy()
        scaler.fit(X_train)
        self.scaler = scaler
        self.X_train_scaled = self.scaler.transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.__is_scaled__ = True

    # PRINT METHODS
    def print_data_shapes(self):
        print(f"Shape of X_train_df: {self.X_train_df.shape}")
        print(f"Shape of y_train_df: {self.y_train_df.shape}")
        print(f"Shape of X_test_df: {self.X_test_df.shape}")
        print(f"Shape of y_test_df: {self.y_test_df.shape}")

    def print_data_stats(self):
        print(
            f"X_train_df: σ = {np.std(self.X_train_df.to_numpy()):.6f}, μ = {np.mean(self.X_train_df.to_numpy()):.6f}")
        print(f"X_test_df: σ = {np.std(self.X_test_df.to_numpy()):.6f}, μ = {np.mean(self.X_test_df.to_numpy()):.6f}")
        if self.__is_scaled__:
            print(f"X_train_scaled: σ = {np.std(self.X_train_scaled):.6f}, μ = {np.mean(self.X_train_scaled):.6f}")
            print(f"X_test_scaled: σ = {np.std(self.X_test_scaled):.6f}, μ = {np.mean(self.X_test_scaled):.6f}")
        else:
            logging.warning("Data is not yet scaled. No prints for scaled data available!")
