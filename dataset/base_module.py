###
# File: /DITracker/dataset/base_module.py
# Created Date: Monday, September 23rd 2024
# Author: Zihan
# -----
# Last Modified: Monday, 23rd September 2024 11:08:27 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###


import abc
import os
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from filelock import FileLock
import pickle
import random


class DataModule(abc.ABC):
    def __init__(
        self, normalize=True, append_one=False, data_dir=None, logger=None, seed=0
    ):
        self.normalize = normalize
        self.append_one = append_one
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../data"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)

    @abc.abstractmethod
    def load_data(self):
        """Load and return the raw dataset (x, y). Should be implemented by each subclass."""
        pass

    def preprocess(self, x, y):
        """Preprocess the data if needed (e.g., normalization, append one column)."""
        if self.normalize:
            self.logger.info("Normalizing data")
            if x.ndim > 2:  # For image data, normalize per channel
                x = (x - np.mean(x, axis=(1, 2, 3), keepdims=True)) / np.std(
                    x, axis=(1, 2, 3), keepdims=True
                )
            else:
                x = (x - np.mean(x)) / np.std(x)

        if self.append_one and x.ndim == 2:
            self.logger.info("Appending ones to data")
            x = np.c_[x, np.ones(x.shape[0])]

        return x, y

    def fetch(self, n_tr, n_val, n_test):
        # Generate a unique cache file name that includes the data shape
        cache_file = os.path.join(
            self.data_dir,
            f"{self.__class__.__name__}_tr{n_tr}_val{n_val}_test{n_test}_seed{self.seed}.pkl",
        )
        lock_file = f"{cache_file}.lock"
        self.logger.info(
            f"Fetching data with seed {self.seed}, n_tr={n_tr}, n_val={n_val}, n_test={n_test}"
        )

        with FileLock(lock_file):
            return self._load_data_cahced_or_not(cache_file, n_tr, n_val, n_test)

    # TODO Rename this here and in `fetch`
    def _load_data_cahced_or_not(self, cache_file, n_tr, n_val, n_test):
        if os.path.exists(cache_file):
            self.logger.info(f"Loading data from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        self.logger.info("Cache not found. Loading and processing raw data.")
        x, y = self.load_data()

        np.random.seed(self.seed)
        random.seed(self.seed)

        # Split data
        x_tr, x_temp, y_tr, y_temp = train_test_split(
            x, y, train_size=n_tr, test_size=n_val + n_test, random_state=self.seed
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            train_size=n_val,
            test_size=n_test,
            random_state=self.seed + 100,
        )

        # Preprocess the data
        x_tr, y_tr = self.preprocess(x_tr, y_tr)
        x_val, y_val = self.preprocess(x_val, y_val)
        x_test, y_test = self.preprocess(x_test, y_test)

        result = ((x_tr, y_tr), (x_val, y_val), (x_test, y_test))

        # Cache the processed data with shape information
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        self.logger.info(f"Data processed and saved to cache: {cache_file}")

        return result
