###
# File: /DITracker/dataset/adult_module.py
# Created Date: Monday, September 23rd 2024
# Author: Zihan
# -----
# Last Modified: Monday, 23rd September 2024 9:55:53 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###
# DITracker/dataset/adult_module.py
import os
import pandas as pd
import numpy as np
import sys

# 动态添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.base_module import DataModule  # 导入 core 模块


class AdultModule(DataModule):
    def load_data(self):
        self.logger.info("Loading Adult data")
        train_file = os.path.join(self.data_dir, "adult-training.csv")
        test_file = os.path.join(self.data_dir, "adult-test.csv")

        columns = [
            "Age",
            "Workclass",
            "fnlgwt",
            "Education",
            "Education-Num",
            "Marital Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Capital Gain",
            "Capital Loss",
            "Hours/Week",
            "Native Country",
            "Income",
        ]

        # Load data into Pandas DataFrame
        train_data = pd.read_csv(train_file, names=columns)
        test_data = pd.read_csv(test_file, names=columns, skiprows=1)

        # Combine train and test data
        df = pd.concat([train_data, test_data], ignore_index=True)
        df["Income"] = df["Income"].apply(lambda x: 1 if ">50K" in x else 0)

        # Preprocess categorical features
        x = pd.get_dummies(df.drop(columns=["Income"]), drop_first=True).values
        y = df["Income"].values

        return x, y
