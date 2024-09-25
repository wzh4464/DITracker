###
# File: /DITracker/data_loader.py
# Created Date: Thursday, September 26th 2024
# Author: Zihan
# -----
# Last Modified: Thursday, 26th September 2024 12:31:39 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import numpy as np
import torch
from dataset.data_factory import DataModuleFactory
import pandas as pd
import os


class DataLoader:
    def __init__(
        self,
        key,
        model_type,
        seed,
        device,
        data_dir,
        n_tr,
        n_val,
        n_test,
        relabel_percentage=None,
        logger=None,
    ):
        self.key = key
        self.model_type = model_type
        self.seed = seed
        self.device = device
        self.data_dir = data_dir
        self.n_tr = n_tr
        self.n_val = n_val
        self.n_test = n_test
        self.relabel_percentage = relabel_percentage
        self.logger = logger

        self.data_module = DataModuleFactory.create(
            self.key,
            normalize=True,
            append_one=False,
            data_dir=data_dir,
            logger=logger,
            seed=seed,
        )

    def load_data(self):
        (x_tr, y_tr), (x_val, y_val), _ = self.data_module.fetch(
            self.n_tr, self.n_val, self.n_test
        )

        # Flatten the data for DNN and LogReg
        if self.model_type in ["dnn", "logreg"]:
            x_tr = x_tr.reshape(x_tr.shape[0], -1)
            x_val = x_val.reshape(x_val.shape[0], -1)

        x_tr = torch.from_numpy(x_tr).to(torch.float32).to(self.device)
        y_tr = torch.from_numpy(y_tr).to(torch.float32).unsqueeze(1).to(self.device)
        x_val = torch.from_numpy(x_val).to(torch.float32).to(self.device)
        y_val = torch.from_numpy(y_val).to(torch.float32).unsqueeze(1).to(self.device)

        data_sizes = {"n_tr": self.n_tr, "n_val": self.n_val, "n_test": self.n_test}

        if self.relabel_percentage:
            x_tr, y_tr = self.relabel_data(x_tr, y_tr)

        return x_tr, y_tr, x_val, y_val, data_sizes

    def relabel_data(self, x_tr, y_tr):
        num_to_relabel = int(self.n_tr * self.relabel_percentage / 100)
        relabeled_indices = np.random.choice(self.n_tr, num_to_relabel, replace=False)
        y_tr[relabeled_indices] = 1 - y_tr[relabeled_indices]
        self.save_relabeled_indices(relabeled_indices)
        self.logger.info(
            f"Relabeled {num_to_relabel} samples ({self.relabel_percentage}% of training data)"
        )
        return x_tr, y_tr

    def save_relabeled_indices(self, relabeled_indices):
        relabeled_indices_fn = os.path.join(
            self.data_dir, f"relabeled_indices_{self.seed:03d}.csv"
        )
        pd.DataFrame({"relabeled_indices": relabeled_indices}).to_csv(
            relabeled_indices_fn, index=False
        )
        self.logger.debug(f"Relabeled indices saved to {relabeled_indices_fn}")
