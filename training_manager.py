###
# File: /DITracker/training_manager.py
# Created Date: Thursday, September 26th 2024
# Author: Zihan
# -----
# Last Modified: Thursday, 26th September 2024 12:34:20 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from models.model_factory import get_model

# from dataset.data_factory import DataModuleFactory
from utils.logging_utils import setup_logging
import pandas as pd
import toml
from sklearn.linear_model import LogisticRegressionCV
from data_loader import DataLoader


class TrainingManager:
    def __init__(
        self,
        key: str,
        model_type: str,
        seed: int,
        gpu: int,
        save_dir: str,
        n_tr: int = None,
        n_val: int = None,
        n_test: int = None,
        num_epoch: int = None,
        batch_size: int = None,
        lr: float = None,
        compute_counterfactual: bool = True,
        relabel_percentage: float = None,
        logger=None,
    ):
        self.key = key
        self.model_type = model_type
        self.seed = seed
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        if save_dir is None:
            save_dir = os.path.join("result", f"{model_type}_{key}")
        self.save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), save_dir
        )

        # Load default configuration
        config = self.load_config()
        self.n_tr = n_tr or config[key]["n_tr"]
        self.n_val = n_val or config[key]["n_val"]
        self.n_test = n_test or config[key]["n_test"]
        self.num_epoch = num_epoch or 21  # Default value if not provided
        self.batch_size = batch_size or 60  # Default value if not provided
        self.lr = lr or 0.01  # Default value if not provided
        self.compute_counterfactual = compute_counterfactual
        self.relabel_percentage = relabel_percentage

        self.logger = logger or setup_logging(
            f"{key}_{model_type}", seed, self.save_dir
        )

        # if mps
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.logger.info("MPS is enabled.")

        os.makedirs(self.save_dir, exist_ok=True)

        self.data_loader = DataLoader(
            key=self.key,
            model_type=self.model_type,
            seed=self.seed,
            device=self.device,
            data_dir=os.path.join(self.save_dir, "data"),
            n_tr=self.n_tr,
            n_val=self.n_val,
            n_test=self.n_test,
            relabel_percentage=self.relabel_percentage,
            logger=self.logger,
        )

    def load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "dataset", "config.toml")
        with open(config_path, "r") as f:
            return toml.load(f)

    def initialize_model(self, input_dim):
        return get_model(self.model_type, input_dim, self.logger).to(self.device)

    def train(self) -> Dict[str, Any]:
        x_tr, y_tr, x_val, y_val, data_sizes = self.data_loader.load_data()

        if self.model_type == "logreg":
            model = LogisticRegressionCV(
                random_state=self.seed, fit_intercept=False, cv=5
            )
            x_tr_np = x_tr.cpu().numpy()
            y_tr_np = y_tr.cpu().numpy().ravel()
            model.fit(x_tr_np, y_tr_np)
            alpha = 1 / (model.C_[0] * data_sizes["n_tr"])
        else:
            alpha = 0.001

        input_dim = x_tr.shape[1:]
        net_func = lambda: self.initialize_model(input_dim)
        num_steps = int(np.ceil(data_sizes["n_tr"] / self.batch_size))

        list_of_sgd_models = []
        list_of_counterfactual_models = (
            [[] for _ in range(data_sizes["n_tr"])]
            if self.compute_counterfactual
            else None
        )
        main_losses = []
        test_accuracies = []
        train_losses = [np.nan]

        for n in range(-1, data_sizes["n_tr"] if self.compute_counterfactual else 0):
            self.logger.info(f"Training model {n+1}/{data_sizes['n_tr']}")
            torch.manual_seed(self.seed)
            model = net_func()
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.0)

            lr_n = self.lr
            skip = [n]
            info = []
            c = 0

            for epoch in range(self.num_epoch):
                epoch_loss = 0.0
                np.random.seed(epoch)
                idx_list = np.array_split(
                    np.random.permutation(data_sizes["n_tr"]), num_steps
                )

                for i in range(num_steps):
                    info.append({"idx": idx_list[i], "lr": lr_n})
                    c += 1

                    m = net_func()
                    m.load_state_dict(model.state_dict())

                    if n < 0:
                        m.to("cpu")
                        list_of_sgd_models.append(m)
                        if c % num_steps == 0 or c == num_steps * self.num_epoch:
                            with torch.no_grad():
                                val_loss = loss_fn(model(x_val), y_val).item()
                                main_losses.append(val_loss)
                                test_pred = (model(x_val) > 0).float()
                                test_acc = (test_pred == y_val).float().mean().item()
                                test_accuracies.append(test_acc)
                    elif self.compute_counterfactual:
                        if c % num_steps == 0 or c == num_steps * self.num_epoch:
                            m.to("cpu")
                            list_of_counterfactual_models[n].append(m)

                    idx = idx_list[i]
                    b = idx.size
                    idx = np.setdiff1d(idx, skip)
                    z = model(x_tr[idx])
                    loss = loss_fn(z, y_tr[idx])

                    if (
                        c % num_steps == 0 or c == num_steps * self.num_epoch
                    ) and n < 0:
                        train_losses.append(loss.item())

                    epoch_loss += loss.item()

                    for p in model.parameters():
                        loss += 0.5 * alpha * (p * p).sum()

                    optimizer.zero_grad()
                    loss.backward()

                    for p in model.parameters():
                        p.grad.data *= idx.size / b

                    optimizer.step()

                    if self.lr > 0:
                        lr_n *= np.sqrt(c / (c + 1))
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_n

            if n < 0:
                m = net_func()
                m.load_state_dict(model.state_dict())
                m.to("cpu")
                list_of_sgd_models.append(m)
                with torch.no_grad():
                    val_loss = loss_fn(model(x_val), y_val).item()
                    main_losses.append(val_loss)
                    test_pred = (model(x_val) > 0).float()
                    test_acc = (test_pred == y_val).float().mean().item()
                    test_accuracies.append(test_acc)
            elif self.compute_counterfactual:
                m = net_func()
                m.load_state_dict(model.state_dict())
                m.to("cpu")
                list_of_counterfactual_models[n].append(m)

        results = {
            "models": list_of_sgd_models,
            "info": info,
            "counterfactual": list_of_counterfactual_models,
            "alpha": alpha,
            "main_losses": main_losses,
            "test_accuracies": test_accuracies,
            "train_losses": train_losses,
            "seed": self.seed,
            "n_tr": data_sizes["n_tr"],
            "n_val": data_sizes["n_val"],
            "n_test": data_sizes["n_test"],
            "num_epoch": self.num_epoch,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }

        self.save_results(results)
        return results

    def save_results(self, results):
        fn = os.path.join(self.save_dir, f"sgd{self.seed:03d}.dat")
        torch.save(results, fn)

        step_fn_csv = os.path.join(self.save_dir, f"step_{self.seed:03d}.csv")
        pd.DataFrame(
            {
                "step": range(len(results["info"])),
                "lr": [d["lr"] for d in results["info"]],
                "idx": [d["idx"] for d in results["info"]],
            }
        ).to_csv(step_fn_csv, index=False)

        csv_fn = os.path.join(self.save_dir, f"metrics_{self.seed:03d}.csv")
        pd.DataFrame(
            {
                "epoch": range(len(results["main_losses"])),
                "main_loss": results["main_losses"],
                "test_accuracy": results["test_accuracies"],
                "train_loss": results["train_losses"],
            }
        ).to_csv(csv_fn, index=False)

        self.logger.debug(f"Training completed. Results saved to {fn} and {csv_fn}")


import argparse


def main():
    parser = argparse.ArgumentParser(description="Train Models & Save")
    parser.add_argument("--target", default="adult", type=str, help="target data")
    parser.add_argument("--model", default="logreg", type=str, help="model type")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="gpu index")
    parser.add_argument("--save_dir", type=str, help="directory to save results")
    parser.add_argument(
        "--no-loo",
        action="store_false",
        dest="compute_counterfactual",
        help="Disable the computation of counterfactual models (leave-one-out)",
    )
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument(
        "--relabel", type=float, help="percentage of training data to relabel"
    )
    # Add other arguments as needed

    args = parser.parse_args()

    trainer = TrainingManager(
        key=args.target,
        model_type=args.model,
        seed=args.seed,
        gpu=args.gpu,
        save_dir=args.save_dir,
        lr=args.lr,
        compute_counterfactual=args.compute_counterfactual,
        relabel_percentage=args.relabel,
    )

    results = trainer.train()
    print("Training completed.")


if __name__ == "__main__":
    main()
