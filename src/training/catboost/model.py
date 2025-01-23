import os
from typing import Any
from typing import Dict

import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class CatBoostModel:
    def __init__(self):
        self.model = None

    @staticmethod
    def model_from_config(config: Dict[str, Any], seed: int) -> "CatBoostModel":
        instance = CatBoostModel()
        model_config = config["models"]["catboost"]
        occurrence_lower_bound = (
            model_config["occurrence_lower_bound"]
            if not config["is_test"]
            else model_config["occurrence_lower_bound_test"]
        )
        iterations = (
            int(model_config["iterations"])
            if not config["is_test"]
            else int(model_config["iterations_test"])
        )

        instance.model = CatBoostRegressor(
            learning_rate=float(model_config["learning_rate"]),
            bagging_temperature=float(model_config["bagging_temperature"]),
            random_strength=int(model_config["random_strength"]),
            l2_leaf_reg=float(model_config["l2_leaf_reg"]),
            depth=int(model_config["depth"]),
            iterations=iterations,
            early_stopping_rounds=int(model_config["early_stopping_rounds"]),
            loss_function=model_config["loss_function"],
            eval_metric=model_config["eval_metric"],
            random_state=seed,
            verbose=500,
            custom_metric=["MAE", model_config["eval_metric"]],
            tokenizers=[
                {
                    "tokenizer_id": model_config["tokenizer_id"],
                    "separator_type": "ByDelimiter",
                    "split_by_set": "True",
                    "delimiter": "\n -.\t\\s/,:;()[]{}!#$*|+=?`\"'_",
                    "languages": ["ru", "en"],
                    "number_process_policy": "Replace",
                    "number_token": "[NUMBER]",
                }
            ],
            dictionaries=[
                {
                    "dictionary_id": model_config["dictionary_id"],
                    "max_dictionary_size": model_config["max_dictionary_size"],
                    "occurrence_lower_bound": occurrence_lower_bound,
                    "gram_order": model_config["gram_order"],
                }
            ],
            feature_calcers=["BoW"],
        )
        return instance

    @staticmethod
    def model_from_file(model_path: str) -> "CatBoostModel":
        instance = CatBoostModel()
        instance.model = CatBoostRegressor()
        instance.model.load_model(model_path)
        return instance

    def get_params(self):
        return self.model.get_params()

    def train_and_evaluate_with_metrics(
        self,
        train_pool,
        test_pool=None,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        to_eval=True,
    ):
        if to_eval and test_pool is not None:
            self.model.fit(train_pool, eval_set=test_pool)
            y_pred = self.model.predict(X_test)
            return {
                "y_test": y_test,
                "y_pred": y_pred,
                "r2": r2_score(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2_train": self.model.evals_result_["learn"]["R2"],
                "r2_test": self.model.evals_result_["validation"]["R2"],
            }
        else:
            self.model.fit(train_pool)
            y_pred = self.model.predict(X_train)
            return {
                "r2": r2_score(y_train, y_pred),
                "mae": mean_absolute_error(y_train, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_train, y_pred)),
            }

    def save_model(self, config: Dict[str, Any], model_name: str):
        save_dir = config["models"]["catboost"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}.cbm")
        self.model.save_model(save_path)

    def predict(self, X):
        return self.model.predict(X)
