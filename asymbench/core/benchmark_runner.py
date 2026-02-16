import itertools
import json
from pathlib import Path

from asymbench.core.experiment import Experiment
from asymbench.data.loader import load_dataset
from asymbench.data.splitter import MoleculeSplitter
from asymbench.optimization.base import get_optimizer
from asymbench.preprocessing.feature_preprocessor import FeaturePreprocessor
from asymbench.preprocessing.targets_scaler import TargetScaler
from asymbench.representations.base import get_representation


class BenchmarkRunner:
    def __init__(self, config):
        self.config = config

        # Load dataset once
        self.dataset = load_dataset(config["dataset"])

    def run(self):
        results = []

        combos = itertools.product(
            self.config["representations"],
            self.config["models"],
            self.config["splits"],
            self.config["train_set_sizes"],
            self.config["split_by_mol_col"],
            self.config["seeds"],
        )
        pre_cfg = self.config["preprocessing"]
        y_scl_cfg = self.config["target_scaling"]

        for (
            rep_cfg,
            model_cfg,
            split_cfg,
            train_set_size,
            split_by_mol_col,
            seed,
        ) in combos:

            split_cfg["train_size"] = train_set_size

            exp = self._build_experiment(
                rep_cfg=rep_cfg,
                pre_cfg=pre_cfg,
                y_scl_cfg=y_scl_cfg,
                model_cfg=model_cfg,
                split_cfg=split_cfg,
                split_by_mol_col=split_by_mol_col,
                seed=seed,
            )
            metrics = exp.run()

            result = {
                "representation": rep_cfg["type"],
                "model": model_cfg["type"],
                "split": split_cfg["sampler"],
                "seed": seed,
                **metrics,
            }

            results.append(result)

        self._save_results(results)
        return results

    def _build_experiment(
        self,
        rep_cfg,
        pre_cfg,
        y_scl_cfg,
        model_cfg,
        split_cfg,
        split_by_mol_col,
        seed,
    ):
        # Build representation config
        rep_config = {
            "representation": rep_cfg,
            "data": self.config["dataset"],
        }

        representation = get_representation(rep_config)

        preprocessing = FeaturePreprocessor(pre_cfg)
        y_scaling = TargetScaler(y_scl_cfg["scaling"])

        # Build model
        optimizer = get_optimizer(model_cfg, seed)

        split_strategy = MoleculeSplitter(split_cfg)

        return Experiment(
            dataset=self.dataset,
            smiles_columns=self.config["dataset"]["smiles_columns"],
            target=self.config["dataset"]["target"],
            split_by_mol_col=split_by_mol_col,
            representation=representation,
            preprocessing=preprocessing,
            y_scaling=y_scaling,
            optimizer=optimizer,
            split_strategy=split_strategy,
            seed=seed,
            cache_dir=self.config["log_dirs"]["runs"],
        )

    def _save_results(self, results):
        log_dir = self.config["log_dirs"]["benchmark"]
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{log_dir}/raw_results.json", "w") as f:
            json.dump(results, f, indent=2)
