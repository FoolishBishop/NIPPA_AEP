from utils import CosmicModule, NonLinearConstrastiveLearning, LinearConstrastiveLearning, CosmicRayDataset
from lightorch.htuning.optuna import htuning
import optuna
from typing import Dict, Any

def objective(trial: optuna.trial.Trial) -> Dict[str, Any]:
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log = True)
    optimizer = trial.suggest_categorical('optimizere', ['rms', 'adam', 'sgd'])

    return dict(
        lr = lr,
        optimizer = optimizer,

    )

if __name__ == '__main__':
    htuning(
        model_class = LinearConstrastiveLearning,
        hparam_objective=objective,
        datamodule=CosmicModule,
        valid_metrics='Training/Pearson',
        directions = ['minimize'],
        n_trials = 150,
    )