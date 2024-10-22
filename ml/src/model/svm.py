from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np


## SVC 하이퍼 파라미터
param_grid = {
    'C': np.arange(0.0, 1, 0.01),       # C 값 후보
    'gamma': [1, 0.1, 0.01, 0.001],  # gamma 값 후보
    'coef0': np.arange(0.0, 5.1, 0.1),
}

def svc_tuning(target: str, train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0, sample_size: int = 300):
    if sample_size > 0:
        train = train.sample(n=sample_size, random_state=seed)
        valid = valid.sample(n=sample_size, random_state=seed)
    # svc = SVC(random_state=seed, kernel='rbf', gamma="scale")
    for kernel in ('poly', 'rbf'):
        svc = SVC(kernel=kernel, random_state=seed)
        random_search = RandomizedSearchCV(svc, param_grid, refit=True, cv=5, verbose=0, random_state=seed)
        random_search.fit(train.drop(columns=target), train[target])
        print(f"svc kernel({kernel}) best score:", random_search.score(valid.drop(columns=target), valid[target]))  
        result = random_search.cv_results_.get('params')[random_search.cv_results_.get('rank_test_score').argmin()]
        print(f"svc kernel({kernel}) best hyperparameter:", result)
        print()
