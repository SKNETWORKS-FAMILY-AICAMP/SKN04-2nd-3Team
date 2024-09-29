from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
import pandas as pd
import numpy as np


## SVC
# 이상치 처리에 민감하지 않음
# 복잡한 데이터셋 처리 가능

param_grid = {
    'C': np.arange(0.0, 1, 0.01),       # C 값 후보
    # 'gamma': [1, 0.1, 0.01, 0.001],  # gamma 값 후보
    'coef0': np.arange(0.0, 5.1, 0.1),
}

def svc(target: str, train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0, sample_size: int = 300):
    if sample_size > 0:
        train = train.sample(n=sample_size, random_state=seed)
        valid = valid.sample(n=sample_size, random_state=seed)
    svc = SVC(random_state=seed, kernel='rbf', C=0.01)
    svc.fit(train.drop(columns=target), train[target])
    acc = svc.score(valid.drop(columns=target), valid[target])
    # print(f"svc {kernel}", f"{acc: .3f}")
    print(f"svc rbf", f"{acc: .3f}")

def svc_tuned(target: str, train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0, sample_size: int = 300):
    if sample_size > 0:
        train = train.sample(n=sample_size, random_state=seed)
        valid = valid.sample(n=sample_size, random_state=seed)
    svc = SVC(random_state=seed, kernel='rbf', C=0.53, gamma="scale", coef0=0.29)
    svc.fit(train.drop(columns=target), train[target])
    acc = svc.score(valid.drop(columns=target), valid[target])
    # print(f"svc {kernel}", f"{acc: .3f}")
    print(f"svc rbf", f"{acc: .3f}")

def svc2(target: str, train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0, sample_size: int = 300):
    if sample_size > 0:
        train = train.sample(n=sample_size, random_state=seed)
        valid = valid.sample(n=sample_size, random_state=seed)
    # for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
    svc = SVC(random_state=seed, kernel='rbf', C=10)
    svc.fit(train.drop(columns=target), train[target])
    acc = svc.score(valid.drop(columns=target), valid[target])
    # print(f"svc {kernel}", f"{acc: .3f}")
    print(f"svc rbf", f"{acc: .3f}")

def svc_tuning(target: str, train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0, sample_size: int = 300):
    if sample_size > 0:
        train = train.sample(n=sample_size, random_state=seed)
        valid = valid.sample(n=sample_size, random_state=seed)
    svc = SVC(random_state=seed, kernel='rbf', gamma="scale")
    random_search = RandomizedSearchCV(svc, param_grid, refit=True, cv=5, verbose=2, random_state=seed)
    random_search.fit(train.drop(columns=target), train[target])
    print("Best Parameters found by GridSearchCV:")
    print(random_search.cv_results_.get('params')[random_search.cv_results_.get('rank_test_score').argmin()])
    # print(grid_search.best_params_)
    print("svc:", random_search.score(valid.drop(columns=target), valid[target]))    