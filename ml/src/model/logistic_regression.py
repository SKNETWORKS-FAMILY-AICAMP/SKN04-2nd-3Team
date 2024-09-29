from sklearn.linear_model import LogisticRegression
import pandas as pd


def logit(target: str, train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0, compare_columns: list = []):
    # string = f'{target} ~ {" + ".join(compare_columns)}'
    # 로지스틱 회귀 모델 정의
    model = LogisticRegression(random_state=seed).fit(train.drop(columns=[target]), train[target])
    # 모델 예측 값
    print("logistic:", model.score(valid.drop(columns=[target]), valid[target]), sep='\t')