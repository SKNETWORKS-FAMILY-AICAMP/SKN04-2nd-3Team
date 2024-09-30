from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def decision_tree(target: str, train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0):
    tree = DecisionTreeClassifier(random_state=seed)
    tree.fit(train.drop(columns=target), train.Churn)
    tree.score(valid.drop(columns=target), valid.Churn)
    print("decision_tree:", tree.score(valid.drop(columns=target), valid.Churn), sep='\t')