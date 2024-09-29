from sklearn.preprocessing import LabelEncoder
import pandas as pd


## 범주형 변수 인코딩하기
def astype_to_category(dfs: list):
    dfs.Churn = dfs.Churn.astype('int')
    # dfs.ServiceArea = dfs.ServiceArea.astype('category')
    # dfs.ChildrenInHH = dfs.ChildrenInHH.astype('category')
    # dfs.HandsetRefurbished = dfs.HandsetRefurbished.astype('category')
    # dfs.HandsetWebCapable = dfs.HandsetWebCapable.astype('category')
    # dfs.TruckOwner = dfs.TruckOwner.astype('category')
    # dfs.RVOwner = dfs.RVOwner.astype('category')
    # dfs.Homeownership = dfs.Homeownership.astype('category')
    # dfs.BuysViaMailOrder = dfs.BuysViaMailOrder.astype('category')
    # dfs.RespondsToMailOffers = dfs.RespondsToMailOffers.astype('category')
    # dfs.OptOutMailings = dfs.OptOutMailings.astype('category')
    # dfs.NonUSTravel = dfs.NonUSTravel.astype('category')
    # dfs.OwnsComputer = dfs.OwnsComputer.astype('category')
    # dfs.NewCellphoneUser = dfs.NewCellphoneUser.astype('category')
    # dfs.NotNewCellphoneUser = dfs.NotNewCellphoneUser.astype('category')
    # dfs.OwnsMotorcycle = dfs.OwnsMotorcycle.astype('category')
    # dfs.HandsetPrice = dfs.HandsetPrice.astype('category')
    # dfs.MadeCallToRetentionTeam = dfs.MadeCallToRetentionTeam.astype('category')
    # dfs.CreditRating = dfs.CreditRating.astype('category')
    # dfs.PrizmCode = dfs.PrizmCode.astype('category')
    # dfs.Occupation = dfs.Occupation.astype('category')
    # dfs.MaritalStatus = dfs.MaritalStatus.astype('category')

    return dfs

## 데이터 종류가 object인 칼럼을 카테고리화 하기
def convert_category_into_integer(df: pd.DataFrame, columns: list):
    label_encoders = {}
    for column in columns:
        label_encoder = LabelEncoder()
        df.loc[:, column] = label_encoder.fit_transform(df[column])

        label_encoders.update({column: label_encoder})
    
    return df, label_encoders