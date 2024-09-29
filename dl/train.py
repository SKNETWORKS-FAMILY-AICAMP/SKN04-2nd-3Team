from src.data import LeaveDataset, LeaveDataModule
from src.utils import convert_category_into_integer
from src.model.mlp import Model
from src.training import LeaveModule

import numpy as np
import random
import json
import nni
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import seaborn as sns
import pandas as pd


def main(configs):
    # 'train' 데이터셋을 로드
    train = pd.read_csv('./data/train.csv')
    # test = pd.read_csv('./data/test.csv')

    # 결측값이 있는 모든 행 제거
    train = train.dropna()
    train = train.drop(columns=['CustomerID'])
    # test = test.drop(columns=['CustomerID'])
    # train['Churn'] = np.where(train['Churn'] == "Yes", 1, 0)
    
    # 범주형 열을 정수형으로 변환
    categorical_columns = ['Churn','ServiceArea', 'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner', 'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings', 'NonUSTravel', 'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser', 'NotNewCellphoneUser', 'OwnsMotorcycle', 'HandsetPrice', 'MadeCallToRetentionTeam',  'CreditRating', 'PrizmCode', 'Occupation', 'MaritalStatus']

    # train, _ = convert_category_into_integer(train, ( 'Churn','ServiceArea', 'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner', 'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings', 'NonUSTravel',  'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser', 'NotNewCellphoneUser', 'OwnsMotorcycle', 'HandsetPrice', 'MadeCallToRetentionTeam', 'CreditRating', 'PrizmCode', 'Occupation', 'MaritalStatus'))
    train, _ = convert_category_into_integer(train, categorical_columns)
    # test, _ = convert_category_into_integer(test, ( 'ServiceArea', 'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner', 'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings', 'NonUSTravel',  'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser', 'NotNewCellphoneUser', 'OwnsMotorcycle', 'HandsetPrice', 'MadeCallToRetentionTeam', 'CreditRating', 'PrizmCode', 'Occupation', 'MaritalStatus'))

    # 데이터프레임을 float32로 변환
    train = train.astype(np.float32)
    # test = test.astype(np.float32)

    # 데이터셋을 학습용과 임시 데이터로 분할
    train, temp = train_test_split(train, test_size=0.4, random_state=seed)
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

    
    standard_scaler = StandardScaler()
    numeric_columns = ['ActiveSubs', 'AdjustmentsToCreditRating', 'AgeHH1', 'AgeHH2','BlockedCalls', 'CallForwardingCalls', 'CallWaitingCalls','CurrentEquipmentDays', 'CustomerCareCalls', 'DirectorAssistedCalls','DroppedBlockedCalls', 'DroppedCalls', 'HandsetModels', 'Handsets','InboundCalls', 'IncomeGroup', 'MonthlyMinutes', 'MonthlyRevenue','MonthsInService', 'OffPeakCallsInOut', 'OutboundCalls','OverageMinutes', 'PeakCallsInOut', 'PercChangeMinutes','PercChangeRevenues', 'ReceivedCalls', 'ReferralsMadeBySubscriber','RetentionCalls', 'RetentionOffersAccepted', 'RoamingCalls','ThreewayCalls', 'TotalRecurringCharge', 'UnansweredCalls','UniqueSubs']
    # numeric_columns = [col for col in categorical_columns if train[col].dtype != 'object']
    # numeric_columns = train.columns.diffrence(categorical_columns)
    # numeric = list(train.columns.difference(categorical_columns))
    # train[numeric_columns] = standard_scaler.fit_transform(train[numeric_columns])
    # valid[numeric_columns] = standard_scaler.transform(valid[numeric_columns])
    # test[numeric_columns] = standard_scaler.transform(test[numeric_columns])

    train.loc[:, numeric_columns] = standard_scaler.fit_transform(train.loc[:,numeric_columns])
    valid.loc[:, numeric_columns] = standard_scaler.transform(valid.loc[:,numeric_columns])
    test.loc[:, numeric_columns] = standard_scaler.transform(test.loc[:,numeric_columns])

    # 데이터셋 객체로 변환
    train_dataset = LeaveDataset(train)
    valid_dataset = LeaveDataset(valid)
    test_dataset = LeaveDataset(test)

    # 데이터 모듈 생성 및 데이터 준비
    leave_data_module = LeaveDataModule(batch_size=configs.get('batch_size'))
    leave_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    # 모델 생성
    configs.update({'input_dim': len(train.columns)-1})
    model = Model(configs)

    # LightningModule 인스턴스 생성
    leave_module = LeaveModule(
        model=model,
        configs=configs,
    )
    
    # Trainer 인스턴스 생성 및 설정
    del configs['output_dim'], configs['seed']
    exp_name = ','.join([f'{key}={value}' for key, value in configs.items()])
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=5)
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'leave/please',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=leave_module,
        datamodule=leave_data_module,
    )
    trainer.test(
        model=leave_module,
        datamodule=leave_data_module,
    )

    # trainer.save_checkpoint('model_checkpoint.ckpt')
    torch.save(leave_module.state_dict(), 'model_weights.pth')

if __name__ == '__main__':
    # 사용 가능한 GPU가 있는 경우 'cuda', 그렇지 않으면 'cpu' 사용
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # hyperparameter
    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    # seed 설정
    seed = configs.get('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if configs.get('nni'):
        nni_params = nni.get_next_parameter()
        configs.update(nni_params)
        # configs.update({'batch_size': nni_params.get('batch_size')})
        # configs.update({'hidden_dim': nni_params.get('hidden_dim')})
        # configs.update({'learning_rate': nni_params.get('learning_rate')})
        # configs.update({'dropout_ratio': nni_params.get('dropout_ratio')})

    # CUDA 설정
    if device == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    main(configs)