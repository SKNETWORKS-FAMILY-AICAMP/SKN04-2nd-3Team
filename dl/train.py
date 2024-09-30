from src.data import LeaveDataset, LeaveDataModule
from src.utils import convert_category_into_integer
from src.model.mlp import Model
from src.training import LeaveModule

import torch

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import random
import json
import nni

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def main(configs):
    # 'train.csv' 데이터를 불러옴
    train = pd.read_csv('../data/train.csv')

    # 결측값이 있는 행 제거
    train = train.dropna()

    # 사용할 열만 선택하여 필터링
    train = train.filter(items=['Churn','CustomerCareCalls', 'DroppedCalls', 'MonthsInService','OverageMinutes'])
    
    # OverageMinutes 값이 50 미만인 데이터만 선택 (이상치 제거 등의 목적)
    train = train.query('OverageMinutes < 50')

    # 범주형 데이터를 정수형으로 변환 (여기서는 'Churn' 열만 변환)
    categorical_columns = ['Churn']
    train, _ = convert_category_into_integer(train, categorical_columns)

    # 데이터 타입을 float32로 변환 (학습을 위한 데이터 포맷 통일)
    train = train.astype(np.float32)

    # 데이터셋을 학습용과 임시 데이터셋으로 6:4 비율로 분할
    train, temp = train_test_split(train, test_size=0.4, random_state=seed)
    
    # 임시 데이터를 다시 검증용과 테스트용으로 5:5 비율로 분할
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

    # 학습 데이터에서 레이블 추출 ('Churn' 열)
    train_labels = train['Churn'].astype(int)

    # 클래스 별 데이터 개수를 계산 (각 클래스의 빈도 수)
    class_counts = np.bincount(train_labels)
    
    # 클래스가 두 개 있는지 확인하는 assert 구문
    assert len(class_counts) == 2

    # 클래스 가중치를 계산 (빈도가 낮은 클래스에 더 높은 가중치를 부여)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # 가중치의 합이 1이 되도록 정규화

    # 가중치를 텐서로 변환하여 GPU 또는 CPU로 이동
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights.to(device)
    
    # 수치형 데이터를 표준화 (평균 0, 표준편차 1로 변환)
    standard_scaler = StandardScaler()
    numeric_columns = ['CustomerCareCalls', 'DroppedCalls', 'MonthsInService','OverageMinutes']

    # 학습, 검증, 테스트 데이터셋에 대해 동일하게 표준화 적용
    train.loc[:, numeric_columns] = standard_scaler.fit_transform(train.loc[:, numeric_columns])
    valid.loc[:, numeric_columns] = standard_scaler.transform(valid.loc[:, numeric_columns])
    test.loc[:, numeric_columns] = standard_scaler.transform(test.loc[:, numeric_columns])

    # LeaveDataset 클래스를 이용해 학습, 검증, 테스트 데이터를 PyTorch 데이터셋으로 변환
    train_dataset = LeaveDataset(train)
    valid_dataset = LeaveDataset(valid)
    test_dataset = LeaveDataset(test)

    # 데이터 모듈 생성 및 데이터셋 준비 (데이터로더를 생성하는 역할)
    leave_data_module = LeaveDataModule(batch_size=configs.get('batch_size'))
    leave_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    # 모델 입력 차원을 설정하고 모델을 생성
    configs.update({'input_dim': len(train.columns) - 1})
    model = Model(configs)

    # LeaveModule 생성 (모델, 설정 정보, 클래스 가중치를 포함한 모듈)
    leave_module = LeaveModule(
        model=model,
        configs=configs,
        class_weights=class_weights  # 클래스 가중치를 전달 (불균형 문제 해결을 위해)
    )
    
    # 설정 파일에서 output_dim과 seed 값을 제거
    del configs['output_dim'], configs['seed']
    
    # 실험 이름을 설정 (config 값을 기반으로)
    exp_name = ','.join([f'{key}={value}' for key, value in configs.items()])
    
    # Trainer 인스턴스 생성 (학습 설정)
    trainer_args = {
        'max_epochs': configs.get('epochs'),  # 최대 에포크 수
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=5)  # 조기 종료 설정 (검증 손실을 기준으로)
        ],
        'logger': TensorBoardLogger(
            'tensorboard',  # 로그를 저장할 디렉터리
            f'leave/please',  # 실험 이름
        ),
    }

    # GPU 사용 여부에 따라 설정을 추가
    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    # Trainer 인스턴스 생성 (학습 및 평가를 관리)
    trainer = Trainer(**trainer_args)

    # 학습 시작 (fit 함수로 모델 학습)
    trainer.fit(
        model=leave_module,
        datamodule=leave_data_module,
    )
    
    # 테스트 실행 (fit 후 테스트 데이터셋으로 모델 평가)
    trainer.test(
        model=leave_module,
        datamodule=leave_data_module,
    )

if __name__ == '__main__':
    # 사용할 디바이스 결정 (GPU가 있을 경우 'gpu', 없으면 'cpu')
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # 설정 파일 (configs.json) 로드
    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    # seed 값을 설정 (랜덤성 제어를 위해)
    seed = configs.get('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # NNI를 사용하는 경우 추가적인 하이퍼파라미터 설정 업데이트
    if configs.get('nni'):
        nni_params = nni.get_next_parameter()
        configs.update(nni_params)

    # GPU 사용 시 추가적인 설정
    if device == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # 메인 함수 실행
    main(configs)
