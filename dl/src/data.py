import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L


class LeaveDataset(Dataset):
    def __init__(self, data):  # 생성자 메서드
        super().__init__()  # 부모 클래스의 생성자를 호출하여 초기화
        self.data = data  # 데이터프레임을 저장

    def __len__(self):
        # 데이터셋의 전체 샘플 수를 반환
        return len(self.data)

    def __getitem__(self, idx):
        # 인덱스 `idx`에 해당하는 데이터 샘플을 반환
        
        # 데이터프레임에서 'Churn' 열을 제외한 특성값을 가져와서 NumPy 배열로 변환한 뒤, PyTorch 텐서로 변환
        X = torch.from_numpy(self.data.iloc[idx].drop('Churn').values).float()
        
        # 'Churn' 열의 값을 텐서로 변환하여 레이블을 생성
        y = torch.Tensor([self.data.iloc[idx].Churn]).float()
        
        # 입력 데이터와 레이블을 딕셔너리 형태로 반환
        return {
            'X': X,
            'y': y,
        }


class LeaveDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size  # 배치 크기를 저장

    def prepare(self, train_dataset, valid_dataset, test_dataset):
        # 데이터셋을 저장하는 메서드
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def setup(self, stage: str):
        # 데이터셋을 로드하여 각 단계에 맞게 데이터를 설정하는 메서드
        if stage == "fit":
            # 학습과 검증 단계에 사용할 데이터셋 설정
            self.train_data = self.train_dataset
            self.valid_data = self.valid_dataset

        if stage == "test":
            # 테스트 단계에 사용할 데이터셋 설정
            self.test_data = self.test_dataset

    def train_dataloader(self):
        # 학습 데이터 로더 반환
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,  # 배치 크기 설정
            shuffle=True,  # 데이터셋을 섞어서 로드
        )

    def val_dataloader(self):
        # 검증 데이터 로더 반환
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,  # 배치 크기 설정
            shuffle=False,  # 데이터셋을 섞지 않음
        )

    def test_dataloader(self):
        # 테스트 데이터 로더 반환
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,  # 배치 크기 설정
            shuffle=False,  # 데이터셋을 섞지 않음
        )




