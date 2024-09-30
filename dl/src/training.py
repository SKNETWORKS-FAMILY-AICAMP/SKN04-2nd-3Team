import numpy as np
import nni

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L


class LeaveModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,          # nn.Module을 상속받은 모델
        configs: dict,             # 설정 정보를 담고 있는 딕셔너리
        class_weights: torch.Tensor,  # 클래스 가중치를 나타내는 텐서 (불균형 데이터 해결을 위해 사용)
    ):
        super().__init__()
        self.model = model                     # 모델을 인스턴스 변수로 저장
        self.configs = configs                 # 설정 정보를 인스턴스 변수로 저장
        self.learning_rate = configs.get('learning_rate')  # 학습률을 설정에서 가져옴
        self.class_weights = class_weights     # 클래스 가중치를 설정

        self.val_losses = []  # 검증 손실을 저장하는 리스트
        self.test_losses = []  # 테스트 손실을 저장하는 리스트

    def training_step(self, batch, batch_idx):
        # 학습 단계에서 호출됨. 한 배치(batch)에 대한 계산을 수행
        X = batch.get('X')  # 입력 데이터
        y = batch.get('y').float()  # 레이블을 float 형식으로 변환
        y = y.view(-1)  # 레이블의 차원을 맞춤 (1차원 벡터로 변환)

        output = self.model(X).squeeze()  # 모델의 예측 결과를 구하고 차원을 맞춤
        sample_weights = self.class_weights[y.long()]  # 레이블에 따라 해당하는 가중치를 적용
        sample_weights = sample_weights.to(output.device)  # 가중치를 모델이 있는 장치(GPU/CPU)로 이동

        # 가중치가 적용된 이진 크로스 엔트로피 손실 계산
        self.loss = F.binary_cross_entropy_with_logits(output, y, weight=sample_weights)        
        return self.loss  # 계산된 손실을 반환

    def on_train_epoch_end(self, *args, **kwargs):
        # 한 에포크가 끝난 후 학습 손실을 기록
        self.log_dict(
            {'loss/train_loss': self.loss},  # 학습 손실을 기록
            on_epoch=True,  # 에포크마다 기록
            prog_bar=True,  # 진행 바에 표시
            logger=True,  # 로거에도 기록
        )

    def validation_step(self, batch, batch_idx):
        # 검증 단계에서 호출됨. 배치별 검증 손실 계산
        if batch_idx == 0:
            self.val_losses.clear()  # 새로운 검증 스텝마다 손실을 초기화

        X = batch.get('X')  # 입력 데이터
        y = batch.get('y').float()  # 레이블을 float 형식으로 변환
        y = y.view(-1)  # 레이블의 차원을 맞춤 (1차원 벡터로 변환)
        output = self.model(X).squeeze()  # 모델의 예측 결과를 구하고 차원을 맞춤

        sample_weights = self.class_weights[y.long()]  # 레이블에 따라 해당하는 가중치를 적용
        sample_weights = sample_weights.to(output.device)  # 가중치를 모델이 있는 장치(GPU/CPU)로 이동

        # 가중치가 적용된 이진 크로스 엔트로피 손실 계산
        self.val_loss = F.binary_cross_entropy_with_logits(output, y, weight=sample_weights)
        self.val_losses.append(self.val_loss.detach().item())  # 손실 값을 리스트에 저장

        return self.val_loss  # 검증 손실 반환

    def on_validation_epoch_end(self):
        # 검증 에포크가 끝난 후 검증 손실 및 학습률을 기록
        self.log_dict(
            {'loss/val_loss': self.val_loss,  # 검증 손실을 기록
             'learning_rate': self.optimizers().param_groups[0].get('lr')},  # 학습률 기록
            on_epoch=True,  # 에포크마다 기록
            prog_bar=True,  # 진행 바에 표시
            logger=True,  # 로거에도 기록
        )

        if self.configs.get('nni'):
            nni.report_intermediate_result(np.mean(self.val_losses))  # NNI가 활성화된 경우 중간 결과 보고

    def test_step(self, batch, batch_idx):
        # 테스트 단계에서 호출됨. 배치별 테스트 손실과 예측값 계산
        if batch_idx == 0:
            self.test_preds = []  # 예측값 초기화
            self.test_targets = []  # 실제 레이블 초기화
            self.test_losses.clear()  # 테스트 손실 초기화

        X = batch.get('X')  # 입력 데이터
        y = batch.get('y').float()  # 레이블을 float 형식으로 변환
        y = y.view(-1)  # 레이블의 차원을 맞춤 (1차원 벡터로 변환)

        output = self.model(X).squeeze()  # 모델의 예측 결과를 구하고 차원을 맞춤
        preds = torch.sigmoid(output).round().detach().cpu().numpy()  # 시그모이드 함수로 확률 변환 후 이진화
        self.test_preds.extend(preds.flatten())  # 예측값 저장
        self.test_targets.extend(y.cpu().numpy())  # 실제 레이블 저장

        sample_weights = self.class_weights[y.long()]  # 레이블에 따라 해당하는 가중치를 적용
        sample_weights = sample_weights.to(output.device)  # 가중치를 모델이 있는 장치(GPU/CPU)로 이동

        # 가중치가 적용된 이진 크로스 엔트로피 손실 계산
        test_loss = F.binary_cross_entropy_with_logits(output, y, weight=sample_weights)
        self.test_losses.append(test_loss.detach().item())  # 손실 값을 리스트에 저장
        
        return test_loss  # 테스트 손실 반환

    def on_test_epoch_end(self):
        # 테스트 에포크가 끝난 후 분류 결과 출력
        from sklearn.metrics import classification_report

        # 예측값과 실제 레이블로 분류 리포트를 생성
        report = classification_report(
            self.test_targets,  # 실제 레이블
            self.test_preds,  # 예측값
            target_names=['0', '1'],  # 클래스 이름 설정
            zero_division=0  # 0으로 나누는 오류 방지
        )
        print(report)  # 분류 리포트 출력

        if self.configs.get('nni'):
            nni.report_final_result(np.mean(self.test_losses))  # NNI가 활성화된 경우 최종 결과 보고

    def configure_optimizers(self):
        # 옵티마이저 및 학습률 스케줄러 설정
        optimizer = optim.Adam(
            self.model.parameters(),  # 모델 파라미터를 Adam 옵티마이저에 전달
            lr=self.learning_rate,  # 학습률 설정
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # 손실이 최소화될 때 학습률 감소
            factor=0.5,  # 학습률을 절반으로 감소
            patience=5,  # 손실이 개선되지 않으면 5 에포크 후 학습률 감소
        )

        return {
            'optimizer': optimizer,  # 옵티마이저 반환
            'lr_scheduler': {
                'scheduler': scheduler,  # 학습률 스케줄러 반환
                'monitor': 'loss/val_loss',  # 검증 손실을 기준으로 학습률 조정
                'interval': 'epoch',  # 에포크 단위로 조정
                'frequency': 1,  # 매 에포크마다 조정
            }
        }
