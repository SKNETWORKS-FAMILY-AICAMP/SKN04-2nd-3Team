# SKN04-2nd-3Team

# 제발 가지마세요
<p align="center"><img src="https://github.com/user-attachments/assets/70e145c2-8248-445b-ad98-af5b065f064b" width="700" height="300"/></p>

<hr>

## 1. 팀 소개
### 팀명 : 제발 가지마세요

### 팀원


<div align="center">
	
<table style="width: 100%; border-collapse: collapse;">
    <tr>
        <th style="width: 25%; text-align: center;">김문수</th>
        <th style="width: 25%; text-align: center;">이지수</th>
        <th style="width: 25%; text-align: center;">김태욱</th>
        <th style="width: 25%; text-align: center;">안준용</th>
    </tr>
    <tr>
        <td style="text-align: center;">Project Leader/eda</td>
        <td style="text-align: center;">ML</td>
        <td style="text-align: center;">DL</td>
        <td style="text-align: center;">readme/git</td>
    </tr>
    <tr>
        <td style="text-align: center;">
            <img src="https://github.com/user-attachments/assets/dc6bcf90-3d13-42b0-a786-715c439ce82c" width="150" height="150" />
        </td>
        <td style="text-align: center;">
            <img src="https://github.com/user-attachments/assets/bfcd973c-f5a2-4387-8ff6-44871b3ee15e" width="150" height="150" />
        </td>
        <td style="text-align: center;">
            <img src="https://github.com/user-attachments/assets/ba21b09c-b2b6-4842-920d-fd4853b61672" width="150" height="150" />
        </td>
        <td style="text-align: center;">
            <img src="https://github.com/user-attachments/assets/22a242fd-bc86-4329-9456-a5ea1ff69df1" width="150" height="150" />
        </td>
        <tr>
        <td style="text-align: center;">가지</td>
        <td style="text-align: center;">마</td>
        <td style="text-align: center;">고구마</td>
        <td style="text-align: center;">벤제마</td>
    </tr>
        
    
</table>
</div>

<hr>

## 2. 프로젝트 개요 
### 개발 기간
2024-09-27 ~ 2024-09-30
### 프로젝트 명 
바짓가랑이 붙잡기 
### 프로젝트 소개 
'바짓가랑이 붙잡기' 프로젝트는 가상의 통신사를 대상으로, 이탈 고객 데이터를 분석하여 고객의 니즈를 파악하고 미래에 이탈 가능성이 있는 고객을 예측하는 프로젝트입니다. 이를 바탕으로 적절한 대응 방안을 제시해 고객 이탈을 방지하고, 고객 만족도를 향상시키는 것이 목표입니다.
### 프로젝트 내용
#### 프로젝트 배경
과거에는 통신사의 선택 기준이 단순히 통화 품질에만 국한되었으나, 모바일 시장의 확대로 인해 통신사를 선택하는 기준이 다변화되었습니다. 이에 따라 통신사들은 고객 이탈 방지를 위한 다양한 전략을 고민하고 있으며, 본 프로젝트는 이를 데이터 기반으로 분석하여 실질적인 해결책을 도출하는 데 목적이 있습니다.
#### 프로젝트 목표 
- **고객이탈 데이터 분석**: 이탈한 고객들의 데이터를 분석하여 이탈 원인을 규명하고 고객 서비스 개선 방안을 도출합니다.
- **ML/DL 모델 선정**: 여러 머신러닝 및 딥러닝 모델을 비교 분석하여 고객 이탈 예측에 가장 적합한 모델을 선정합니다.
- **고객 세그먼테이션**: 수익, 사용량, 서비스 기간 등을 기준으로 고객을 세분화하고, 각 세그먼트에 맞는 맞춤형 고객 유지 전략을 수립합니다.
- **서비스 개선 및 VIP 고객 관리**: 수익 또는 장기 고객의 이탈을 방지하기 위해, 맞춤형 혜택이나 VIP 서비스를 제안합니다.
<hr>

### 🔨 기술 스택
<div>
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=mysql&logoColor=white">
<img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=mysql&logoColor=white">
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=mysql&logoColor=white">
<img src="https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=git&logoColor=white">
<img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">
<img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
</div>

<hr>

### Prerequisites
**이 프로젝트를 실행하기 위해 필요한 패키지 등을 정의**

```cmd
pip install -r requirements.txt
```

<hr>

### Usage
**이 코드를 실행하기 위해 어떠한 코드를 어떻게 실행해야 하는지 작성**

```cmd
python main.py
```

<hr> 

### Data

<img src="https://github.com/user-attachments/assets/b00f460f-78ac-4087-81d6-09270080c4e8" width="300" height="300"/>
<img src="https://github.com/user-attachments/assets/fd1c1ee5-bb4d-4513-b057-91e802b2eec6" width="300" height="300"/>

데이터셋의 기본 정보
총 고객 수 : 51,047명 <br><br>
이탈 고객 수: 14,711명 (71.2%) <br><br>
유지 고객 수: 36,336명 (28.8%) <br><br>
결측치 수 : 1,295 <br><br>
훈련 대상 수 : 49,752 <br>


<hr>

## EDA


#### 특성별 히트맵 : 통신 관련 자료를 각 특성별 상관관계 비율에 대한 히트맵<br>
<img src="https://github.com/user-attachments/assets/f8c69a56-5164-4c40-aadb-51c8e0d77984" width="500" height="500"/>
<br> 전체 히트맵의 상관관계는 고객이탈과의 관계가 전체 적으로 높아 보이지 않음
<br><br>

#### 중요 특성 히트맵 : 고객 이탈 상관관계가 높은 특성에 대한 히트맵<br>
<img src="https://github.com/user-attachments/assets/8bfd8060-22e7-49f6-98bd-2e754fc1a9b8" width="500" height="500"/>
<br> 고객 이탈 상관관계가 높은 특성을 가진 히트맵 차트를 그렸지만 각 특성별 상관 관계는 높아 보이지만 고객 이탈 상관관계는 높아 보이지 않음

<br><br>

#### 월간 수익구간별 이탈률 : 고객의 수익구간별 고객이탈에 대한 비율에 대한 차트 <br>
<img src="https://github.com/user-attachments/assets/2d01c09c-abb0-46e5-9f2e-7734a3fa0990" width="500" height="500"/>
<br> 전체 수익 구간을 5등 부분으로 나눌때 수익구간이 적을때 근소하게 높은 수치가 보임
<br> 
<br>

#### 가입 기간별 고객 이탈 상관분석<br>
<img src="https://github.com/user-attachments/assets/850a8050-57d2-4f5f-8834-8d7b9071000f" width="500" height="500"/>
<br> 고객 가입 초기 12개월 안의 고객이탈 비율이 높게 나타남.
<br>
<br>

#### 고객 서비스 통화 횟수별 이탈률<br>
<img src="https://github.com/user-attachments/assets/f75f4de8-ff5d-47a7-a673-91ca23e1cd4b" width="500" height="500"/>
<br> 서비스 센터 통화 횟수가 3~4 횟수에서 이탈률이 증가함.
<br>
<br>

#### Drop Call 횟수별 이탈률<br>
<img src="https://github.com/user-attachments/assets/257031dd-85f8-47cf-86a6-9bbf96859c04" width="500" height="500"/>
<br> 통신사 통화 품질불량 횟수가 2 ~ 10 회에서 높게 나타남
<br>
<br>

#### 초과사용시간별 이탈률<br>
<img src="https://github.com/user-attachments/assets/2ddc0aba-f3ac-47e5-9e11-9c271ccdfd8f" width="500" height="500"/>
<br> 사용 초과 고객에 대하여 높은 상관관계가 보임
<br>
<br>

<hr>

### Modeling

ML : 분석결과 상관관계가 있는 특성들이 없기때문에 차원축소를 통해 특성 수를 줄이고 클러스터링을 통해 군집을 확인해봤지만 차원축소로는 해결하기 힘들다고 판단하게됨. 다음 시도로 svm의 비선형 커널(poly,rbf)를 이용하여 모델을 만들었고 결과가 이전 모델보다 향상되어 채택하게 됨.
결과적으로 CustomerCareCalls, MonthsInService, DroppedCalls, OverageMinutes 네개의 특성이 이탈률과 연관이 있다고 판단됨.

DL : 
#### 가중치 보정 전
<br>
<img src="https://github.com/user-attachments/assets/e53a785e-db65-4f81-b2e5-a742cc568179" width="500" height="200"/>
<br> 
<br><br>

#### 가중치 보정 후<br>
<img src="https://github.com/user-attachments/assets/d4b6e927-41a4-4359-a7b8-549b3e332f92" width="500" height="200"/>
<br> 

<br><br>

#### 가중치 보정 후 + 리니어 4개 <br>
<img src="https://github.com/user-attachments/assets/ebbd51c1-b21f-4f4e-81bd-bda894b1690a" width="500" height="200"/>
<br>
<br> 
<br>


<hr>

### 수행 결과

***modeling 결과 해석 및 결론 도출***

<hr>

<hr>

### 주요 인사이트 돌출 및 제안

#### 1. 장기 고객 관리: 서비스 이용 기간이 길어질수록 이탈 가능성이 높아지므로, 장기 고객을 위한 특별 혜택이나 로열티 프로그램을 고려해볼 수 있습니다.

#### 2. 구독 최적화: 고유 구독 수와 활성 구독 수가 많을수록 이탈 가능성이 높아지므로, 고객의 실제 니즈에 맞는 최적화된 구독 패키지를 제안하는 것이 중요합니다.

#### 3. 수익 모델 개선: 월 수익과 총 반복 요금이 낮을수록 이탈 가능성이 약간 높아지므로, 고객에게 더 많은 가치를 제공하면서 수익을 높일 수 있는 방안을 모색해야 합니다.

#### 4. 사용량 기반 전략: 월간 사용 시간과 초과 사용 시간이 수익과 강한 상관관계를 보이므로, 사용량 기반의 개인에 맞는 맞품형 요금제나 프로모션을 고려해볼 수 있습니다.

#### 5. 고객 제안 : 다양한 변수들 간의 상관관계를 바탕으로 고객을 세분화하여, 각 중요 특정에 맞는 맞춤형 서비스와 마케팅 전략을 수립할 수 있습니다.



### 한 줄 회고

김문수: EDA 분석을 했는데 생각만큼 이탈 상관분석 비율 이 나오지 않아 힘들었다. 그래도 풀젝은 끝나서 좋다.