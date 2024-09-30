# SKN04-2nd-3Team

# 제발 가지마세요
<p align="center"><img src="https://github.com/user-attachments/assets/70e145c2-8248-445b-ad98-af5b065f064b" width="300" height="300"/></p>

<hr>

## 1. 팀 소개
### 팀명 : 제발 가지마세요
 
### 팀원


<div align="center">
	
|   &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp;  김문수  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;  이지수  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;  김태욱  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |     &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;  안준용  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;   |
|------------------------------------------|--------------------------------------|------------------------------------------|-----------------------------------|
|   &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp;  팀장/ EDA  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;  EDA/ML  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;  EDA/DL  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |     &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;  EDA/git/readme  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;   |

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
총 고객 수 : 51,047명 <br>
이탈 고객 수: 14,711명 (71.2%) <br>
유지 고객 수: 36,336명 (28.8%) <br>
결측치 수 : 1,295 <br>
훈련 대상 수 : 49,752 <br>


<hr>

### EDA
<hr>
칼럼별 히트맵<br>
<img src="https://github.com/user-attachments/assets/f8c69a56-5164-4c40-aadb-51c8e0d77984" width="400" height="400"/>
<br>
중요자료 히트맵<br>
<img src="https://github.com/user-attachments/assets/8bfd8060-22e7-49f6-98bd-2e754fc1a9b8" width="400" height="400"/>
<br>
가입 기간별 고객 이탈 상관분석<br>
<img src="https://github.com/user-attachments/assets/850a8050-57d2-4f5f-8834-8d7b9071000f" width="400" height="400"/>
<br>
고객 서비스 통화 횟수별 이탈률<br>
<img src="https://github.com/user-attachments/assets/f75f4de8-ff5d-47a7-a673-91ca23e1cd4b" width="400" height="400"/>
<br>
드랍콜 횟수별 이탈률<br>
<img src="https://github.com/user-attachments/assets/257031dd-85f8-47cf-86a6-9bbf96859c04" width="400" height="400"/>
<br>
초과사용시간별 이탈률<br>
<img src="https://github.com/user-attachments/assets/2ddc0aba-f3ac-47e5-9e11-9c271ccdfd8f" width="400" height="400"/>
<br>
월간 수익구간별 이탈률<br>
<img src="https://github.com/user-attachments/assets/2d01c09c-abb0-46e5-9f2e-7734a3fa0990" width="400" height="400"/>

<hr>

### Modeling

***modling 과정에 대한 내용 작성 (cv, tuning 포함)***

<hr>

### 수행 결과

***modeling 결과 해석 및 결론 도출***

<hr>

### 한 줄 회고

***회고 작성***