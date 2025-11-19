# rPPG Dataset README

## 1. 개요
이 디렉토리는 **TS-CAN 기반 rPPG 모델의 개인별(Self-Calibration) 학습**을 위해 사용되는  
`SelfCali_C00.mat` ~ `SelfCali_C10.mat` 형식의 데이터 파일들로 구성되어 있습니다.  
각 파일은 특정 사용자 또는 특정 촬영 세션에서 수집된 rPPG 원천 데이터를 포함합니다.

---

## 2. 데이터 구조

각 `.mat` 파일은 다음 두 개의 핵심 변수를 포함합니다.

### `dXsub`
- **크기:** `(T, 36, 36, 6)`
- **구성:**
  - 채널 0–2 → **diff 영상**: ΔR, ΔG, ΔB  
  - 채널 3–5 → **raw 영상**: R, G, B  
- **역할:**  
  - TS-CAN 모델의 **motion branch(diff)**  
  - TS-CAN 모델의 **appearance branch(raw)** 입력으로 사용됩니다.

### `dysub`
- **크기:** `(T, 1)`
- **내용:** BVP / rPPG ground truth 등 생리 신호 레이블  
- **역할:** 학습 시 회귀(label) 값으로 사용됩니다.

---

## 3. 사용 목적

SelfCali 데이터셋은 다음 두 가지 목적에 활용됩니다.

### 1) 개인별(Self-Calibration) 파인튜닝
- 기존 베이스 모델(`cv_0_epoch48_model.hdf5`)에 사용자 특성(얼굴 톤, 조명, 움직임)을 반영하여  
  **정확도를 높이기 위한 추가 학습 단계**입니다.

### 2) TS-CAN 모델 입력 구성
- 데이터는 `frame_depth = 10` 프레임 단위의 시퀀스로 변환되어  
  diff / raw 입력을 TS-CAN의 두 branch에 공급합니다.
- 레이블(`dysub`)은 시퀀스 동안의 평균 또는 마지막 값을 사용합니다.

---

## 4. 학습 과정 요약

1. `SelfCali_C*.mat` 파일 전체 로드  
2. `dXsub`에서 diff / raw 채널 분리  
3. window size = 10 프레임으로 시퀀스 구성  
4. 기존 TS-CAN 가중치 로드  
5. 낮은 learning rate로 **파인튜닝 진행**  
6. `personalized_selfcali.hdf5` 등의 이름으로 새로운 가중치 저장  
7. ONNX 변환 → 실시간 rPPG 추론(DeepStream 등)에서 사용

---

## 5. 파일 목록 예시
SelfCali_C00.mat
SelfCali_C01.mat
SelfCali_C02.mat
…
SelfCali_C10.mat
