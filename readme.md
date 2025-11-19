# DeepStream rPPG Project (TS-CAN)

이 프로젝트는 **NVIDIA DeepStream SDK**를 사용하여 실시간으로 사람의 얼굴을 감지하고, **rPPG (Remote Photoplethysmography)** 기술을 통해 비접촉으로 심박 신호(Blood Volume Pulse)를 추출하는 애플리케이션입니다.

기존 Android용 `TS-CAN` 딥러닝 모델을 **ONNX**로 변환하여 Jetson(Nano, Xavier, Orin) 및 GPU 서버 환경에 이식했습니다.

## 📋 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [파일 구조](#파일-구조)
3. [설치 및 환경 설정](#설치-및-환경-설정)
4. [실행 방법 (Step-by-Step)](#실행-방법)
5. [트러블슈팅](#트러블슈팅)

---

## 💻 시스템 요구사항

* **Hardware:** NVIDIA Jetson Series (Nano, TX2, Xavier, Orin) 또는 NVIDIA GPU가 장착된 Linux PC.
* **Software:**
    * Ubuntu 18.04 / 20.04
    * **DeepStream SDK 6.x** 이상
    * Python 3.6+
    * DeepStream Python Bindings (`pyds`)
* **Python Libraries:**
    ```bash
    pip3 install opencv-python numpy onnxruntime-gpu tensorflow tf2onnx
    ```

---

## 📂 파일 구조

프로젝트 폴더는 다음과 같이 구성되어야 합니다.

```text
.
├── 1_export.py             # [1단계] Keras 모델(.hdf5)을 ONNX로 변환하는 스크립트
├── 2_inference.py          # [핵심] rPPG 전처리(미분, 정규화) 및 추론 엔진 클래스
├── 3_app.py                # [2단계] DeepStream 파이프라인 메인 실행 앱
├── model.py                # TS-CAN 모델 구조 정의 (설계도)
├── cv_0_epoch48_model.hdf5 # 학습된 모델 가중치 파일 (필수)
├── pgie_config.txt         # DeepStream 얼굴(사람) 인식 설정 파일
└── rppg_model.onnx         # (1단계 실행 후 생성됨) 딥스트림용 모델 파일
```

---

## 🚀 실행 방법

### 1단계: 모델 변환 (Export to ONNX)
DeepStream에서 사용하기 위해 학습된 Keras 모델(`hdf5`)을 `ONNX` 포맷으로 변환합니다.

1. `cv_0_epoch48_model.hdf5` 파일이 폴더에 있는지 확인합니다.
2. 아래 명령어를 실행합니다.

```bash
python3 1_export.py
```

> **성공 시:** 폴더에 `rppg_model.onnx` 파일이 생성됩니다.

### 2단계: DeepStream 앱 실행
변환된 모델과 DeepStream 파이프라인을 연결하여 앱을 실행합니다.

1. `pgie_config.txt`가 폴더에 있는지 확인합니다.
2. 아래 명령어를 실행합니다.

```bash
python3 3_app.py
```

> **작동 확인:** 카메라 창이 열리고, 사람 얼굴 위에 초록색 박스와 함께 `Signal: 0.xxxx` 형태의 수치가 실시간으로 표시되면 성공입니다.

---

## ⚙️ 설정 가이드 (Configuration)

### 얼굴 인식 대상 변경 (중요)
`pgie_config.txt`는 기본적으로 **ResNet10** 모델을 사용하며, 이는 사람(Person), 자동차(Car) 등을 모두 감지합니다. `3_app.py`에서 정확한 대상을 필터링해야 합니다.

* **`3_app.py` 수정:**
    * ResNet10 기본 모델 사용 시: `if obj_meta.class_id == 2:` (사람)
    * FaceDetect 전용 모델 사용 시: `if obj_meta.class_id == 0:` (얼굴)

### 모델 파라미터 수정
rPPG 모델의 입력 크기나 버퍼 길이를 변경하려면 `2_inference.py`를 수정하세요.
```python
self.buffer = deque(maxlen=10)  # 프레임 버퍼 길이 (TS-CAN 기본값: 10)
self.img_size = 36              # 입력 이미지 크기 (TS-CAN 기본값: 36)
```

---

## 🔧 트러블슈팅 (FAQ)

**Q1. `rppg_model.onnx` 파일이 안 만들어져요.**
* `cv_0_epoch48_model.hdf5` 파일이 없으면 랜덤 가중치로 생성됩니다. 학습된 파일 경로가 맞는지 `1_export.py` 내부를 확인하세요.

**Q2. 앱은 켜지는데 `Signal` 값이 안 떠요.**
* `3_app.py`의 `class_id`가 맞는지 확인하세요. (사람은 2번, 얼굴 전용 모델은 0번입니다.)
* 카메라에 얼굴이 너무 작게 잡히면 인식이 안 될 수 있습니다. 가까이 다가가세요.

**Q3. 속도가 너무 느려요.**
* `2_inference.py`에서 `onnxruntime`이 GPU(`CUDAExecutionProvider`)를 사용하고 있는지 확인하세요.
* `jetson_stats`(`jtop`)를 켜서 GPU 점유율을 확인해 보세요.
