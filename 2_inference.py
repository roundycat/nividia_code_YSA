#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rPPG ONNX 추론 엔진

- TS-CAN 기반 ONNX 모델(rppg_model.onnx)을 로드해서
- 얼굴 ROI 시퀀스(10프레임)를 입력받아
- 스칼라 신호(예: BVP / rPPG)를 추론하는 클래스 정의

DeepStream 연동용으로:
    - 객체 ID(DeepStream object_id)마다 버퍼를 따로 관리
    - update(obj_id, face_roi_bgr)를 반복 호출하면
      10프레임이 쌓인 시점에 float 값을 반환
"""

import os
from collections import defaultdict, deque
from typing import Dict, Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime 패키지를 먼저 설치하세요: pip install onnxruntime-gpu 혹은 onnxruntime")


class RPPGEngine:
    """
    DeepStream에서 객체별로 얼굴 ROI를 받아 rPPG를 추론하는 엔진.

    - object_id 별로 프레임 버퍼를 관리
    - 버퍼 길이가 FRAME_DEPTH(기본 10)에 도달하면 ONNX 추론 수행
    """

    def __init__(
        self,
        model_path: str = "rppg_model.onnx",
        frame_depth: int = 10,
        img_size: int = 36,
        use_gpu: bool = True,
    ) -> None:
        self.model_path = model_path
        self.frame_depth = frame_depth
        self.img_size = img_size

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {self.model_path}")

        # ONNX Runtime 세션 생성 (GPU → CPU 순으로 시도)
        providers = []
        if use_gpu:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(self.model_path, providers=providers)

        # 입력/출력 정보 (디버그용)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        # 객체 ID 별로 최근 프레임 버퍼 관리
        self.buffers: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.frame_depth)
        )

    # --------------------
    # 내부 유틸
    # --------------------
    def _preprocess_roi(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        BGR ROI (H, W, 3, uint8) → 정규화된 RGB (img_size, img_size, 3, float32)
        """
        if roi_bgr is None or roi_bgr.size == 0:
            raise ValueError("입력 ROI가 비어 있습니다.")

        # 리사이즈
        resized = cv2.resize(roi_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 0~1 정규화
        rgb = rgb.astype(np.float32) / 255.0

        return rgb

    def _build_inputs(self, seq: np.ndarray) -> Dict[str, np.ndarray]:
        """
        시퀀스 텐서( T, H, W, C ) → ONNX 입력 dict 생성
        - raw: (1, T, H, W, C)
        - diff: (1, T, H, W, C)  (프레임 차이 + 정규화)
        """
        # (T, H, W, C) → (1, T, H, W, C)
        raw_input = np.expand_dims(seq, axis=0).astype(np.float32)

        # diff 계산
        diff_input = np.zeros_like(raw_input, dtype=np.float32)
        diff_input[:, 1:, ...] = raw_input[:, 1:, ...] - raw_input[:, :-1, ...]
        # 간단한 표준편차 정규화
        std = np.std(diff_input)
        if std > 1e-6:
            diff_input /= std

        # 입력 이름은 1_export.py에서 지정한 이름과 동일해야 함
        # 여기서는 "input_diff", "input_raw" 로 가정
        inputs = {
            "input_diff": diff_input,
            "input_raw": raw_input,
        }
        return inputs

    # --------------------
    # 외부 API
    # --------------------
    def reset_object(self, object_id: int) -> None:
        """특정 객체 ID의 프레임 버퍼 초기화"""
        if object_id in self.buffers:
            self.buffers[object_id].clear()

    def reset_all(self) -> None:
        """모든 객체 프레임 버퍼 초기화"""
        self.buffers.clear()

    def update(self, object_id: int, face_roi_bgr: np.ndarray) -> Optional[float]:
        """
        DeepStream에서 매 프레임마다 호출하는 함수.

        Parameters
        ----------
        object_id : int
            DeepStream NvDsObjectMeta.object_id
        face_roi_bgr : np.ndarray
            해당 객체의 얼굴 ROI (BGR, HxWx3, uint8)

        Returns
        -------
        Optional[float]
            - 버퍼가 아직 frame_depth에 도달하지 않았으면 None
            - 도달했다면 ONNX 추론 결과(스칼라)를 float 로 반환
        """
        try:
            frame_norm = self._preprocess_roi(face_roi_bgr)
        except Exception:
            # 전처리 실패 시 해당 프레임은 무시
            return None

        buf = self.buffers[object_id]
        buf.append(frame_norm)

        # 아직 시퀀스 길이가 충분치 않으면 추론하지 않음
        if len(buf) < self.frame_depth:
            return None

        # deque → np.ndarray (T, H, W, C)
        seq = np.stack(list(buf), axis=0)  # (T, H, W, C)

        inputs = self._build_inputs(seq)

        # 추론
        outputs = self.session.run(self.output_names, inputs)

        # 출력이 (1, 1) 또는 (1,) 형태라고 가정
        if isinstance(outputs, list):
            out = outputs[0]
        else:
            out = outputs

        value = float(out.flatten()[0])
        return value


if __name__ == "__main__":
    # 간단한 로컬 테스트용 (실제 DeepStream에서는 사용 안 해도 됨)
    engine = RPPGEngine("rppg_model.onnx", frame_depth=10, img_size=36)
    print("입력 이름:", engine.input_names)
    print("출력 이름:", engine.output_names)
    print("간단 초기화 테스트 완료.")