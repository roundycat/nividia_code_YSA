#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepStream + rPPG 연동 앱 (3_app.py)

- 입력: 파일 / RTSP / 카메라 등 (GStreamer URI)
- PGIE(Primary GIE)로 사람 검출 (ResNet10 detector + pgie_config.txt)
- 각 객체의 bbox ROI를 잘라 rPPGEngine 에 전달
- 10프레임 단위로 rPPG 신호를 추론하고 화면에 값 표시
"""

import sys
import ctypes
import math
import logging

import cv2
import numpy as np

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import Gst, GObject, GLib  # noqa: E402

import pyds  # noqa: E402

from rppg_inference import RPPGEngine  # 위에서 만든 엔진 모듈


# ------------------------------
# 설정값
# ------------------------------
PGIE_CONFIG_PATH = "pgie_config.txt"
RPPG_MODEL_PATH = "rppg_model.onnx"

# ResNet10 기본 people detector 기준 (labels.txt 에서 확인)
PERSON_CLASS_ID = 2  # 필요하면 0 또는 다른 값으로 수정

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("deepstream-rppg")


# ------------------------------
# DeepStream 유틸 함수들
# ------------------------------
def bus_call(bus, message, loop):
    """GStreamer bus 메시지 콜백 (EOS, ERROR 처리)"""
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        logger.info("End-Of-Stream reached.")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error(f"Error: {err}, {debug}")
        loop.quit()
    return True


def get_frame_numpy(gst_buffer, frame_meta):
    """
    DeepStream NvBufSurface -> numpy 배열 (RGBA) 로 변환
    """
    # 버퍼에서 NvBufSurface 가져오기
    surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
    # surface 는 (h, w, 4) RGBA float32 또는 uint8
    frame = surface.copy()
    # RGBA → BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return frame


# ------------------------------
# rPPG + OSD pad probe
# ------------------------------
rppg_engine = RPPGEngine(RPPG_MODEL_PATH, frame_depth=10, img_size=36)


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    nvosd 의 sink pad 에 붙는 프로브.
    - 각 프레임/객체에 대해 rPPG 추론 수행
    - 텍스트 메타를 추가하여 화면에 표시
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.warning("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # 배치 메타
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # frame_meta 객체로 캐스팅
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # 프레임 numpy 변환
        frame = get_frame_numpy(gst_buffer, frame_meta)
        frame_h, frame_w, _ = frame.shape

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # 사람(class_id)만 사용
            if obj_meta.class_id != PERSON_CLASS_ID:
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
                continue

            rect_params = obj_meta.rect_params

            # bbox 좌표 계산 및 클리핑
            left = max(int(rect_params.left), 0)
            top = max(int(rect_params.top), 0)
            width = max(int(rect_params.width), 0)
            height = max(int(rect_params.height), 0)

            right = min(left + width, frame_w - 1)
            bottom = min(top + height, frame_h - 1)

            if right <= left or bottom <= top:
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
                continue

            # 얼굴/상체 ROI crop
            roi = frame[top:bottom, left:right, :]

            # rPPG 추론 호출 (객체 ID 기준으로 버퍼 관리)
            object_id = obj_meta.object_id
            signal_value = rppg_engine.update(object_id, roi)

            # 아직 버퍼가 꽉 차지 않았으면 None 이므로 표시하지 않음
            if signal_value is not None:
                # OSD에 표시할 문자열 (필요하면 BPM으로 변환 로직을 따로 추가)
                display_text = f"Signal: {signal_value:.4f}"

                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                display_meta.num_text = 1

                text_params = display_meta.text_params[0]
                text_params.display_text = display_text

                # 텍스트 위치 (bbox 위쪽에 표시)
                text_params.x_offset = left
                text_params.y_offset = max(top - 10, 0)

                # 글꼴 크기, 색상 설정
                text_params.font_params.font_name = "Serif"
                text_params.font_params.font_size = 11
                text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)  # 흰색

                # 배경색 (반투명 검정)
                text_params.set_bg_clr = 1
                text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

                # display_meta 를 frame_meta 에 붙이기
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ------------------------------
# 파이프라인 생성
# ------------------------------
def create_pipeline(source_uri: str) -> Gst.Pipeline:
    """
    DeepStream 파이프라인 생성:
    uridecodebin → nvstreammux → nvinfer → nvvideoconvert → nvdsosd → nveglglessink
    """
    Gst.init(None)

    pipeline = Gst.Pipeline.new("deepstream-rppg-pipeline")

    if not pipeline:
        raise RuntimeError("Failed to create pipeline")

    # 요소 생성
    source = Gst.ElementFactory.make("uridecodebin", "source")
    if not source:
        raise RuntimeError("Unable to create uridecodebin")
    source.set_property("uri", source_uri)

    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    if not streammux:
        raise RuntimeError("Unable to create nvstreammux")
    streammux.set_property("batched-push-timeout", 4000000)
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", 1280)
    streammux.set_property("height", 720)
    streammux.set_property("live-source", 1)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        raise RuntimeError("Unable to create nvinfer (pgie)")
    pgie.set_property("config-file-path", PGIE_CONFIG_PATH)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
    if not nvvidconv:
        raise RuntimeError("Unable to create nvvideoconvert")

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        raise RuntimeError("Unable to create nvdsosd")

    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        raise RuntimeError("Unable to create nveglglessink")
    sink.set_property("sync", 0)

    # 요소들 파이프라인에 추가
    for elem in [streammux, pgie, nvvidconv, nvosd, sink]:
        pipeline.add(elem)

    # source 는 decodebin 이라 pad-added 시점에 streammux와 연결해야 함
    pipeline.add(source)

    def cb_newpad(decodebin, pad, data):
        logger.info("Decodebin pad added")
        caps = pad.get_current_caps()
        if not caps:
            caps = pad.get_allowed_caps()
        string = caps.to_string()
        logger.info("  caps: %s", string)

        # streammux 의 sink pad 찾기
        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            logger.error("Unable to get streammux sink pad")
            return

        if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
            logger.error("Failed to link decodebin to streammux")

    source.connect("pad-added", cb_newpad, None)

    # 나머지 요소 연결: streammux → pgie → nvvidconv → nvosd → sink
    if not streammux.link(pgie):
        raise RuntimeError("Failed to link streammux -> pgie")
    if not pgie.link(nvvidconv):
        raise RuntimeError("Failed to link pgie -> nvvidconv")
    if not nvvidconv.link(nvosd):
        raise RuntimeError("Failed to link nvvidconv -> nvosd")
    if not nvosd.link(sink):
        raise RuntimeError("Failed to link nvosd -> sink")

    # nvosd 의 sink pad 에 pad-probe 설정
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        raise RuntimeError("Unable to get sink pad of nvosd")

    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

    return pipeline


# ------------------------------
# main
# ------------------------------
def main():
    if len(sys.argv) < 2:
        print(
            "사용법: python3 3_app.py <GStreamer URI>\n"
            "예시:\n"
            "  - 로컬 파일 : python3 3_app.py file:///home/user/video.mp4\n"
            "  - RTSP 스트림: python3 3_app.py rtsp://user:pass@ip:port/stream\n"
        )
        sys.exit(1)

    source_uri = sys.argv[1]

    # GObject 메인루프
    loop = GLib.MainLoop()

    # 파이프라인 생성
    pipeline = create_pipeline(source_uri)

    # Bus 설정 (EOS / ERROR 처리)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # 재생 시작
    logger.info("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, stopping...")

    # 종료 처리
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()