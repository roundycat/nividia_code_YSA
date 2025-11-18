import tensorflow as tf
import tf2onnx
from model import TS_CAN  # 사용자가 제공한 model.py

# --- train_txt.py 분석 결과 적용 ---
# train_txt.py에서 args.frame_depth=10, args.img_size=36으로 설정됨
FRAME_DEPTH = 10
IMG_SIZE = 36
BATCH_SIZE = 1  # 실시간 추론은 1명씩 처리

def export_onnx():
    print("🚀 모델 변환 시작...")
    
    # 1. 모델 구조 생성 (TS_CAN 사용)
    # model.py에 정의된 TS_CAN 구조를 그대로 가져옴
    model = TS_CAN(
        n_frame=FRAME_DEPTH, 
        nb_filters1=32, 
        nb_filters2=64, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3), 
        nb_dense=128
    )

    # (중요) 학습된 가중치가 있다면 여기서 로드 (없으면 랜덤 가중치)
    # train_txt.py에 언급된 가중치 파일 경로
    weights_path = './cv_0_epoch48_model.hdf5'
    try:
        model.load_weights(weights_path)
        print(f"✅ 가중치 로드 성공: {weights_path}")
    except:
        print("⚠️ 경고: 학습된 가중치 파일이 없어 '랜덤 가중치'로 변환합니다.")

    # 2. 입력 스펙 정의 (DeepStream이 알 수 있도록)
    # TS_CAN은 입력이 2개임: (Diff, Raw)
    spec = (
        tf.TensorSpec((BATCH_SIZE, FRAME_DEPTH, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="input_diff"),
        tf.TensorSpec((BATCH_SIZE, FRAME_DEPTH, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="input_raw")
    )

    # 3. ONNX 변환
    output_path = "rppg_model.onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print(f"🎉 변환 완료: {output_path} (이 파일을 DeepStream에서 씁니다)")

if __name__ == "__main__":
    export_onnx()