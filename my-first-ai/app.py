from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import logging
import time

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = app.logger

# 상태 및 누적 텍스트 관리
accumulated_text = ""
last_label = None
label_start_time = time.time()

try:
    # 모델 로드
    model = tf.keras.models.load_model('model.h5')
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise RuntimeError(f"Model loading failed: {e}")

@app.route('/')
def health_check():
    """헬스 체크용 엔드포인트"""
    return jsonify({
        "status": "active",
        "model": "loaded",
        "accumulated_text": accumulated_text
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """수화 번역 요청 처리"""
    global accumulated_text, last_label, label_start_time

    # 1. 클라이언트 데이터 수신
    data = request.get_json()
    if not data or 'data' not in data:
        logger.warning("⚠️ Invalid request: missing 'data' field")
        return jsonify({"error": "Request must include 'data' field"}), 400

    try:
        # 2. 데이터 형식 검증 및 변환
        if not isinstance(data['data'], list):
            raise ValueError("Data must be a list")

        input_data = np.array(data['data']).reshape(1, -1)
        logger.info(f"📥 Received data: {str(data['data'])[:50]}...")

        # 3. 예측 수행
        prediction = model.predict(input_data)
        class_id = np.argmax(prediction)
        confidence = float(prediction[0][class_id])
        current_label = str(data.get('label', str(class_id)))  # 실제 라벨명 사용
        logger.info(f"🔍 Predicted: {current_label}, Confidence: {confidence:.2f}")

        # 4. START/STOP/DEFAULT 처리
        EXCLUDED_LABELS = {"START", "STOP", "DEFAULT"}
        
        # START 명령
        if 'command' in data and data['command'] == "START":
            accumulated_text = ""
            last_label = None
            label_start_time = time.time()
            logger.info("🔄 START: 텍스트 초기화")
            return jsonify({"status": "started", "accumulated_text": accumulated_text})
        
        # STOP 명령
        if 'command' in data and data['command'] == "STOP":
            result = accumulated_text
            accumulated_text = ""
            last_label = None
            label_start_time = time.time()
            logger.info("⏹️ STOP: 텍스트 반환 및 초기화")
            return jsonify({"result": result, "status": "stopped"})
        
        # DEFAULT 라벨 필터링
        if current_label in EXCLUDED_LABELS:
            logger.info(f"🚫 Excluded label: {current_label}")
            return jsonify({
                "status": "excluded",
                "current_label": current_label,
                "message": "This label is excluded from accumulation"
            })

        # 5. 수화 번역 누적 (2초 이상 같은 수화 유지 시만)
        if confidence > 0.6 and current_label.isalpha():  # 정확도 60% 이상, 알파벳만
            current_time = time.time()
            
            # 라벨 변경 감지
            if last_label != current_label:
                last_label = current_label
                label_start_time = current_time
                logger.info(f"🔄 라벨 변경: {current_label}")
            
            # 2초 이상 동일 라벨 유지 시 누적
            elif current_time - label_start_time >= 2.0:
                accumulated_text += current_label
                label_start_time = current_time
                logger.info(f"📝 누적: {current_label} → {accumulated_text}")

        # 6. 결과 반환
        return jsonify({
            "result": accumulated_text,
            "current_label": current_label,
            "confidence": confidence,
            "status": "recording"
        })

    except ValueError as ve:
        logger.error(f"🚫 Value error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("🔥 Unexpected error during prediction")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
