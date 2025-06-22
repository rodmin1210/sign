from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import logging
import time

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignLanguageAPI")

# 상태 관리 초기화
accumulated_text = ""
last_label = None
label_start_time = time.time()
model = None  # 지연 로딩을 위해 None으로 초기화

def load_model():
    """모델 지연 로딩 함수"""
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model('model.h5')
            # 모델이 컴파일되지 않은 경우 수동 컴파일
            if not model._is_compiled:
                model.compile(optimizer='adam', loss='categorical_crossentropy')
            logger.info("✅ Model loaded and compiled successfully")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    return model

@app.route('/')
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({
        "status": "active",
        "model_loaded": model is not None,
        "accumulated_text": accumulated_text
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """예측 요청 처리"""
    global accumulated_text, last_label, label_start_time

    # 1. 지연 모델 로딩
    model = load_model()
    
    # 2. 요청 데이터 검증
    data = request.get_json()
    if not data:
        logger.warning("⚠️ Empty request received")
        return jsonify({"error": "Request body is empty"}), 400
        
    # 3. 명령어 처리 (START/STOP)
    if 'command' in data:
        command = data['command'].upper()
        if command == "START":
            accumulated_text = ""
            last_label = None
            label_start_time = time.time()
            logger.info("🔄 START: Text buffer reset")
            return jsonify({"status": "started", "accumulated_text": accumulated_text})
            
        elif command == "STOP":
            result = accumulated_text
            accumulated_text = ""
            logger.info(f"⏹️ STOP: Returned text: '{result}'")
            return jsonify({"result": result, "status": "stopped"})
    
    # 4. 데이터 검증
    if 'data' not in data or not isinstance(data['data'], list):
        logger.warning("⚠️ Invalid data format")
        return jsonify({"error": "Missing or invalid 'data' field"}), 400
        
    # 5. 예측 수행
    try:
        input_data = np.array(data['data']).reshape(1, -1)
        prediction = model.predict(input_data)
        class_id = np.argmax(prediction)
        confidence = float(prediction[0][class_id])
        current_label = data.get('label', str(class_id))
        
        # 6. 라벨 필터링 (DEFAULT 제외)
        EXCLUDED_LABELS = {"START", "STOP", "DEFAULT"}
        if current_label in EXCLUDED_LABELS:
            logger.info(f"🚫 Skipped excluded label: {current_label}")
            return jsonify({
                "status": "excluded",
                "current_label": current_label
            })
        
        # 7. 텍스트 누적 로직
        current_time = time.time()
        if confidence > 0.6 and current_label.isalpha():
            # 라벨 변경 감지
            if last_label != current_label:
                last_label = current_label
                label_start_time = current_time
                logger.info(f"🔄 Label changed to: {current_label}")
            
            # 2초 이상 동일 라벨 유지 시 누적
            elif current_time - label_start_time >= 2.0:
                accumulated_text += current_label
                label_start_time = current_time
                logger.info(f"✍️ Accumulated: '{current_label}' → Full text: '{accumulated_text}'")
        
        return jsonify({
            "status": "success",
            "current_label": current_label,
            "confidence": confidence,
            "accumulated_text": accumulated_text
        })
        
    except Exception as e:
        logger.exception(f"🔥 Prediction failed: {e}")
        return jsonify({"error": "Prediction processing failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
