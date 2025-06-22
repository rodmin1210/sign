from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# 모델 로드
model = tf.keras.models.load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. 클라이언트에서 전송한 데이터 받기
    data = request.json['data']  # 예: [1.0, 2.0, 3.0]
    
    # 2. 모델 입력 형식으로 변환
    input_data = np.array(data).reshape(1, -1)
    
    # 3. 예측 수행
    prediction = model.predict(input_data).tolist()
    
    # 4. 결과 반환
    return jsonify({"result": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
