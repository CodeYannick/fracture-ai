import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
import numpy as np
from model import get_resnet_mnist_model
import os

app = Flask(__name__)
CORS(app)  # 启用所有路由的 CORS (跨域资源共享)

# 加载模型
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = get_resnet_mnist_model()

# 使用绝对路径加载 Checkpoint，防止相对路径问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "resnet_mnist.pth")

try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from {checkpoint_path} (模型加载成功)")
except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {checkpoint_path}. Please train the model first. (错误: 未找到模型文件，请先运行训练)")
    # 如果找不到文件，模型将使用随机权重初始化（仅用于确保服务不崩溃，但预测结果将是随机的）

model.to(device)
model.eval() # 设置为评估模式

# 预处理转换 (Preprocessing transform)
# 必须与训练时的归一化参数保持一致
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 转换为单通道灰度图
    transforms.Resize((28, 28)),                 # 调整大小为 28x28
    transforms.ToTensor(),                       # 转为 Tensor
    transforms.Normalize((0.1307,), (0.3081,))   # 归一化
])

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # 解码 Base64 图像数据
        # 前端发送的数据格式通常是 "data:image/png;base64,....."
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # 使用 PIL 打开图像
        image = Image.open(io.BytesIO(image_bytes))
        
        # 预处理
        input_tensor = transform(image).unsqueeze(0).to(device) # unsqueeze(0) 增加 batch 维度
        
        # 推理 (Inference)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1) # 计算概率
            confidence, predicted = torch.max(probabilities, 1) # 获取最大概率及其对应的类别
            
        return jsonify({
            'digit': int(predicted.item()),
            'confidence': float(confidence.item())
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 运行在 0.0.0.0 以便局域网访问 (如果在 Docker 中)
    # 禁用 reloader 以保证后台运行的稳定性
    app.run(host='0.0.0.0', port=5001, debug=False)
