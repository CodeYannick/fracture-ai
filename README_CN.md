# MNIST ResNet Web App (中文说明)

这是一个使用 PyTorch 在 MNIST 数据集上实现 ResNet18 的 AI 项目，并通过 Flask API 和 React + Vite 前端进行展示。

## 功能特性

- **模型**: ResNet18 (针对 1 通道 28x28 输入和 10 类输出进行了修改)。
- **后端**: 支持 CORS 的 Flask API。
- **前端**: React + Vite，带有 HTML5 Canvas 用于手写输入。
- **加速**: 自动支持 Apple Silicon (MPS/Metal)、CUDA 和 CPU。

## 目录结构说明

```
fracture-ai/
├── app.py                # Flask 后端 API 服务，处理前端请求并调用模型进行预测
├── model.py              # PyTorch 模型定义文件，包含修改后的 ResNet18 结构
├── train.py              # 模型训练脚本，负责下载数据、训练模型并保存权重
├── export_onnx.py        # 将训练好的 PyTorch 模型导出为 ONNX 格式的工具脚本
├── inspect_model.py      # 查看 .pth 模型权重文件结构的工具脚本
├── requirements.txt      # Python 项目依赖列表
├── README.md             # 英文项目说明文档
├── README_CN.md          # 中文项目说明文档
├── checkpoints/          # 存放训练好的模型权重文件 (.pth)
│   └── resnet_mnist.pth  # 训练生成的具体模型权重
├── data/                 # 存放自动下载的 MNIST 数据集
├── flask.log             # Flask 服务的运行日志
└── web/                  # 前端 React + Vite 项目目录
    ├── index.html        # 前端入口 HTML 文件
    ├── package.json      # 前端项目配置和依赖列表
    ├── vite.config.js    # Vite 配置文件，包含跨域代理设置
    └── src/              # 前端源代码目录
        ├── main.jsx      # React 应用入口
        ├── App.jsx       # 主应用组件，包含手写板逻辑和 API 调用
        ├── App.css       # 主样式文件
        └── index.css     # 全局样式文件
```

## 环境准备

- Python 3.8+
- Node.js & npm

## 安装与运行

### 1. 后端设置

安装 Python 依赖:
```bash
pip install -r requirements.txt
```

训练模型 (如果尚未训练):
```bash
python train.py
```

启动 Flask API 服务:
```bash
python app.py
```
服务将在 `http://127.0.0.1:5001` 启动。

### 2. 前端设置

进入 web 目录:
```bash
cd web
```

安装 Node 依赖:
```bash
npm install
```

启动开发服务器:
```bash
npm run dev
```
前端页面将在 `http://localhost:5173` 启动。

## 使用方法

1. 在浏览器中打开 `http://localhost:5173`。
2. 在画板上手写一个数字 (0-9)。
3. 点击 "Recognize" (识别) 按钮获取后台模型的预测结果。
