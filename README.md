# MNIST ResNet Web App

An AI project implementing ResNet18 on the MNIST dataset using PyTorch, served via a Flask API and a React + Vite frontend.

## Features

- **Model**: ResNet18 (modified for 1-channel 28x28 input and 10 classes).
- **Backend**: Flask API with CORS support.
- **Frontend**: React + Vite with HTML5 Canvas for handwriting input.
- **Acceleration**: Supports Apple Silicon (MPS/Metal), CUDA, and CPU automatically.

## Prerequisites

- Python 3.8+
- Node.js & npm

## Setup

### 1. Backend Setup

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Train the model (if you haven't already):
```bash
python train.py
```

Start the Flask API server:
```bash
python app.py
```
The server will start on `http://127.0.0.1:5001`.

### 2. Frontend Setup

Navigate to the web directory:
```bash
cd web
```

Install Node dependencies:
```bash
npm install
```

Start the development server:
```bash
npm run dev
```
The frontend will start on `http://localhost:5173`.

## Usage

1. Open `http://localhost:5173` in your browser.
2. Draw a digit (0-9) on the canvas.
3. Click "Recognize" to get the prediction from the backend model.

## Project Structure

- `model.py`: PyTorch model definition.
- `train.py`: Training script.
- `app.py`: Flask API server.
- `web/`: React + Vite frontend project.
- `checkpoints/`: Saved model weights.
