# Handwritten Digit Recognition

A web application for recognizing handwritten digits using a TensorFlow/Keras CNN model.

## Setup

1. **Create and activate virtual environment** (if not already done):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not already trained):
   ```bash
   python train_model.py
   ```
   This will create `digit_model.h5` in the project directory.

## Running the Application

1. **Start the backend server**:
   ```bash
   source venv/bin/activate
   python backend.py
   ```
   Or use the convenience script:
   ```bash
   ./start_backend.sh
   ```
   
   The server will start on `http://127.0.0.1:8080`

2. **Open the frontend**:
   - Open `index.html` in your web browser
   - Or serve it using a local web server:
     ```bash
     python3 -m http.server 8000
     ```
     Then open `http://localhost:8000` in your browser

## Usage

1. Draw a digit on the canvas
2. Click "Predict" to get the prediction
3. Click "Clear" to clear the canvas

## Note

- The backend runs on port **8080** (not 5000) because macOS uses port 5000 for AirPlay Receiver
- Make sure the backend is running before using the frontend
- The model file (`digit_model.h5`) must exist for predictions to work
