#!/bin/bash
# Quick Start Script for EmotiSense AI

echo "╔════════════════════════════════════════════════════════╗"
echo "║         🎤 EMOTISENSE AI - QUICK START                ║"
echo "║     Indian Speech Emotion Recognition System           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Check Python
echo "[1] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi
python3 --version
echo "✅ Python found"
echo ""

# Check required files
echo "[2] Checking project files..."
files=("app.py" "index.html" "requirements.txt" "best_model_weights.h5" "scaler.pickle")
for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        exit 1
    fi
    echo "✅ $file"
done
echo ""

# Install dependencies
echo "[3] Installing dependencies (this may take 2-3 minutes)..."
pip install -r requirements.txt --break-system-packages -q
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Test model loading
echo "[4] Verifying model files..."
python3 << 'EOF'
import h5py
import pickle
import json

try:
    # Check weights
    f = h5py.File('best_model_weights.h5', 'r')
    print(f"✅ Weights file: {len(f['layers'])} layers found")
    f.close()
    
    # Check scaler
    scaler = pickle.load(open('scaler.pickle', 'rb'))
    print(f"✅ Scaler: {scaler.n_features_in_} features")
    
    # Check model config
    with open('CNN_model.json') as f:
        config = json.load(f)
    print(f"✅ Model config: {config['config']['layers'][-1]['config']['units']} output classes")
    
    print("\n✅ All model files verified successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Model verification failed"
    exit 1
fi
echo ""

# Start server
echo "[5] Starting EmotiSense server..."
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  🚀 Server running at: http://localhost:5000          ║"
echo "║                                                        ║"
echo "║  1. Open browser to http://localhost:5000            ║"
echo "║  2. Click 'Start Recording'                          ║"
echo "║  3. Speak for 2-3 seconds                            ║"
echo "║  4. View emotion detection results                   ║"
echo "║                                                        ║"
echo "║  Press Ctrl+C to stop the server                    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

python3 app.py
