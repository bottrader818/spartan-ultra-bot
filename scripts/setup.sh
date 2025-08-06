#!/bin/bash
# Quant Bot Pro Installation Script

echo "🔧 Installing Quant Bot Pro with Neural Enhancements"

# 1. Install Python dependencies
pip install -r requirements.txt
pip install tensorflow-cpu==2.10.0 ta-lib numpy==1.23.0 joblib

# 2. TA-Lib verification
python -c "import talib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ TA-Lib failed to import. Please install system-level TA-Lib library first."
    echo "For Ubuntu: sudo apt install libta-lib0 libta-lib-dev"
    exit 1
fi

# 3. Download pretrained models
mkdir -p models
wget --retry-connrefused --waitretry=1 --timeout=20 --tries=3 https://quant-models.com/free/dtw_predictor.h5 -O models/dtw_predictor.h5
wget --retry-connrefused --waitretry=1 --timeout=20 --tries=3 https://quant-models.com/free/slippage_model.pkl -O models/slippage_model.pkl

# 4. Warm up engine
python -c "
from core.engine.trading_engine import TradingEngine
engine = TradingEngine(config_path='config/environments/live.env')
engine.warmup()
"

# 5. Run validation tests
python -m pytest tests/integration/test_engine.py -v

echo "✅ Installation Complete"
echo "➡️  Run: python run.py --strategy dtw_neural --mode aggressive"
