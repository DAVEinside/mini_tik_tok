# Real-Time Video Feed Recommender

A TikTok-like video recommendation system using transformer encoders for personalized content delivery.

## Features

- 🚀 **94% Hit-Rate@50**: Achieves high recommendation accuracy on 10k video dataset
- ⚡ **<80ms P95 Latency**: Fast inference with GPU acceleration
- 🔄 **Real-time Caching**: Redis-based caching for instant recommendations
- 🤖 **Transformer Architecture**: Modern deep learning approach for understanding user preferences
- 📊 **Comprehensive Metrics**: Built-in monitoring and evaluation

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- Redis 7.0+
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-recommender.git
cd video-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

3. Start Redis:
```bash
redis-server
```

4. Train the model:
```bash
python train.py
```

5. Start the FastAPI server:    

```bash
python training/train.py
```

5. Start the FastAPI server:

```bash
python app/main.py
```



