import torch
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from app.models.transformer_recommender import VideoRecommender
from app.core.config import settings

def evaluate_hit_rate(
    model: VideoRecommender,
    data_loader: DataLoader,
    k_values: List[int] = [10, 20, 50],
    device: torch.device = torch.device('cuda')
) -> Dict[str, float]:
    """Evaluate hit rate at different k values"""
    
    model.eval()
    hit_rates = {k: [] for k in k_values}
    
    with torch.no_grad():
        for batch in data_loader:
            user_ids = batch['user_id'].to(device)
            history_ids = batch['history'].to(device)
            masks = batch['mask'].to(device)
            targets = batch['target'].to(device)
            
            for k in k_values:
                # Get top-k recommendations
                recs, _ = model.get_recommendations(user_ids, history_ids, masks, top_k=k)
                
                # Check if target is in recommendations
                hits = (recs == targets.unsqueeze(1)).any(dim=1).float()
                hit_rates[k].extend(hits.cpu().numpy())
    
    # Calculate average hit rates
    avg_hit_rates = {f"hit_rate@{k}": np.mean(hit_rates[k]) for k in k_values}
    
    return avg_hit_rates

def evaluate_latency(
    model: VideoRecommender,
    num_samples: int = 1000,
    batch_sizes: List[int] = [1, 8, 16, 32],
    device: torch.device = torch.device('cuda')
) -> Dict[str, Dict[str, float]]:
    """Evaluate model inference latency"""
    
    import time
    
    model.eval()
    latency_results = {}
    
    for batch_size in batch_sizes:
        latencies = []
        
        # Warmup
        for _ in range(10):
            dummy_users = torch.randint(0, 1000, (batch_size,)).to(device)
            dummy_history = torch.randint(0, settings.VIDEO_DATASET_SIZE, (batch_size, 50)).to(device)
            with torch.no_grad():
                model.get_recommendations(dummy_users, dummy_history, top_k=50)
        
        # Measure latency
        for _ in range(num_samples // batch_size):
            users = torch.randint(0, 1000, (batch_size,)).to(device)
            history = torch.randint(0, settings.VIDEO_DATASET_SIZE, (batch_size, 50)).to(device)
            
            start_time = time.time()
            
            with torch.no_grad():
                model.get_recommendations(users, history, top_k=50)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate statistics
        latency_results[f"batch_{batch_size}"] = {
            "mean_ms": np.mean(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99)
        }
    
    return latency_results

def plot_metrics(hit_rates: Dict[str, float], latencies: Dict[str, Dict[str, float]]):
    """Plot evaluation metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Hit rates
    k_values = [int(k.split('@')[1]) for k in hit_rates.keys()]
    rates = list(hit_rates.values())
    
    ax1.plot(k_values, rates, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=0.94, color='r', linestyle='--', label='Target: 94%')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Hit Rate')
    ax1.set_title('Hit Rate @ K')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Latencies
    batch_sizes = [int(k.split('_')[1]) for k in latencies.keys()]
    p95_latencies = [v['p95_ms'] for v in latencies.values()]
    
    ax2.bar(batch_sizes, p95_latencies, alpha=0.7)
    ax2.axhline(y=80, color='r', linestyle='--', label='Target: 80ms')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('P95 Latency (ms)')
    ax2.set_title('Inference Latency by Batch Size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png')
    plt.show()

if __name__ == "__main__":
    # Load model
    device = torch.device(f"cuda:{settings.GPU_DEVICE_ID}" if settings.USE_GPU and torch.cuda.is_available() else "cpu")
    
    model = VideoRecommender(
        num_videos=settings.VIDEO_DATASET_SIZE,
        num_users=1000,
        embedding_dim=settings.EMBEDDING_DIM,
        num_heads=settings.NUM_HEADS,
        num_layers=settings.NUM_LAYERS
    )
    
    # Load checkpoint
    checkpoint = torch.load(settings.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Generate test data
    from training.train import generate_synthetic_data, VideoInteractionDataset
    
    _, test_data = generate_synthetic_data()
    test_dataset = VideoInteractionDataset(test_data, settings.VIDEO_DATASET_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate hit rates
    print("Evaluating hit rates...")
    hit_rates = evaluate_hit_rate(model, test_loader, k_values=[10, 20, 50], device=device)
    print("Hit Rates:", hit_rates)
    
    # Evaluate latency
    print("\nEvaluating inference latency...")
    latencies = evaluate_latency(model, num_samples=1000, device=device)
    
    for batch_size, stats in latencies.items():
        print(f"\n{batch_size}:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.2f}")
    
    # Plot results
    plot_metrics(hit_rates, latencies)