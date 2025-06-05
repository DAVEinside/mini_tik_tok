"""Preprocess videos and extract features for recommendation system"""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from pathlib import Path
import json
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoPreprocessor:
    """Extract features from videos for recommendation"""
    
    def __init__(
        self,
        video_dir: str = "data/videos/",
        output_dir: str = "data/processed/",
        device: str = "cuda"
    ):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize feature extractors
        self._init_models()
        
        # Video preprocessing settings
        self.target_fps = 1  # Extract 1 frame per second
        self.target_size = (224, 224)
        
    def _init_models(self):
        """Initialize pretrained models for feature extraction"""
        # For demo, we'll use ResNet for visual features
        # In production, you might use:
        # - TimeSformer or VideoMAE for video understanding
        # - CLIP for multimodal features
        # - VGGish for audio features
        
        from torchvision.models import resnet50, ResNet50_Weights
        
        self.visual_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.visual_model = torch.nn.Sequential(*list(self.visual_model.children())[:-1])
        self.visual_model.to(self.device)
        self.visual_model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frames(self, video_path: Path, max_frames: int = 32) -> List[np.ndarray]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        if fps > 0:
            # Extract frames at target_fps
            frame_interval = int(fps / self.target_fps)
            frame_indices = list(range(0, total_frames, frame_interval))[:max_frames]
        else:
            # Fallback: extract evenly spaced frames
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def extract_visual_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract visual features from frames using pretrained model"""
        if not frames:
            return np.zeros((2048,))
        
        # Process frames
        frame_tensors = []
        for frame in frames:
            tensor = self.transform(frame)
            frame_tensors.append(tensor)
        
        # Batch process
        batch = torch.stack(frame_tensors).to(self.device)
        
        with torch.no_grad():
            features = self.visual_model(batch)
            features = features.squeeze(-1).squeeze(-1)
        
        # Average pool features across frames
        avg_features = features.mean(dim=0).cpu().numpy()
        
        return avg_features
    
    def extract_audio_features(self, video_path: Path) -> np.ndarray:
        """Extract audio features from video"""
        # Simplified: return random features
        # In production, use VGGish, wav2vec2, or similar
        return np.random.randn(128)
    
    def extract_text_features(self, metadata: Dict) -> np.ndarray:
        """Extract text features from title and description"""
        # Simplified: return random features
        # In production, use BERT, RoBERTa, or similar
        return np.random.randn(768)
    
    def process_video(self, video_id: int, video_path: Path, metadata: Dict) -> Dict:
        """Process a single video and extract all features"""
        logger.info(f"Processing video {video_id}: {video_path}")
        
        features = {
            'video_id': video_id,
            'path': str(video_path),
            'metadata': metadata
        }
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            
            if frames:
                # Extract visual features
                visual_features = self.extract_visual_features(frames)
                features['visual_features'] = visual_features.tolist()
                
                # Extract audio features
                audio_features = self.extract_audio_features(video_path)
                features['audio_features'] = audio_features.tolist()
                
                # Extract text features
                text_features = self.extract_text_features(metadata)
                features['text_features'] = text_features.tolist()
                
                features['success'] = True
            else:
                features['success'] = False
                features['error'] = "No frames extracted"
                
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            features['success'] = False
            features['error'] = str(e)
        
        return features
    
    def process_dataset(self, video_metadata_path: str = "data/processed/video_metadata.json"):
        """Process all videos in the dataset"""
        # Load metadata
        with open(video_metadata_path, 'r') as f:
            video_metadata = json.load(f)
        
        # For demo, we'll generate synthetic features
        logger.info("Generating synthetic video features for demo...")
        
        all_features = {}
        feature_matrix = []
        
        for video_id, metadata in tqdm(video_metadata.items()):
            # In production, you would process actual video files
            # For demo, generate features based on categories
            
            visual_features = np.random.randn(2048)
            audio_features = np.random.randn(128)
            text_features = np.random.randn(768)
            
            # Make features category-specific
            for cat in metadata['categories']:
                cat_idx = hash(cat) % 100
                visual_features[cat_idx * 20:(cat_idx + 1) * 20] += 1.0
            
            # Normalize
            visual_features = visual_features / np.linalg.norm(visual_features)
            
            # Combine features
            combined_features = np.concatenate([
                visual_features,
                audio_features,
                text_features
            ])
            
            all_features[video_id] = {
                'visual': visual_features.tolist(),
                'audio': audio_features.tolist(),
                'text': text_features.tolist(),
                'combined': combined_features.tolist()
            }
            
            feature_matrix.append(combined_features)
        
        # Save features
        feature_matrix = np.array(feature_matrix)
        np.save(self.output_dir / "video_features_full.npy", feature_matrix)
        
        with open(self.output_dir / "video_features_detailed.json", 'w') as f:
            json.dump(all_features, f)
        
        logger.info(f"Processed {len(video_metadata)} videos")
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Save feature statistics
        stats = {
            "num_videos": len(video_metadata),
            "feature_dim": feature_matrix.shape[1],
            "visual_dim": 2048,
            "audio_dim": 128,
            "text_dim": 768,
            "mean_norm": float(np.mean(np.linalg.norm(feature_matrix, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(feature_matrix, axis=1)))
        }
        
        with open(self.output_dir / "feature_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess videos for recommendation")
    parser.add_argument("--video-dir", type=str, default="data/videos/", help="Video directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    preprocessor = VideoPreprocessor(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    preprocessor.process_dataset()

if __name__ == "__main__":
    main()