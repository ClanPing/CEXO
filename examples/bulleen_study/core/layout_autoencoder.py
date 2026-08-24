#!/usr/bin/env python3
"""
Layout Autoencoder Module
=========================

Implements PyTorch autoencoder for learning latent behavioral descriptors
from construction site layouts. The autoencoder learns a compressed representation
of layouts that can be used as behavioral descriptors in MAP-Elites.

Key Features:
- Encodes layouts into fixed-dimensional vector representations
- Learns meaningful latent space through reconstruction
- Can be trained periodically on the evolving population
- Provides learned behavioral descriptors for quality-diversity
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from .config import FACILITY_TYPES


# =============================================================================
# REPRODUCIBILITY UTILITIES
# =============================================================================

def set_random_seeds(seed: int):
    """
    Set all random seeds for reproducibility.
    
    This ensures that:
    - Python's random module is seeded
    - NumPy's random generator is seeded
    - PyTorch's random generator is seeded (CPU and GPU)
    - PyTorch operations are deterministic
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional determinism for PyTorch >= 1.8
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_generator(seed: int) -> torch.Generator:
    """
    Create a PyTorch generator with specific seed for DataLoader.
    
    Args:
        seed: Random seed value
    
    Returns:
        Seeded PyTorch generator
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# =============================================================================
# LAYOUT ENCODING UTILITIES
# =============================================================================

def encode_layout_to_vector(facilities: List[Dict], 
                           entrances: List[Tuple[float, float]] = None,
                           normalize: bool = True) -> np.ndarray:
    """
    Convert a layout into a fixed-dimensional vector representation.
    
    Encoding scheme:
    - For each facility type: [count_ratio, mean_x, mean_y, spread_x, spread_y, rotated_ratio]
    - Entrance information: [num_entrances, avg_x, avg_y]
    - Global statistics: [facility_count, centroid_x, centroid_y, avg_spacing]
    
    Args:
        facilities: List of facility dictionaries
        entrances: Optional entrance positions
        normalize: Whether to normalize coordinates to [0, 1]
    
    Returns:
        Fixed-size numpy array encoding the layout
    """
    # Initialize encoding vector
    features_per_type = 6
    encoding_size = len(FACILITY_TYPES) * features_per_type + 3 + 4  # facilities + entrances + global
    encoding = np.zeros(encoding_size, dtype=np.float32)
    
    idx = 0
    
    # Encode each facility type as aggregate statistics so repeated broad
    # classes such as storage or core remain visible to the autoencoder.
    facilities_by_type = {ftype: [] for ftype in FACILITY_TYPES}
    for facility in facilities:
        facilities_by_type.setdefault(facility["type"], []).append(facility)
    
    for ftype in FACILITY_TYPES:
        typed_facilities = facilities_by_type.get(ftype, [])
        if typed_facilities:
            positions = np.array([f["center"] for f in typed_facilities], dtype=np.float32)
            rotations = np.array(
                [1.0 if int(f.get("rotation", 0)) % 180 == 90 else 0.0 for f in typed_facilities],
                dtype=np.float32,
            )
            mean_pos = np.mean(positions, axis=0)
            spread = np.std(positions, axis=0) if len(typed_facilities) > 1 else np.array([0.0, 0.0])
            encoding[idx] = len(typed_facilities) / max(1, len(facilities))
            encoding[idx + 1] = mean_pos[0]
            encoding[idx + 2] = mean_pos[1]
            encoding[idx + 3] = spread[0]
            encoding[idx + 4] = spread[1]
            encoding[idx + 5] = float(np.mean(rotations))
        else:
            encoding[idx:idx + features_per_type] = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
        idx += features_per_type
    
    # Encode entrance information
    if entrances and len(entrances) > 0:
        entrance_array = np.array(entrances)
        encoding[idx] = len(entrances) / 5.0  # Normalize by max expected entrances
        encoding[idx + 1] = np.mean(entrance_array[:, 0])
        encoding[idx + 2] = np.mean(entrance_array[:, 1])
    else:
        encoding[idx:idx + 3] = [0.0, 0.5, 0.5]
    idx += 3
    
    # Global statistics
    if len(facilities) > 0:
        positions = np.array([f["center"] for f in facilities])
        centroid = np.mean(positions, axis=0)
        encoding[idx] = min(len(facilities) / 20.0, 1.0)
        encoding[idx + 1] = centroid[0]
        encoding[idx + 2] = centroid[1]
        
        # Average spacing (mean pairwise distance)
        if len(facilities) > 1:
            distances = []
            for i in range(len(facilities)):
                for j in range(i + 1, len(facilities)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            encoding[idx + 3] = np.mean(distances)
        else:
            encoding[idx + 3] = 0.0
    else:
        encoding[idx:idx + 4] = [0.0, 0.5, 0.5, 0.0]
    
    return encoding


def encode_layout_batch(layouts: List[Tuple[List[Dict], List[Tuple[float, float]]]]) -> np.ndarray:
    """
    Encode a batch of layouts into a matrix.
    
    Args:
        layouts: List of (facilities, entrances) tuples
    
    Returns:
        Numpy array of shape (batch_size, encoding_dim)
    """
    encoded_layouts = []
    for facilities, entrances in layouts:
        encoded = encode_layout_to_vector(facilities, entrances)
        encoded_layouts.append(encoded)
    
    return np.array(encoded_layouts, dtype=np.float32)


# =============================================================================
# AUTOENCODER ARCHITECTURE
# =============================================================================

class LayoutEncoder(nn.Module):
    """Encoder network: Layout vector -> Latent representation"""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = None):
        super(LayoutEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, latent_dim))
        layers.append(nn.Tanh())  # Bounded latent space [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class LayoutDecoder(nn.Module):
    """Decoder network: Latent representation -> Reconstructed layout vector"""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super(LayoutDecoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1] range
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class LayoutAutoencoder(nn.Module):
    """
    Complete autoencoder for learning latent behavioral descriptors.
    
    The latent space serves as learned behavioral descriptors for MAP-Elites.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 2, 
                 encoder_hidden: List[int] = None,
                 decoder_hidden: List[int] = None):
        super(LayoutAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = LayoutEncoder(input_dim, latent_dim, encoder_hidden)
        self.decoder = LayoutDecoder(latent_dim, input_dim, decoder_hidden)
    
    def forward(self, x):
        """Full forward pass: encode and decode"""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        """Encode layouts to latent space (behavioral descriptors)"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent vectors back to layout space"""
        return self.decoder(z)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class AutoencoderTrainer:
    """Handles training of the layout autoencoder"""
    
    def __init__(self, autoencoder: LayoutAutoencoder, 
                 learning_rate: float = 0.001,
                 device: str = None,
                 seed: int = None):
        self.autoencoder = autoencoder
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.autoencoder.to(self.device)
        
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.training_history = []
        self.seed = seed
        
        # Set initial seed if provided
        if self.seed is not None:
            set_random_seeds(self.seed)
    
    def train_epoch(self, data_loader, epoch: int = 0) -> float:
        """Train for one epoch"""
        self.autoencoder.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            # DataLoader returns a list/tuple, extract the tensor
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            
            # Forward pass
            reconstructed, latent = self.autoencoder(batch)
            
            # Reconstruction loss
            loss = self.criterion(reconstructed, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.training_history.append(avg_loss)
        
        return avg_loss
    
    def train(self, layouts: List[Tuple[List[Dict], List[Tuple[float, float]]]], 
              epochs: int = 50, batch_size: int = 32, verbose: bool = True,
              training_seed: int = None) -> Dict:
        """
        Train autoencoder on a collection of layouts.
        
        Args:
            layouts: List of (facilities, entrances) tuples
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print progress
            training_seed: Specific seed for this training session (overrides instance seed)
        
        Returns:
            Dictionary with training statistics
        """
        # Use training_seed if provided, otherwise use instance seed
        seed_to_use = training_seed if training_seed is not None else self.seed
        
        # Set seeds for reproducibility
        if seed_to_use is not None:
            set_random_seeds(seed_to_use)
        
        # Encode all layouts
        layout_vectors = encode_layout_batch(layouts)
        
        # Create data loader with deterministic behavior
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(layout_vectors)
        )
        
        # Create generator for reproducible shuffling
        generator = get_generator(seed_to_use) if seed_to_use is not None else None
        
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            generator=generator,  # Ensures reproducible shuffling
            drop_last=True,  # Drop last incomplete batch (important for BatchNorm)
            worker_init_fn=lambda worker_id: np.random.seed(seed_to_use + worker_id) if seed_to_use else None
        )
        
        # Training loop
        for epoch in range(epochs):
            loss = self.train_epoch(data_loader, epoch)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return {
            "final_loss": self.training_history[-1] if self.training_history else 0.0,
            "history": self.training_history,
            "epochs": epochs,
            "num_samples": len(layouts)
        }
    
    def save_model(self, filepath: str):
        """Save autoencoder model"""
        torch.save({
            'model_state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load autoencoder model"""
        checkpoint = torch.load(filepath)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])


# =============================================================================
# LEARNED BEHAVIORAL DESCRIPTOR EXTRACTION
# =============================================================================

class LearnedBehavioralDescriptors:
    """
    Wrapper for extracting learned behavioral descriptors from layouts.
    Uses trained autoencoder to map layouts to latent behavioral space.
    """
    
    def __init__(self, autoencoder: LayoutAutoencoder, device: str = None):
        self.autoencoder = autoencoder
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.autoencoder.to(self.device)
        self.autoencoder.eval()
    
    def get_behavioral_descriptors(self, facilities: List[Dict], 
                                  entrances: List[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Extract learned behavioral descriptors from a layout.
        
        Args:
            facilities: Layout facilities
            entrances: Entrance positions
        
        Returns:
            Tuple of (bd1, bd2) in range [0, 1]
        """
        # Encode layout to vector
        layout_vector = encode_layout_to_vector(facilities, entrances)
        
        # Convert to tensor
        layout_tensor = torch.FloatTensor(layout_vector).unsqueeze(0).to(self.device)
        
        # Get latent representation
        with torch.no_grad():
            latent = self.autoencoder.encode(layout_tensor)
        
        # Convert latent to behavioral descriptors [0, 1]
        latent_np = latent.cpu().numpy()[0]
        
        # Map from [-1, 1] to [0, 1]
        bd1 = (latent_np[0] + 1.0) / 2.0
        bd2 = (latent_np[1] + 1.0) / 2.0 if len(latent_np) > 1 else 0.5
        
        # Ensure bounds
        bd1 = float(np.clip(bd1, 0.0, 1.0))
        bd2 = float(np.clip(bd2, 0.0, 1.0))
        
        return (bd1, bd2)
    
    def get_batch_descriptors(self, layouts: List[Tuple[List[Dict], List[Tuple[float, float]]]]) -> List[Tuple[float, float]]:
        """
        Extract behavioral descriptors for a batch of layouts efficiently.
        
        Args:
            layouts: List of (facilities, entrances) tuples
        
        Returns:
            List of (bd1, bd2) tuples
        """
        # Encode all layouts
        layout_vectors = encode_layout_batch(layouts)
        layout_tensor = torch.FloatTensor(layout_vectors).to(self.device)
        
        # Get latent representations
        with torch.no_grad():
            latent = self.autoencoder.encode(layout_tensor)
        
        # Convert to behavioral descriptors
        latent_np = latent.cpu().numpy()
        
        descriptors = []
        for i in range(len(latent_np)):
            bd1 = (latent_np[i, 0] + 1.0) / 2.0
            bd2 = (latent_np[i, 1] + 1.0) / 2.0 if latent_np.shape[1] > 1 else 0.5
            
            bd1 = float(np.clip(bd1, 0.0, 1.0))
            bd2 = float(np.clip(bd2, 0.0, 1.0))
            
            descriptors.append((bd1, bd2))
        
        return descriptors


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_encoding_dimension() -> int:
    """Get the fixed encoding dimension for layouts"""
    return len(FACILITY_TYPES) * 6 + 3 + 4


def create_autoencoder(latent_dim: int = 2, 
                      encoder_hidden: List[int] = None,
                      decoder_hidden: List[int] = None) -> LayoutAutoencoder:
    """
    Factory function to create a layout autoencoder.
    
    Args:
        latent_dim: Dimension of latent space (typically 2 for MAP-Elites)
        encoder_hidden: Hidden layer dimensions for encoder
        decoder_hidden: Hidden layer dimensions for decoder
    
    Returns:
        Initialized LayoutAutoencoder
    """
    input_dim = get_encoding_dimension()
    
    if encoder_hidden is None:
        encoder_hidden = [128, 64, 32]
    
    if decoder_hidden is None:
        decoder_hidden = [32, 64, 128]
    
    return LayoutAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden
    )


def visualize_latent_space(autoencoder: LayoutAutoencoder, 
                          layouts: List[Tuple[List[Dict], List[Tuple[float, float]]]],
                          labels: List[str] = None) -> None:
    """
    Visualize the latent space distribution of layouts.
    Useful for debugging and understanding learned representations.
    
    Args:
        autoencoder: Trained autoencoder
        layouts: List of (facilities, entrances) tuples
        labels: Optional labels for each layout
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    # Get learned descriptors
    bd_extractor = LearnedBehavioralDescriptors(autoencoder)
    descriptors = bd_extractor.get_batch_descriptors(layouts)
    
    # Plot
    bds = np.array(descriptors)
    plt.figure(figsize=(8, 8))
    plt.scatter(bds[:, 0], bds[:, 1], alpha=0.6, s=50)
    
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (bds[i, 0], bds[i, 1]), fontsize=8)
    
    plt.xlabel("Learned BD1")
    plt.ylabel("Learned BD2")
    plt.title("Latent Behavioral Space")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
