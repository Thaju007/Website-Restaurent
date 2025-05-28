import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: BASIC WGAN FOR TABULAR DATA
# =============================================================================

class BasicGenerator(nn.Module):
    """Basic Generator for tabular data"""
    def __init__(self, noise_dim, hidden_dims, output_dim):
        super(BasicGenerator, self).__init__()
        
        layers = []
        prev_dim = noise_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Tanh for normalized data
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, noise):
        return self.model(noise)

class BasicCritic(nn.Module):
    """Basic Critic (Discriminator) for WGAN"""
    def __init__(self, input_dim, hidden_dims):
        super(BasicCritic, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for WGAN critic)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class BasicWGAN:
    """Basic WGAN for tabular data generation"""
    
    def __init__(self, data_dim, noise_dim=100, generator_dims=[128, 128], 
                 critic_dims=[128, 128], lr=0.0002, device='cpu'):
        self.device = device
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        
        # Initialize networks
        self.generator = BasicGenerator(noise_dim, generator_dims, data_dim).to(device)
        self.critic = BasicCritic(data_dim, critic_dims).to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.c_optimizer = optim.Adam(self.critic.parameters(), lr=lr, betas=(0.5, 0.9))
        
        # Training history
        self.losses = {'generator': [], 'critic': []}
    
    def gradient_penalty(self, real_data, fake_data, lambda_gp=10):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        
        # Random interpolation factor
        alpha = torch.rand(batch_size, 1).to(self.device)
        
        # Interpolated samples
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Critic output for interpolated samples
        critic_interpolated = self.critic(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gradient penalty
        gradients_norm = gradients.norm(2, dim=1)
        penalty = ((gradients_norm - 1) ** 2).mean() * lambda_gp
        
        return penalty
    
    def train_step(self, real_data, critic_steps=5):
        """Single training step"""
        batch_size = real_data.size(0)
        
        # Train Critic
        for _ in range(critic_steps):
            # Real data
            real_validity = self.critic(real_data)
            
            # Fake data
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_data = self.generator(noise)
            fake_validity = self.critic(fake_data.detach())
            
            # Gradient penalty
            gp = self.gradient_penalty(real_data, fake_data)
            
            # Critic loss (Wasserstein loss + gradient penalty)
            c_loss = fake_validity.mean() - real_validity.mean() + gp
            
            # Update critic
            self.c_optimizer.zero_grad()
            c_loss.backward()
            self.c_optimizer.step()
        
        # Train Generator
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_data = self.generator(noise)
        fake_validity = self.critic(fake_data)
        
        # Generator loss
        g_loss = -fake_validity.mean()
        
        # Update generator
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), c_loss.item()
    
    def generate_samples(self, num_samples):
        """Generate synthetic samples"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_dim).to(self.device)
            samples = self.generator(noise)
        self.generator.train()
        return samples.cpu().numpy()

# =============================================================================
# STEP 2: DATA PREPROCESSING FOR TABULAR DATA
# =============================================================================

class TabularDataPreprocessor:
    """Handles preprocessing of tabular data for GAN training"""
    
    def __init__(self):
        self.numerical_scaler = StandardScaler()
        self.categorical_encoders = {}
        self.column_info = {}
        self.fitted = False
    
    def fit(self, data, categorical_columns=None):
        """Fit the preprocessor on data"""
        if categorical_columns is None:
            categorical_columns = []
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = [col for col in data.columns if col not in categorical_columns]
        
        # Store column information
        self.column_info = {
            'categorical': categorical_columns,
            'numerical': self.numerical_columns,
            'total_dim': 0
        }
        
        # Fit numerical scaler
        if self.numerical_columns:
            self.numerical_scaler.fit(data[self.numerical_columns])
            self.column_info['numerical_dim'] = len(self.numerical_columns)
        else:
            self.column_info['numerical_dim'] = 0
        
        # Fit categorical encoders
        categorical_dims = 0
        for col in categorical_columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(data[[col]])
            self.categorical_encoders[col] = encoder
            categorical_dims += len(encoder.categories_[0])
        
        self.column_info['categorical_dim'] = categorical_dims
        self.column_info['total_dim'] = self.column_info['numerical_dim'] + categorical_dims
        
        self.fitted = True
        
        print(f"Data preprocessing fitted:")
        print(f"  - Numerical columns: {len(self.numerical_columns)} → {self.column_info['numerical_dim']} dims")
        print(f"  - Categorical columns: {len(categorical_columns)} → {categorical_dims} dims")
        print(f"  - Total dimensions: {self.column_info['total_dim']}")
    
    def transform(self, data):
        """Transform data to GAN input format"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        transformed_parts = []
        
        # Transform numerical columns
        if self.numerical_columns:
            numerical_data = self.numerical_scaler.transform(data[self.numerical_columns])
            transformed_parts.append(numerical_data)
        
        # Transform categorical columns
        for col in self.categorical_columns:
            categorical_data = self.categorical_encoders[col].transform(data[[col]])
            transformed_parts.append(categorical_data)
        
        # Concatenate all parts
        if transformed_parts:
            return np.concatenate(transformed_parts, axis=1)
        else:
            return np.array([]).reshape(len(data), 0)
    
    def inverse_transform(self, transformed_data):
        """Convert GAN output back to original format"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        result_parts = {}
        current_idx = 0
        
        # Inverse transform numerical columns
        if self.numerical_columns:
            numerical_dim = self.column_info['numerical_dim']
            numerical_data = transformed_data[:, current_idx:current_idx + numerical_dim]
            numerical_original = self.numerical_scaler.inverse_transform(numerical_data)
            
            for i, col in enumerate(self.numerical_columns):
                result_parts[col] = numerical_original[:, i]
            
            current_idx += numerical_dim
        
        # Inverse transform categorical columns
        for col in self.categorical_columns:
            encoder = self.categorical_encoders[col]
            categorical_dim = len(encoder.categories_[0])
            categorical_data = transformed_data[:, current_idx:current_idx + categorical_dim]
            
            # Convert one-hot back to categories
            categorical_original = encoder.inverse_transform(categorical_data)
            result_parts[col] = categorical_original.ravel()
            
            current_idx += categorical_dim
        
        return pd.DataFrame(result_parts)

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def create_sample_data():
    """Create sample tabular data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic data
    data = {
        'age': np.random.normal(35, 10, n_samples).clip(18, 80),
        'income': np.random.lognormal(10, 0.5, n_samples).clip(20000, 200000),
        'score': np.random.beta(2, 5, n_samples) * 100,
        'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
        'status': np.random.choice(['active', 'inactive'], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

def train_basic_wgan_example():
    """Example of training basic WGAN on tabular data"""
    print("=== STEP 1: BASIC WGAN TRAINING ===")
    
    # Create sample data
    data = create_sample_data()
    print(f"Created sample data: {data.shape}")
    print(data.head())
    print("\nData types:")
    print(data.dtypes)
    
    # Preprocess data
    preprocessor = TabularDataPreprocessor()
    categorical_cols = ['category', 'status']
    
    preprocessor.fit(data, categorical_columns=categorical_cols)
    transformed_data = preprocessor.transform(data)
    
    print(f"\nTransformed data shape: {transformed_data.shape}")
    
    # Convert to tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    real_data_tensor = torch.FloatTensor(transformed_data).to(device)
    
    # Initialize WGAN
    wgan = BasicWGAN(
        data_dim=transformed_data.shape[1],
        noise_dim=100,
        generator_dims=[128, 128],
        critic_dims=[128, 128],
        device=device
    )
    
    print(f"\nInitialized WGAN:")
    print(f"  - Data dimension: {transformed_data.shape[1]}")
    print(f"  - Noise dimension: 100")
    print(f"  - Device: {device}")
    
    # Training parameters
    epochs = 100
    batch_size = 64
    
    print(f"\nStarting training for {epochs} epochs...")
    
    # Training loop
    for epoch in range(epochs):
        epoch_g_losses = []
        epoch_c_losses = []
        
        # Create batches
        n_batches = len(real_data_tensor) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_real = real_data_tensor[start_idx:end_idx]
            
            g_loss, c_loss = wgan.train_step(batch_real)
            epoch_g_losses.append(g_loss)
            epoch_c_losses.append(c_loss)
        
        # Record losses
        avg_g_loss = np.mean(epoch_g_losses)
        avg_c_loss = np.mean(epoch_c_losses)
        wgan.losses['generator'].append(avg_g_loss)
        wgan.losses['critic'].append(avg_c_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: G_loss={avg_g_loss:.4f}, C_loss={avg_c_loss:.4f}")
    
    # Generate samples
    print("\n=== GENERATING SYNTHETIC DATA ===")
    synthetic_samples = wgan.generate_samples(500)
    synthetic_df = preprocessor.inverse_transform(synthetic_samples)
    
    print("Generated synthetic data:")
    print(synthetic_df.head())
    print("\nSynthetic data statistics:")
    print(synthetic_df.describe())
    
    print("\nOriginal data statistics:")
    print(data.describe())
    
    return wgan, preprocessor, data, synthetic_df

if __name__ == "__main__":
    # Run the basic WGAN example
    wgan, preprocessor, original_data, synthetic_data = train_basic_wgan_example()
    
    print("\n" + "="*50)
    print("BASIC WGAN TRAINING COMPLETED!")
    print("="*50)
    print("\nNext steps:")
    print("1. Analyze the quality of generated data")
    print("2. Add conditional layer for categorical control")
    print("3. Implement CTGAN conditional sampling")
    print("4. Compare results")
