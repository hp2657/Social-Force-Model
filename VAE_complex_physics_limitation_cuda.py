import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Model parameters
latent_dim = 16  # Latent space dimension
hidden_dim = 64  # LSTM hidden layer dimension
sequence_length = 1000  # Time steps per trajectory
num_pedestrians = 50  # Number of pedestrians
feature_dim = 2  # Dimension of each step's (x, y) coordinates
social_force_threshold = 0.3  # Minimum safety distance between pedestrians
wall_force_threshold = 0.3    # Minimum safety distance between pedestrians and walls
delta_t = 0.01  # Time step

# Wall parameters
walls = [
    [-4, -4, -4, 4], [-4, 4, 4, 4], [-4, 4, -4, -4],
    [4, 4, 1, 4], [4, 4, -4, -1]
]

# Load data
print("Loading data...")
simulated_data = np.load('state record.npy', allow_pickle=True)
simulated_data = [arr[:, :2] for arr in simulated_data]
data_array = np.stack(simulated_data, axis=1)  # [num_pedestrians, sequence_length, feature_dim]
data_tensor = torch.tensor(data_array, dtype=torch.float32)

print(f"Data Tensor Shape: {data_tensor.shape}")
print(f"Data Tensor Min: {data_tensor.min()}, Max: {data_tensor.max()}")
print(f"Data Tensor Contains NaN: {torch.isnan(data_tensor).any()}")
print(f"Data Tensor Contains Inf: {torch.isinf(data_tensor).any()}")

# Load data to device (use CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_tensor = data_tensor.to(device)

# Encoder
class Encoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

# Decoder
class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, feature_dim, sequence_length):
        super(LSTMDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, feature_dim)
        self.sequence_length = sequence_length

    def forward(self, z, initial_positions):
        h0 = self.fc(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
        h, _ = self.lstm(h0)
        trajectory = self.fc_out(h)
        trajectory += initial_positions.unsqueeze(1)  # Add initial position offset
        return trajectory

# VAE
class VAE(nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim, sequence_length):
        super(VAE, self).__init__()
        self.encoder = Encoder(feature_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, feature_dim, sequence_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, initial_positions):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, initial_positions)
        return recon_x, mu, logvar

# Compute acceleration
def compute_acceleration(positions, delta_t):
    velocities = (positions[:, 1:, :] - positions[:, :-1, :]) / delta_t
    accelerations = (velocities[:, 1:, :] - velocities[:, :-1, :]) / delta_t
    return accelerations

# Acceleration loss
def acceleration_loss(actual_acc, expected_acc):
    return F.mse_loss(actual_acc, expected_acc)

# Compute expected acceleration (physical constraints)
def compute_expected_acceleration(positions, velocities, walls, social_force_threshold, wall_force_threshold):
    batch_size, num_pedestrians, _ = positions.shape
    accelerations = torch.zeros_like(velocities).to(device)

    # Interaction forces between pedestrians
    for i in range(num_pedestrians):
        for j in range(i + 1, num_pedestrians):
            distance = torch.norm(positions[:, i, :] - positions[:, j, :], dim=-1, keepdim=True)
            direction = positions[:, i, :] - positions[:, j, :]
            normalized_direction = direction / (distance + 1e-6)
            overlap = F.relu(social_force_threshold - distance)
            force = overlap * normalized_direction
            accelerations[:, i, :] += force
            accelerations[:, j, :] -= force

    # Interaction forces between pedestrians and walls
    for wall in walls:
        wall_start = torch.tensor([wall[0], wall[2]], dtype=torch.float32, device=device)
        wall_end = torch.tensor([wall[1], wall[3]], dtype=torch.float32, device=device)
        wall_vector = wall_end - wall_start
        wall_length_squared = torch.dot(wall_vector, wall_vector)

        for i in range(num_pedestrians):
            point_vector = positions[:, i, :] - wall_start
            proj_len = torch.sum(point_vector * wall_vector, dim=-1, keepdim=True) / wall_length_squared
            proj_len = torch.clamp(proj_len, 0.0, 1.0)
            proj_point = wall_start + proj_len * wall_vector
            distance_to_wall = torch.norm(positions[:, i, :] - proj_point, dim=-1, keepdim=True)
            direction_to_wall = positions[:, i, :] - proj_point
            normalized_direction = direction_to_wall / (distance_to_wall + 1e-6)
            overlap = F.relu(wall_force_threshold - distance_to_wall)
            force = overlap * normalized_direction
            accelerations[:, i, :] += force

    return accelerations

# Loss function
def vae_loss_with_physics(recon_x, x, mu, logvar, walls, social_force_threshold, wall_force_threshold, delta_t):
    recon_loss = F.mse_loss(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Compute actual acceleration
    actual_acc = compute_acceleration(recon_x, delta_t)

    # Compute expected acceleration
    velocities = (recon_x[:, 1:, :] - recon_x[:, :-1, :]) / delta_t
    expected_acc = compute_expected_acceleration(recon_x[:, :-2, :], velocities[:, :-1, :], walls, social_force_threshold, wall_force_threshold)

    acc_loss = acceleration_loss(actual_acc, expected_acc)
    total_loss = recon_loss + kl_div + 0.01 * acc_loss
    print(recon_loss, kl_div, 0.01 * acc_loss)
    return total_loss

# Initialize model
vae = VAE(feature_dim, hidden_dim, latent_dim, sequence_length).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

# Train model
epochs = 10
for epoch in range(epochs):
    vae.train()
    optimizer.zero_grad()

    initial_positions = data_tensor[:, 0, :]
    recon_data, mu, logvar = vae(data_tensor, initial_positions)

    loss = vae_loss_with_physics(recon_data, data_tensor, mu, logvar, walls, social_force_threshold, wall_force_threshold, delta_t)

    if torch.isnan(loss) or torch.isinf(loss):
        print("Invalid loss detected. Exiting training.")
        break

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Generate new trajectories
vae.eval()
with torch.no_grad():
    z = torch.randn((num_pedestrians, latent_dim), device=device)
    initial_positions = data_tensor[:, 0, :]
    generated_data = vae.decoder(z, initial_positions)
    print("Generated data shape:", generated_data.shape)  # [50, 1000, 2]
