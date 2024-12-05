import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model parameters
latent_dim = 16  # Latent space dimension
hidden_dim = 64  # LSTM hidden layer dimension
sequence_length = 1000  # Number of time steps in each trajectory
num_pedestrians = 50  # Number of pedestrians
feature_dim = 2  # (x, y) coordinate dimensions for each step
social_force_threshold = 0.3  # Minimum safe distance between pedestrians
wall_force_threshold = 0.3    # Minimum safe distance between pedestrians and walls

# Model: Encoder
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

# Model: Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, feature_dim, sequence_length):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, feature_dim)
        self.sequence_length = sequence_length

    def forward(self, z):
        h0 = self.fc(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
        h, _ = self.lstm(h0)
        output = self.fc_out(h)
        return output

# Model: VAE
class VAE(nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim, sequence_length):
        super(VAE, self).__init__()
        self.encoder = Encoder(feature_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, feature_dim, sequence_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# Social force constraints
def social_force(positions):
    penalty = 0.0
    for i in range(num_pedestrians):
        for j in range(i + 1, num_pedestrians):
            distance = torch.norm(positions[:, i, :] - positions[:, j, :], dim=-1)
            penalty += torch.sum(F.relu(social_force_threshold - distance) ** 2)
    return penalty

# Wall force constraints
def wall_interaction_force(positions, walls):
    """
    Calculate the penalty term for the interaction force between pedestrians and walls.
    Args:
        positions (torch.Tensor): Pedestrian positions, shape [batch_size, num_pedestrians, 2]
        walls (list): Definition of walls, each wall is [x_min, x_max, y_min, y_max]
        wall_force_threshold (float): Minimum safe distance between walls and pedestrians
    Returns:
        penalty (float): Penalty term for wall interaction
    """
    penalty = 0.0
    batch_size, num_pedestrians, _ = positions.shape

    for wall in walls:
        # Wall start and end points
        wall_start = torch.tensor([wall[0], wall[2]], dtype=torch.float32)  # (x_min, y_min)
        wall_end = torch.tensor([wall[1], wall[3]], dtype=torch.float32)    # (x_max, y_max)

        # Wall vector
        wall_vector = wall_end - wall_start
        wall_length_squared = torch.dot(wall_vector, wall_vector)  # Length squared of the wall

        for pos in positions:  # Iterate over pedestrian positions per frame
            for pedestrian_pos in pos:  # Iterate over each pedestrian
                # Vector from pedestrian to wall start point
                point_vector = pedestrian_pos - wall_start

                # Projection ratio
                proj_len = torch.dot(point_vector, wall_vector) / wall_length_squared

                # Skip if the projection point is outside the wall segment
                if proj_len < 0.0 or proj_len > 1.0:
                    continue

                # Projection point
                proj_point = wall_start + proj_len * wall_vector

                # Distance to the wall
                distance_to_wall = torch.norm(pedestrian_pos - proj_point)

                # Add penalty if distance is less than the threshold
                if distance_to_wall < wall_force_threshold:
                    penalty += F.relu(wall_force_threshold - distance_to_wall) ** 2

    return penalty

# Loss function
def vae_loss(recon_x, x, mu, logvar, walls):
    recon_loss = F.mse_loss(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    social_penalty = social_force(recon_x)
    wall_penalty = wall_interaction_force(recon_x, walls)
    total_loss = recon_loss + kl_div + social_penalty + wall_penalty
    return total_loss

# Wall parameters
walls = [
    [-4, -4, -4, 4], [-4, 4, 4, 4], [-4, 4, -4, -4],
    [4, 4, 1, 4], [4, 4, -4, -1]
]

# Load data
simulated_data = np.load('state record.npy', allow_pickle=True)
simulated_data = [arr[:, :2] for arr in simulated_data]
data_array = np.stack(simulated_data, axis=1)
data_tensor = torch.tensor(data_array, dtype=torch.float32)

# Model and optimizer
vae = VAE(feature_dim=feature_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, sequence_length=sequence_length)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    vae.train()
    optimizer.zero_grad()
    recon_data, mu, logvar = vae(data_tensor)
    loss = vae_loss(recon_data, data_tensor, mu, logvar, walls)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Generate new trajectories
vae.eval()
with torch.no_grad():
    z = torch.randn((num_pedestrians, latent_dim))
    generated_data = vae.decoder(z)
    print("Generated data shape:", generated_data.shape)  # [50, 1000, 2]

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

num_pedestrians = 50
sequence_length = 1000
feature_dim = 2

# Obstacles
walls = [
    [-4, -4, -4, 4],    # Left wall
    [-4, 4, 4, 4],      # Top wall
    [-4, 4, -4, -4],    # Bottom wall
    [4, 4, 1, 4],       # Upper right wall
    [4, 4, -4, -1]      # Lower right wall
]

# Animation function
def create_animation(data, walls, interval=50):
    """
    Create trajectory animation, showing pedestrian trajectories and obstacles.
    Args:
        data (numpy.ndarray): Pedestrian trajectory data, shape [num_pedestrians, sequence_length, feature_dim]
        walls (list): Obstacles definition, each obstacle represented by [x1, y1, x2, y2]
        interval (int): Time interval (in milliseconds) between frames
    """
    num_pedestrians, sequence_length, feature_dim = data.shape
    assert feature_dim == 2, "Feature dimension must be 2 (x, y coordinates)."

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("Pedestrian Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Draw obstacles
    for wall in walls:
        ax.plot([wall[0], wall[1]], [wall[2], wall[3]], color='red', linewidth=2, label="Obstacle" if wall == walls[0] else "")

    # Initialize pedestrian positions as scatter plot
    scatter = ax.scatter([], [], s=50, c='blue', label="Pedestrians")
    ax.legend()

    # Update function
    def update(frame):
        positions = data[:, frame, :]  # Positions of all pedestrians at the current frame
        scatter.set_offsets(positions)
        return scatter,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=sequence_length, interval=interval, blit=True)
    plt.close(fig)  # Prevent static plot from displaying
    return ani

# Create animation
ani = create_animation(generated_data, walls)

# Save animation as MP4 file
animation_path = "pedestrian_trajectories_with_obstacles.mp4"
ani.save(animation_path, fps=30, extra_args=['-vcodec', 'libx264'])
print(f"Animation saved at {animation_path}")
