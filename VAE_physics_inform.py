import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Wall interaction force constraint
def wall_interaction_force(positions, walls, threshold=0.3):
    penalty = 0.0
    for wall in walls:
        wall_start = torch.tensor([wall[0], wall[2]], dtype=torch.float32).to(positions.device)
        wall_end = torch.tensor([wall[1], wall[3]], dtype=torch.float32).to(positions.device)
        wall_vector = wall_end - wall_start
        wall_length_squared = torch.dot(wall_vector, wall_vector)

        for pedestrian_pos in positions:
            point_vector = pedestrian_pos - wall_start
            proj_len = torch.dot(point_vector, wall_vector) / wall_length_squared
            proj_point = wall_start + proj_len * wall_vector if 0 <= proj_len <= 1 else None
            if proj_point is not None:
                distance_to_wall = torch.norm(pedestrian_pos - proj_point)
                penalty += F.relu(threshold - distance_to_wall) ** 2

    return penalty


# Position Encoder
class PositionEncoder(nn.Module):
    def __init__(self, position_dim, hidden_dim):
        super(PositionEncoder, self).__init__()
        self.fc_position = nn.Linear(position_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, position):
        x = F.relu(self.fc_position(position))
        encoded_position = F.relu(self.fc_output(x))
        return encoded_position


# LSTM Encoder for VAE
class Encoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(feature_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, encoded_position):
        batch_size, seq_len, _ = x.shape
        encoded_position = encoded_position.unsqueeze(1).repeat(1, seq_len, 1)
        x = torch.cat([x, encoded_position], dim=-1)  # Combine input and position encoding
        _, (h_n, _) = self.lstm(x)  # h_n: [1, batch_size, hidden_dim]
        h_n = h_n.squeeze(0)  # Remove first dimension
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar


# LSTM Decoder for VAE
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, feature_dim, sequence_length):
        super(Decoder, self).__init__()
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, feature_dim)
        self.sequence_length = sequence_length

    def forward(self, z):
        h0 = self.fc_latent(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
        h, _ = self.lstm(h0)
        output = self.fc_out(h)
        return output


# VAE Model with Position Encoder
class VAE(nn.Module):
    def __init__(self, feature_dim, hidden_dim, latent_dim, sequence_length):
        super(VAE, self).__init__()
        self.position_encoder = PositionEncoder(position_dim=feature_dim, hidden_dim=hidden_dim)
        self.encoder = Encoder(feature_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, feature_dim, sequence_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, initial_position):
        encoded_position = self.position_encoder(initial_position)
        mu, logvar = self.encoder(x, encoded_position)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


# Loss function
def trajectory_loss(recon_x, target, mu, logvar, walls, lambda_wall=1.0):
    recon_loss = F.mse_loss(recon_x, target, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    wall_penalty = wall_interaction_force(recon_x.view(-1, recon_x.shape[-1]), walls)
    return recon_loss + kld_loss + lambda_wall * wall_penalty


# Data loading function
def load_single_pedestrian_trajectories(file_path):
    data = np.load(file_path, allow_pickle=True)
    data = data[..., :2]  # Extract x, y positions
    all_trajectories = torch.tensor(data, dtype=torch.float32)
    return all_trajectories


# Main program
if __name__ == "__main__":
    # Parameter definitions
    feature_dim = 2
    hidden_dim = 64
    latent_dim = 16
    sequence_length = 1000
    num_epochs = 20
    lambda_wall = 1.0

    # Wall definitions
    walls = [
        [-4, -4, -4, 4],  # Left wall
        [-4, 4, 4, 4],    # Top wall
        [-4, 4, -4, -4],  # Bottom wall
        [4, 4, -4, 4],    # Right wall
    ]

    # Sample file paths (5 samples)
    file_paths = [f"state_record_batch_{i}.npy" for i in range(1, 6)]

    # Initialize model and optimizer
    vae_model = VAE(feature_dim, hidden_dim, latent_dim, sequence_length)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001)

    # Start training
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs} started...")

        for file_path in file_paths:
            try:
                all_trajectories = load_single_pedestrian_trajectories(file_path)
                for pedestrian_idx in range(all_trajectories.shape[1]):
                    optimizer.zero_grad()
                    pedestrian_trajectory = all_trajectories[:, pedestrian_idx, :]
                    initial_position = pedestrian_trajectory[0:1, :]
                    input_trajectory = pedestrian_trajectory.unsqueeze(0)

                    recon_trajectory, mu, logvar = vae_model(input_trajectory, initial_position)
                    loss = trajectory_loss(recon_trajectory, input_trajectory, mu, logvar, walls, lambda_wall)

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

        print(f"Epoch {epoch + 1}/{num_epochs} completed. Total Loss: {total_loss:.4f}")

    # Generate trajectories
    print("Generating sample trajectories...")
    num_pedestrians = 50
    initial_positions = torch.tensor(
        np.random.uniform(-3.5, 3.5, size=(num_pedestrians, feature_dim)), dtype=torch.float32
    )
    generated_trajectories = []

    for i in range(num_pedestrians):
        initial_position = initial_positions[i].unsqueeze(0).unsqueeze(0)  # [1, 1, feature_dim]
        recon_trajectory, _, _ = vae_model(initial_position, initial_positions[i].unsqueeze(0))
        generated_trajectories.append(recon_trajectory.squeeze(0))

    generated_trajectories = torch.stack(generated_trajectories, dim=0)
    print(f"Generated trajectory shape: {generated_trajectories.shape}")

    np.save("generated_trajectories.npy", generated_trajectories.detach().numpy())
    print("Generated trajectories saved to 'generated_trajectories.npy'")


generated_trajectories = generated_trajectories.detach().numpy()


import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Animation function
def create_animation(data, walls, interval=50):
    """
    Create a trajectory animation showing pedestrian trajectories and obstacles.
    Args:
        data (numpy.ndarray): Pedestrian trajectory data, shape [num_pedestrians, sequence_length, feature_dim]
        walls (list): Wall definitions, each wall is represented as [x1, y1, x2, y2]
        interval (int): Frame time interval (milliseconds)
    """
    num_pedestrians, sequence_length, feature_dim = data.shape
    assert feature_dim == 2, "Feature dimension must be 2 (x, y coordinates)."

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("Pedestrian Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Plot obstacles
    for wall in walls:
        ax.plot([wall[0], wall[1]], [wall[2], wall[3]], color='red', linewidth=2, label="Obstacle" if wall == walls[0] else "")

    # Initialize scatter plot for pedestrian positions
    scatter = ax.scatter([], [], s=50, c='blue', label="Pedestrians")
    ax.legend()

    # Update function
    def update(frame):
        positions = data[:, frame, :]  # Positions of all pedestrians at current frame
        scatter.set_offsets(positions)
        return scatter,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=sequence_length, interval=interval, blit=True)
    plt.close(fig)  # Prevent displaying static image
    return ani



# Wall definitions
walls = [
    [-4, -4, -4, 4],    # Left wall
    [-4, 4, 4, 4],      # Top wall
    [-4, 4, -4, -4],    # Bottom wall
    [4, 4, 1, 4],       # Upper right wall
    [4, 4, -4, -1]      # Lower right wall
]


def plot_training_loss(losses):
    """
    Plot the training loss curve.
    Args:
        losses (list): Training losses for each epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_curve.png")
    print("Training loss curve saved as training_loss_curve.png")
    plt.show()
    
    
# Create animation
ani = create_animation(generated_trajectories, walls)

# Save animation as MP4 file
animation_path = "pedestrian_trajectories_with_obstacles.mp4"
ani.save(animation_path, fps=30, extra_args=['-vcodec', 'libx264'])
print(f"Animation saved at {animation_path}")






