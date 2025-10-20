import os
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from vae_model import VariationalAutoEncoder

plt.rcParams['figure.dpi'] = 100

# Device configuration
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
T.manual_seed(42)


# ===========================
# Dataset
# ===========================
class CrafterFrameDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path)  # shape (N, 4, 84, 84)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        return T.tensor(x, dtype=T.float32)  # no label


# ===========================
# VAE Loss
# ===========================
def vae_loss(true: T.Tensor, pred: T.Tensor, mu: T.Tensor, log_var: T.Tensor, beta: float=1.0) -> T.Tensor:
    """Compute total VAE loss = reconstruction + KL divergence."""
    kld_loss = -0.5 * T.sum(1 + log_var - mu.pow(2) - log_var.exp())
    reconstruction_loss = ((true - pred)**2).sum() 
    return reconstruction_loss + beta * kld_loss


# ===========================
# Training
# ===========================
def train(
    dataloader: DataLoader,
    model: VariationalAutoEncoder,
    optimizer: optim.Optimizer,
    hyper_params: dict,
    device: str = "cpu"
):
    """Train the VAE on the given dataloader."""
    n_epochs = hyper_params["max_epochs"]
    beta = hyper_params["beta"]
    save_path = hyper_params["save_dir"]
    display_every = hyper_params["display_every"]

    os.makedirs(save_path, exist_ok=True)
    all_losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []

        for x in dataloader:  # <-- no labels, just x
            x = x.to(device)

            recon, mu, log_var = model(x)
            loss = vae_loss(x, recon, mu, log_var, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        all_losses.append(mean_loss)

        tqdm.write(f"Epoch [{epoch}/{n_epochs}] — Loss: {mean_loss:.4f}")
        model.save(save_path)

        # Display reconstruction
        if epoch % display_every == 0:
            visualize_reconstruction(model, dataloader, device)

    plot_loss_curve(all_losses)


# ===========================
# Visualization
# ===========================
def visualize_reconstruction(model: VariationalAutoEncoder, dataloader: DataLoader, device: str):
    """Display a few input and reconstructed images."""
    model.eval()
    with T.no_grad():
        x = next(iter(dataloader)).to(device)
        recon, _, _ = model(x)

    x = x.cpu().numpy()
    recon = recon.cpu().numpy()

    num_plots = 5
    fig, axes = plt.subplots(2, num_plots, figsize=(12, 4))
    for i in range(num_plots):
        # Average over frame stack to show one grayscale image
        axes[0, i].imshow(np.mean(x[i], axis=0), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        axes[1, i].imshow(np.mean(recon[i], axis=0), cmap='gray')
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()


# ===========================
# Plot Loss Curve
# ===========================
def plot_loss_curve(losses: list[float]):
    """Plot training loss curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="VAE Loss")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("VAE Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# ===========================
# Main
# ===========================
def main():
    batch_size = 64
    dataset_path = "./scripts/ppo/vae/data/CustomCrafterReward-v1_dataset.npy"

    # Load dataset
    dataset = CrafterFrameDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    hyper_params = {
        "lr": 1e-4,
        "input_dims": (4, 64, 64),
        "latent_dims": 128, 
        "beta": 1e-2,
        "max_epochs": 80,
        "display_every": 10,
        "save_dir": "./scripts/ppo/models"
    }

    model = VariationalAutoEncoder(
        input_dims=hyper_params["input_dims"],
        latent_dim=hyper_params["latent_dims"],
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyper_params["lr"])
    train(loader, model, optimizer, hyper_params, device)


if __name__ == "__main__":
    main()
