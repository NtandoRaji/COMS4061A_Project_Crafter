import torch as T
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, fc1_dim: int, fc2_dim: int, output_dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.out = nn.Linear(fc2_dim, output_dim)
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        z = F.relu(self.fc1(x))
        z = F.relu(self.fc2(z))
        return self.out(z)
    

class IntrinsicCuriosityModule(nn.Module):
    def __init__(
        self, input_dim: tuple[int, int, int], latent_dim: int, action_dim: int,
        encoder: nn.Module | None = None, device: str = "cpu", 
    ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.device = device

        if encoder is not None:
            self.encoder = encoder
        else:
            assert isinstance(input_dim, tuple) and len(input_dim) == 3, \
                "For images, input_dim must be (C, H, W)"
            c, w, h = input_dim
            self.encoder = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),

                # 32x32 -> 16x16
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),

                # 16x16 -> 8x8
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                # 8x8 -> 4x4
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, latent_dim),
                nn.ReLU(),
            )

        self.inverse_model = MLP(
            input_dim=2 * latent_dim, fc1_dim=512, fc2_dim=512, output_dim=action_dim
        ).to(device)
        self.forward_model = MLP(
            input_dim=latent_dim + action_dim, fc1_dim=512,fc2_dim= 512, output_dim=latent_dim
        ).to(device)

        self.to(device)
    
    def encode(self, x: T.Tensor) -> T.Tensor:
        """Encodes raw observation into latent feature space."""
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.encoder(x.to(self.device))

    def compute_intrinsic_reward(self, state: T.Tensor, action: T.Tensor, state_: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        phi, phi_next = self.encode(state), self.encode(state_)

        # Inverse model predicts action from (phi, phi_next)
        inverse_model_input  = T.concatenate([phi, phi_next], dim=1)
        pred_action_logits  = self.inverse_model(inverse_model_input)
        inverse_model_loss = F.cross_entropy(pred_action_logits, action)

        # Forward model predicts next feature
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        forward_model_input = T.concatenate([phi, action_onehot], dim=1)
        pred_phi_next = self.forward_model(forward_model_input)
        forward_model_loss = 0.5 * F.mse_loss(pred_phi_next, phi_next)

        # Intrinsic reward = prediction error magnitude
        intrinsic_reward = 0.5 * T.sum((pred_phi_next - phi_next) ** 2, dim=1)

        return intrinsic_reward.detach(), inverse_model_loss, forward_model_loss

    def forward(self, state: T.Tensor, action: T.Tensor, state_: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        return self.compute_intrinsic_reward(state, action, state_)


if __name__ == "__main__":
    image_dim = (4, 64, 64)
    action_dim = 8
    latent_dim = 128

    model = IntrinsicCuriosityModule(input_dim=image_dim, latent_dim=latent_dim, action_dim=action_dim)

    state = T.rand(size=image_dim)
    state_ = T.rand(size=image_dim)
    action = T.tensor([4], dtype=T.long)

    intrinsic_reward, inv_loss, fwd_loss = model(state, action, state_)
    print(
        f"[->] Intrinsic Reward:", intrinsic_reward,
        f"\n[->] Inverse Model Loss: ", inv_loss,
        f"\n[->] Forward Model Loss: ", fwd_loss
    )

