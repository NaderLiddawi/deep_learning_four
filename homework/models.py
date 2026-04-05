from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # input: concat left + right → (B, n_track*2*2) = (B, 40)
        self.mlp = nn.Sequential(
            nn.Linear(n_track * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            track_left:  (B, n_track, 2)
            track_right: (B, n_track, 2)
        Returns:
            (B, n_waypoints, 2)
        """
        B = track_left.shape[0]
        x = torch.cat([track_left, track_right], dim=1).view(B, -1)  # (B, 40)
        return self.mlp(x).view(B, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3, d_model: int = 64):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # project 2D track points into d_model space
        self.input_proj = nn.Linear(2, d_model)

        # n_waypoints learned query embeddings (the "latent array" in Perceiver)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Perceiver-style: queries cross-attend over encoded track points
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,  # (B, seq, d_model) convention
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # project d_model → 2D waypoint coords
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            track_left:  (B, n_track, 2)
            track_right: (B, n_track, 2)
        Returns:
            (B, n_waypoints, 2)
        """
        B = track_left.shape[0]

        # Encode track points: (B, 2*n_track, 2) → (B, 2*n_track, d_model)
        track = torch.cat([track_left, track_right], dim=1)
        memory = self.input_proj(track)

        # Expand query embeddings across batch: (B, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: queries attend over track memory
        out = self.decoder(tgt=queries, memory=memory)  # (B, n_waypoints, d_model)

        return self.output_proj(out)  # (B, n_waypoints, 2)


class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3):
        super().__init__()
        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Input: (B, 3, 96, 128)
        # Each stride-2 conv halves H and W:
        #   After 1: (B,  32, 48, 64)
        #   After 2: (B,  64, 24, 32)
        #   After 3: (B, 128, 12, 16)
        #   After 4: (B, 128,  6,  8)
        # Global avg pool → (B, 128) → head → (B, n_waypoints*2)
        def conv_block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.backbone = nn.Sequential(
            conv_block(3,   32),
            conv_block(32,  64),
            conv_block(64,  128),
            conv_block(128, 128),
        )
        self.head = nn.Linear(128, n_waypoints * 2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image: (B, 3, 96, 128), values in [0, 1]
        Returns:
            (B, n_waypoints, 2)
        """
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.backbone(x)          # (B, 128, 6, 8)
        x = x.mean(dim=(-2, -1))      # global average pool → (B, 128)
        return self.head(x).view(image.shape[0], self.n_waypoints, 2)


# ---------------------------------------------------------------------------
# Factory + I/O helpers (unchanged interface)
# ---------------------------------------------------------------------------

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    model_size_mb = calculate_model_size_mb(m)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: nn.Module) -> str:
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return output_path


def calculate_model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
