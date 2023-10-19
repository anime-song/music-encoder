from dataclasses import dataclass, field


@dataclass
class MusicEncoderConfig:
    # feature extractor
    filter_sizes: list = field(
        default_factory=lambda: [512, ]
    )
    kernel_sizes: list = field(
        default_factory=lambda: [4, ])
    strides: list = field(default_factory=lambda: [1, ])

    # encoder
    hidden_size: int = 512
    num_layers: int = 11
    embedding_dim: int = 512
    temperature: float = 0.1
    dropout: float = 0.1
    num_heads: int = 8
    intermediate_size: int = 2048
    is_gelu_approx: bool = False
    layer_norm_eps: float = 1e-6
    norm_first: bool = True

    # quantizer
    quantizer_embedding_dim: int = 64
    codebook_size: int = 1024
    commitment_cost: float = 0.0
    num_quantizers: int = 8
    ema_decay: float = 0.99
    threshold_ema_dead_code: float = 2.0
    sample_codebook_temperature: float = 0.0
    kmeans_init: bool = False
