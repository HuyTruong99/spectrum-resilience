from dataclasses import dataclass


@dataclass
class MissionConfig:
    """Mission objective that the AI waveform generator will try to satisfy."""

    target_snr_db: float = 10.0
    target_ber: float = 0.01
    num_bits: int = 1024
    carrier_freq: float = 2.4e9
    bandwidth: float = 1e6
    channel_type: str = "awgn"
    jammer_power: float = 0.0
    population_size: int = 20
    num_generations: int = 50
    waveform_dim: int = 8
