import numpy as np


def db_to_linear(db: float) -> float:
    """Convert a decibel value into the equivalent linear power ratio."""
    return 10 ** (db / 10)


def awgn_channel(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Simulate thermal receiver noise by adding complex white Gaussian noise."""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / db_to_linear(snr_db)
    noise = (
        (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        / np.sqrt(2)
        * np.sqrt(noise_power)
    )
    return signal + noise


def rayleigh_fading_channel(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Simulate multipath fading where random reflections change phase and amplitude."""
    h = (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    ) / np.sqrt(2)
    faded_signal = signal * h
    return awgn_channel(faded_signal, snr_db)


def jamming_channel(
    signal: np.ndarray,
    snr_db: float,
    jammer_power=0.5,
) -> np.ndarray:
    """Simulate a hostile narrowband tone jammer added on top of receiver noise."""
    noisy_signal = awgn_channel(signal, snr_db)
    signal_power = np.mean(np.abs(signal) ** 2)
    amplitude = np.sqrt(jammer_power * signal_power)
    samples = np.arange(signal.size).reshape(signal.shape)
    jammer = amplitude * np.exp(1j * 2 * np.pi * 0.1 * samples)
    return noisy_signal + jammer


def simulate_channel(signal: np.ndarray, config) -> np.ndarray:
    """Send a signal through the physical channel model selected by the config."""
    if config.channel_type == "awgn":
        return awgn_channel(signal, config.target_snr_db)
    if config.channel_type == "fading":
        return rayleigh_fading_channel(signal, config.target_snr_db)
    if config.channel_type == "jamming":
        return jamming_channel(signal, config.target_snr_db, config.jammer_power)

    raise ValueError(f"Unknown channel type: {config.channel_type}")
