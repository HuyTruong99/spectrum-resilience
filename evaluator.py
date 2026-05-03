import numpy as np

from spectrum_resilience.channel_simulator import simulate_channel
from spectrum_resilience.receiver import transmit_and_receive


def compute_ber(original_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    """Compute the fraction of compared bits that were recovered incorrectly."""
    compare_length = min(len(original_bits), len(recovered_bits))
    if compare_length == 0:
        return 0.0

    original = original_bits[:compare_length]
    recovered = recovered_bits[:compare_length]
    return np.mean(original != recovered)


def fitness_function(genome, config, num_trials=3) -> float:
    """Score a genome by BER target, spectral efficiency, and BER consistency.

    The BER sub-score rewards meeting the mission bit-error-rate objective.
    The spectral sub-score rewards sending more coded bits per spread symbol.
    The robustness sub-score rewards stable performance across repeated trials.
    """
    params = genome.decode()
    np.random.seed(42)
    trial_bers = []

    for _ in range(num_trials):
        bits = np.random.randint(0, 2, config.num_bits)
        recovered_bits = transmit_and_receive(
            bits,
            params,
            lambda signal: simulate_channel(signal, config),
        )
        trial_bers.append(compute_ber(bits, recovered_bits))

    mean_ber = np.mean(trial_bers)
    std_ber = np.std(trial_bers)
    modulation_order = params["modulation_order"]
    coding_rate = params["coding_rate"]
    spreading_factor = params["spreading_factor"]

    ber_score = np.exp(-max(0, mean_ber - config.target_ber) * 50)
    spectral_score = np.clip(
        np.log2(modulation_order) * coding_rate / spreading_factor / 6,
        0,
        1,
    )
    robustness_score = np.exp(-std_ber * 20)

    return 0.6 * ber_score + 0.3 * spectral_score + 0.1 * robustness_score
