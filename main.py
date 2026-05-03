from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectrum_resilience.channel_simulator import simulate_channel
from spectrum_resilience.config import MissionConfig
from spectrum_resilience.evaluator import compute_ber, fitness_function
from spectrum_resilience.receiver import transmit_and_receive
from spectrum_resilience.visualizer import (
    plot_evolution,
    plot_waveform_comparison,
    print_report,
)
from spectrum_resilience.waveform_generator import EvolutionaryWaveformGenerator


def main():
    """Run the full Spectrum Resilience waveform evolution pipeline."""
    channel_type = sys.argv[1].lower() if len(sys.argv) > 1 else "awgn"
    valid_channels = {"awgn", "fading", "jamming"}
    if channel_type not in valid_channels:
        raise ValueError(
            f"Unknown channel type '{channel_type}'. "
            f"Choose one of: {', '.join(sorted(valid_channels))}"
        )

    config = MissionConfig(
        target_snr_db=10.0,
        target_ber=0.01,
        num_bits=1024,
        channel_type=channel_type,
        jammer_power=0.5 if channel_type == "jamming" else 0.3,
        population_size=20,
        num_generations=50,
        waveform_dim=8,
    )

    print(f"Running channel experiment: {channel_type.upper()}")
    print("=" * 60)
    print("SPECTRUM RESILIENCE STARTUP")
    print("=" * 60)
    print(f"Channel type: {config.channel_type.upper()}")
    print(f"Target SNR  : {config.target_snr_db:.2f} dB")
    print(f"Target BER  : {config.target_ber:.6f}")
    print("=" * 60)

    generator = EvolutionaryWaveformGenerator(config)

    def eval_fn(genome):
        """Evaluate one genome against the configured mission objective."""
        return fitness_function(genome, config)

    best_genome = generator.run(eval_fn, verbose=True)

    np.random.seed(0)
    bits = np.random.randint(0, 2, config.num_bits)
    recovered = transmit_and_receive(
        bits,
        best_genome.decode(),
        lambda signal: simulate_channel(signal, config),
    )
    final_ber = compute_ber(bits, recovered)

    print_report(best_genome, config, final_ber)
    plot_evolution(generator.history, config)
    plot_waveform_comparison(bits, recovered, best_genome.decode(), config)


if __name__ == "__main__":
    main()
