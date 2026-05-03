from pathlib import Path
import sys

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from spectrum_resilience.channel_simulator import simulate_channel
    from spectrum_resilience.config import MissionConfig
    from spectrum_resilience.evaluator import compute_ber, fitness_function
    from spectrum_resilience.receiver import (
        apply_spreading,
        demodulate,
        despread,
        modulate,
        transmit_and_receive,
    )
    from spectrum_resilience.waveform_generator import EvolutionaryWaveformGenerator
except ModuleNotFoundError:
    from channel_simulator import simulate_channel
    from config import MissionConfig
    from evaluator import compute_ber, fitness_function
    from receiver import (
        apply_spreading,
        demodulate,
        despread,
        modulate,
        transmit_and_receive,
    )
    from waveform_generator import EvolutionaryWaveformGenerator


st.set_page_config(page_title="Spectrum Resilience", page_icon="\U0001F4E1", layout="wide")


def build_config(
    channel_type,
    target_snr_db,
    target_ber,
    jammer_power,
    num_generations,
    population_size,
    num_bits,
):
    """Create a MissionConfig from the current dashboard controls."""
    return MissionConfig(
        target_snr_db=target_snr_db,
        target_ber=target_ber,
        num_bits=num_bits,
        channel_type=channel_type,
        jammer_power=jammer_power if channel_type == "jamming" else 0.0,
        population_size=population_size,
        num_generations=num_generations,
        waveform_dim=8,
    )


def make_fitness_figure(history, config):
    """Create a matplotlib figure showing fitness over generations."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    generations = np.arange(1, len(history) + 1)
    ax.plot(generations, history, color="tab:blue", linewidth=2)
    ax.axhline(0.8, color="tab:orange", linestyle="--", label="Target threshold")
    ax.set_title(f"Fitness over generations - {config.channel_type.upper()}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_time_domain_figure(transmitted_signal, received_signal):
    """Create a matplotlib figure comparing transmitted and received real samples."""
    import matplotlib.pyplot as plt

    sample_count = min(200, len(transmitted_signal), len(received_signal))
    sample_axis = np.arange(sample_count)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        sample_axis,
        transmitted_signal.real[:sample_count],
        color="tab:blue",
        label="Transmitted",
    )
    ax.plot(
        sample_axis,
        received_signal.real[:sample_count],
        color="orangered",
        alpha=0.8,
        label="Received",
    )
    ax.set_title("Time domain signal")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Real amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_constellation_figure(received_symbols):
    """Create a matplotlib figure showing the received despread constellation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(received_symbols.real, received_symbols.imag, color="orangered", s=16)
    ax.set_title("Rx constellation")
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def run_final_link(bits, params, config):
    """Run the final transmit/channel/receive chain and keep signals for plots."""
    transmitted_symbols = modulate(bits, params["modulation_order"])
    transmitted_signal = apply_spreading(transmitted_symbols, params["spreading_factor"])
    received_signal = simulate_channel(transmitted_signal, config)
    received_symbols = despread(received_signal, params["spreading_factor"])
    recovered_bits = demodulate(received_symbols, params["modulation_order"])[: len(bits)]
    return transmitted_signal, received_signal, received_symbols, recovered_bits


def run_experiment(config):
    """Evolve one channel experiment and render its results."""
    st.subheader(f"{config.channel_type.upper()} channel")
    generator = EvolutionaryWaveformGenerator(config)
    progress = st.progress(0.0)
    chart_slot = st.empty()

    def eval_fn(genome):
        return fitness_function(genome, config)

    for generation in range(config.num_generations):
        generator.evolve_one_generation(eval_fn)
        progress.progress((generation + 1) / config.num_generations)
        if (generation + 1) % 5 == 0 or generation == config.num_generations - 1:
            chart_slot.pyplot(make_fitness_figure(generator.history, config))

    best_genome = generator.best_genome
    params = best_genome.decode()

    np.random.seed(0)
    bits = np.random.randint(0, 2, config.num_bits)
    recovered = transmit_and_receive(
        bits,
        params,
        lambda signal: simulate_channel(signal, config),
    )
    final_ber = compute_ber(bits, recovered)
    transmitted_signal, received_signal, received_symbols, plot_recovered = run_final_link(
        bits,
        params,
        config,
    )

    pass_fail = "PASS" if final_ber <= config.target_ber else "FAIL"
    metric_cols = st.columns(4)
    metric_cols[0].metric("Final BER", f"{final_ber:.6f}", pass_fail)
    metric_cols[1].metric("Fitness score", f"{best_genome.fitness:.4f}")
    metric_cols[2].metric("Modulation order", params["modulation_order"])
    metric_cols[3].metric("Spreading factor", params["spreading_factor"])

    with st.expander("Decoded waveform parameters", expanded=False):
        st.table(
            [
                {"Parameter": name, "Value": value}
                for name, value in params.items()
            ]
        )

    plot_cols = st.columns(2)
    plot_cols[0].pyplot(make_time_domain_figure(transmitted_signal, received_signal))
    plot_cols[1].pyplot(make_constellation_figure(received_symbols))

    return {
        "channel": config.channel_type,
        "ber": final_ber,
        "fitness": best_genome.fitness,
        "status": pass_fail,
        "plot_ber": compute_ber(bits, plot_recovered),
    }


st.title("Spectrum Resilience Dashboard")

with st.sidebar:
    st.header("Experiment")
    mode = st.radio(
        "Mode",
        ["Single channel", "All three (AWGN -> Fading -> Jamming)"],
    )
    single_channel = None
    if mode == "Single channel":
        single_channel = st.selectbox("Channel type", ["awgn", "fading", "jamming"])

    target_snr_db = st.slider("Target SNR (dB)", 0.0, 30.0, 10.0)
    target_ber = st.select_slider(
        "Target BER",
        options=[0.001, 0.005, 0.01, 0.05, 0.1],
        value=0.01,
    )

    active_channel = single_channel if single_channel else "jamming"
    jammer_power = 0.0
    if active_channel == "jamming" or mode != "Single channel":
        jammer_power = st.slider("Jammer power", 0.0, 1.0, 0.5)

    num_generations = st.slider("Generations", 10, 100, 50)
    population_size = st.slider("Population size", 10, 50, 20)
    num_bits = st.select_slider(
        "Bits per transmission",
        options=[256, 512, 1024, 2048],
        value=1024,
    )
    run_clicked = st.button("Run", type="primary")

st.caption("Configure an experiment in the sidebar, then run the evolutionary search.")

if run_clicked:
    if mode == "Single channel":
        selected_channels = [single_channel]
    else:
        selected_channels = ["awgn", "fading", "jamming"]

    results = []
    for selected_channel in selected_channels:
        st.write(f"Running {selected_channel.upper()} channel")
        mission_config = build_config(
            selected_channel,
            target_snr_db,
            target_ber,
            jammer_power,
            num_generations,
            population_size,
            num_bits,
        )
        results.append(run_experiment(mission_config))

    if len(results) == 3:
        st.header("Comparison")
        comparison_cols = st.columns(3)
        for column, result in zip(comparison_cols, results):
            column.metric(
                result["channel"].upper(),
                f"BER {result['ber']:.6f}",
                result["status"],
            )

        st.bar_chart(
            {
                "Fitness score": {
                    result["channel"].upper(): result["fitness"]
                    for result in results
                }
            }
        )
else:
    st.info("The dashboard is ready. Click Run when you want to start evolution.")
