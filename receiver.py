import numpy as np


def modulate(bits: np.ndarray, modulation_order: int) -> np.ndarray:
    """Convert transmit bits into complex symbols for the selected modulation."""
    bits = np.asarray(bits, dtype=int)

    if modulation_order == 2:
        return (2 * bits - 1).astype(complex)

    bits_per_symbol = int(np.log2(modulation_order))
    padding = (-len(bits)) % bits_per_symbol
    if padding:
        bits = np.pad(bits, (0, padding))

    bit_groups = bits.reshape(-1, bits_per_symbol)
    weights = 2 ** np.arange(bits_per_symbol - 1, -1, -1)
    symbol_ints = bit_groups @ weights

    grid_size = int(np.sqrt(modulation_order))
    real = 2 * (symbol_ints % grid_size) - (grid_size - 1)
    imag = 2 * (symbol_ints // grid_size) - (grid_size - 1)
    normalization = np.sqrt(2 * (modulation_order - 1) / 3)
    return (real + 1j * imag) / normalization


def demodulate(received: np.ndarray, modulation_order: int) -> np.ndarray:
    """Convert received complex symbols back into bits using nearest-point decisions."""
    received = np.asarray(received)

    if modulation_order == 2:
        return (received.real >= 0).astype(int)

    bits_per_symbol = int(np.log2(modulation_order))
    grid_size = int(np.sqrt(modulation_order))
    normalization = np.sqrt(2 * (modulation_order - 1) / 3)
    scaled = received * normalization

    real_idx = np.rint((scaled.real + (grid_size - 1)) / 2).astype(int)
    imag_idx = np.rint((scaled.imag + (grid_size - 1)) / 2).astype(int)
    real_idx = np.clip(real_idx, 0, grid_size - 1)
    imag_idx = np.clip(imag_idx, 0, grid_size - 1)
    symbol_ints = imag_idx * grid_size + real_idx

    bits = [
        (symbol_int >> bit_index) & 1
        for symbol_int in symbol_ints
        for bit_index in range(bits_per_symbol - 1, -1, -1)
    ]
    return np.array(bits, dtype=int)


def apply_spreading(signal: np.ndarray, spreading_factor: int) -> np.ndarray:
    """Repeat each symbol to mimic direct-sequence spread spectrum transmission."""
    return np.repeat(signal, spreading_factor)


def despread(signal: np.ndarray, spreading_factor: int) -> np.ndarray:
    """Average repeated samples back into one symbol estimate per spreading group."""
    usable_length = (len(signal) // spreading_factor) * spreading_factor
    trimmed_signal = signal[:usable_length]
    return trimmed_signal.reshape(-1, spreading_factor).mean(axis=1)


def transmit_and_receive(bits, params, channel_fn) -> np.ndarray:
    """Run bits through modulation, spreading, the channel, despreading, and demodulation."""
    transmitted = modulate(bits, params["modulation_order"])
    spread = apply_spreading(transmitted, params["spreading_factor"])
    received = channel_fn(spread)
    symbols = despread(received, params["spreading_factor"])
    recovered_bits = demodulate(symbols, params["modulation_order"])
    return recovered_bits[:len(bits)]
