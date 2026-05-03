import numpy as np
import struct
import zlib

from spectrum_resilience.receiver import apply_spreading, modulate


def _save_rgb_png(path: str, image: np.ndarray):
    """Save an RGB image array as a PNG without requiring plotting packages."""
    height, width, _ = image.shape
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(name, data):
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw_rows))
    png += chunk(b"IEND", b"")

    with open(path, "wb") as output:
        output.write(png)


def _draw_line(image, start, end, color):
    """Draw a basic line segment on an RGB image array."""
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        x = int(round(x0 + (x1 - x0) * step / steps))
        y = int(round(y0 + (y1 - y0) * step / steps))
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            image[y, x] = color


def _draw_point(image, x, y, color, radius=2):
    """Draw a small filled square point on an RGB image array."""
    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            if 0 <= yy < image.shape[0] and 0 <= xx < image.shape[1]:
                image[yy, xx] = color


def _fallback_plot_evolution(history: list):
    """Create a simple evolution PNG when matplotlib is unavailable."""
    image = np.full((500, 900, 3), 255, dtype=np.uint8)
    blue = np.array([30, 90, 200], dtype=np.uint8)
    orange = np.array([230, 120, 20], dtype=np.uint8)
    gray = np.array([90, 90, 90], dtype=np.uint8)

    left, right, top, bottom = 70, 850, 40, 450
    _draw_line(image, (left, bottom), (right, bottom), gray)
    _draw_line(image, (left, top), (left, bottom), gray)

    threshold_y = bottom - int(0.8 * (bottom - top))
    for x in range(left, right, 16):
        _draw_line(image, (x, threshold_y), (x + 8, threshold_y), orange)

    if history:
        points = []
        for index, fitness in enumerate(history):
            x = left + int(index / max(1, len(history) - 1) * (right - left))
            y = bottom - int(np.clip(fitness, 0, 1) * (bottom - top))
            points.append((x, y))
        for start, end in zip(points, points[1:]):
            _draw_line(image, start, end, blue)
        for point in points:
            _draw_point(image, point[0], point[1], blue)

    _save_rgb_png("evolution.png", image)
    print("Saved evolution.png using fallback plotting.")


def _fallback_plot_waveform_comparison(original_bits, recovered_bits, params):
    """Create a simple waveform analysis PNG when matplotlib is unavailable."""
    image = np.full((800, 1100, 3), 255, dtype=np.uint8)
    blue = np.array([30, 90, 200], dtype=np.uint8)
    red = np.array([220, 70, 30], dtype=np.uint8)
    gray = np.array([90, 90, 90], dtype=np.uint8)

    transmitted_symbols = modulate(original_bits, params["modulation_order"])
    recovered_symbols = modulate(recovered_bits, params["modulation_order"])
    transmitted_signal = apply_spreading(transmitted_symbols, params["spreading_factor"])
    recovered_signal = apply_spreading(recovered_symbols, params["spreading_factor"])

    left, right, top, bottom = 70, 1030, 50, 350
    _draw_line(image, (left, bottom), (right, bottom), gray)
    _draw_line(image, (left, top), (left, bottom), gray)
    count = min(200, len(transmitted_signal), len(recovered_signal))
    if count:
        tx = transmitted_signal.real[:count]
        rx = recovered_signal.real[:count]
        scale = max(1.0, np.max(np.abs(np.concatenate([tx, rx]))))
        tx_points = []
        rx_points = []
        for index in range(count):
            x = left + int(index / max(1, count - 1) * (right - left))
            tx_points.append((x, int((top + bottom) / 2 - tx[index] / scale * 120)))
            rx_points.append((x, int((top + bottom) / 2 - rx[index] / scale * 120)))
        for start, end in zip(tx_points, tx_points[1:]):
            _draw_line(image, start, end, blue)
        for start, end in zip(rx_points, rx_points[1:]):
            _draw_line(image, start, end, red)

    panels = [
        (70, 430, 520, 760, transmitted_symbols, blue),
        (610, 430, 1030, 760, recovered_symbols, red),
    ]
    for x0, y0, x1, y1, symbols, color in panels:
        _draw_line(image, (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2), gray)
        _draw_line(image, ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), gray)
        scale = max(1.0, np.max(np.abs(np.concatenate([symbols.real, symbols.imag]))))
        for symbol in symbols[:1000]:
            x = int((x0 + x1) / 2 + symbol.real / scale * (x1 - x0) * 0.42)
            y = int((y0 + y1) / 2 - symbol.imag / scale * (y1 - y0) * 0.42)
            _draw_point(image, x, y, color)

    _save_rgb_png("waveform_analysis.png", image)
    print("Saved waveform_analysis.png using fallback plotting.")


def plot_evolution(history: list, config):
    """Plot how the best fitness score changes as generations evolve."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_plot_evolution(history)
        return

    generations = np.arange(1, len(history) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(generations, history, color="blue", linewidth=2, label="Best fitness")
    plt.axhline(0.8, color="orange", linestyle="--", label="Target threshold")
    plt.xlabel("Generation")
    plt.ylabel("Fitness score")
    plt.ylim(0, 1)
    plt.title(f"Waveform evolution — {config.channel_type.upper()} channel")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("evolution.png")
    plt.show()


def plot_waveform_comparison(original_bits, recovered_bits, params, config):
    """Plot transmitted and recovered waveform views for quick signal inspection."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_plot_waveform_comparison(original_bits, recovered_bits, params)
        return

    transmitted_symbols = modulate(original_bits, params["modulation_order"])
    recovered_symbols = modulate(recovered_bits, params["modulation_order"])
    transmitted_signal = apply_spreading(transmitted_symbols, params["spreading_factor"])
    recovered_signal = apply_spreading(recovered_symbols, params["spreading_factor"])

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(2, 2)

    time_axis = np.arange(min(200, len(transmitted_signal), len(recovered_signal)))
    ax_time = fig.add_subplot(grid[0, :])
    ax_time.plot(
        time_axis,
        transmitted_signal.real[: len(time_axis)],
        color="blue",
        label="Transmitted",
    )
    ax_time.plot(
        time_axis,
        recovered_signal.real[: len(time_axis)],
        color="orangered",
        alpha=0.8,
        label="Received",
    )
    ax_time.set_title("Time-domain signal comparison")
    ax_time.set_xlabel("Sample")
    ax_time.set_ylabel("Real amplitude")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()

    ax_tx = fig.add_subplot(grid[1, 0])
    ax_tx.scatter(transmitted_symbols.real, transmitted_symbols.imag, color="blue", s=14)
    ax_tx.set_title("Transmitted constellation")
    ax_tx.set_xlabel("In-phase")
    ax_tx.set_ylabel("Quadrature")
    ax_tx.grid(True, alpha=0.3)
    ax_tx.axis("equal")

    ax_rx = fig.add_subplot(grid[1, 1])
    ax_rx.scatter(recovered_symbols.real, recovered_symbols.imag, color="orangered", s=14)
    ax_rx.set_title("Received despread constellation")
    ax_rx.set_xlabel("In-phase")
    ax_rx.set_ylabel("Quadrature")
    ax_rx.grid(True, alpha=0.3)
    ax_rx.axis("equal")

    fig.suptitle(f"Waveform analysis — {config.channel_type.upper()} channel")
    fig.tight_layout()
    fig.savefig("waveform_analysis.png")
    plt.show()


def print_report(best_genome, config, final_ber: float):
    """Print a mission summary with pass/fail status and decoded waveform settings."""
    params = best_genome.decode()
    status = "PASS" if final_ber <= config.target_ber else "FAIL"

    print("=" * 60)
    print("SPECTRUM RESILIENCE MISSION REPORT")
    print("=" * 60)
    print(f"Channel type : {config.channel_type}")
    print(f"Target SNR   : {config.target_snr_db:.2f} dB")
    print(f"Target BER   : {config.target_ber:.6f}")
    print("-" * 60)
    print(f"Final BER    : {final_ber:.6f} [{status}]")
    print(f"Fitness score: {best_genome.fitness:.6f}")
    print("-" * 60)
    print("Decoded waveform parameters")
    print("-" * 60)

    for name, value in params.items():
        if isinstance(value, float):
            print(f"{name:18}: {value:.6f}")
        else:
            print(f"{name:18}: {value}")

    print("=" * 60)
