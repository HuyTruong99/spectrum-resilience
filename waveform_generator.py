import random

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:
        """Small fallback progress wrapper used when tqdm is not installed."""

        def __init__(self, iterable, **kwargs):
            """Store the iterable so code can loop over it without tqdm installed."""
            self.iterable = iterable

        def __iter__(self):
            """Yield values from the wrapped iterable."""
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            """Accept progress metadata without displaying it."""
            return None


class WaveformGenome:
    """A compact set of tunable waveform genes for the evolutionary search."""

    def __init__(self, dim: int):
        """Create a genome with random gene values between 0 and 1."""
        self.genes = np.random.random(dim)
        self.fitness = float("-inf")

    def decode(self) -> dict:
        """Convert raw gene values into named waveform parameters."""
        genes = self.genes
        return {
            "modulation_order": 2 ** (1 + round(genes[0] * 3)),
            "spreading_factor": int(1 + round(genes[1] * 7)),
            "pilot_ratio": 0.05 + genes[2] * 0.25,
            "freq_hop_rate": genes[3],
            "coding_rate": 0.25 + genes[4] * 0.75,
            "power_level": 0.1 + genes[5] * 0.9,
            "pulse_alpha": 0.1 + genes[6] * 0.9,
            "redundancy": genes[7],
        }

    def mutate(self, rate=0.15, strength=0.2):
        """Randomly nudge some genes, then keep every value between 0 and 1."""
        mutation_mask = np.random.random(len(self.genes)) < rate
        perturbations = np.random.normal(0.0, strength, len(self.genes))
        self.genes = np.clip(self.genes + mutation_mask * perturbations, 0.0, 1.0)
        return self

    def crossover(self, other):
        """Create a child by taking early genes from self and later genes from other."""
        child = WaveformGenome(len(self.genes))
        crossover_point = random.randint(1, len(self.genes) - 1)
        child.genes = np.concatenate(
            [self.genes[:crossover_point], other.genes[crossover_point:]]
        )
        return child

    def __repr__(self):
        """Show the most important decoded settings and the current fitness."""
        params = self.decode()
        return (
            "WaveformGenome("
            f"modulation_order={params['modulation_order']}, "
            f"spreading_factor={params['spreading_factor']}, "
            f"coding_rate={params['coding_rate']:.3f}, "
            f"fitness={self.fitness:.4f}"
            ")"
        )


class EvolutionaryWaveformGenerator:
    """Evolves a population of waveform genomes toward higher fitness."""

    def __init__(self, config):
        """Create the initial random population and tracking fields."""
        self.config = config
        self.population = [
            WaveformGenome(config.waveform_dim)
            for _ in range(config.population_size)
        ]
        self.best_genome = None
        self.history = []

    def evolve_one_generation(self, evaluate_fn) -> float:
        """Score, sort, keep the strongest genomes, and breed the next generation."""
        for genome in self.population:
            genome.fitness = evaluate_fn(genome)

        self.population.sort(key=lambda genome: genome.fitness, reverse=True)
        self.best_genome = self.population[0]
        self.history.append(self.best_genome.fitness)

        survivor_count = max(1, len(self.population) // 2)
        survivors = self.population[:survivor_count]
        children = []

        while len(survivors) + len(children) < self.config.population_size:
            if len(survivors) > 1:
                parent_a, parent_b = random.sample(survivors, k=2)
            else:
                parent_a, parent_b = survivors[0], survivors[0]
            child = parent_a.crossover(parent_b).mutate()
            children.append(child)

        self.population = survivors + children
        return self.best_genome.fitness

    def run(self, evaluate_fn, verbose=True) -> WaveformGenome:
        """Run the configured number of generations and return the best genome found."""
        generations = range(self.config.num_generations)
        if verbose:
            generations = tqdm(generations, desc="Evolving waveforms")

        for _ in generations:
            best_fitness = self.evolve_one_generation(evaluate_fn)
            if verbose:
                generations.set_postfix(best_fitness=f"{best_fitness:.4f}")

        return self.best_genome
