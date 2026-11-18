"""Core data structures and genetic algorithm for the Agadir TSP demo."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from io import StringIO
from typing import Callable, Sequence


@dataclass(frozen=True)
class Stop:
    stop_id: str
    latitude: float
    longitude: float
    name: str


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    selection: str = "Tournoi"
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    generations: int = 500
    seed: int = 42

    def __post_init__(self) -> None:
        if self.population_size not in (50, 100, 500):
            raise ValueError("La taille de population doit être 50, 100 ou 500.")
        if self.selection not in ("Tournoi", "Roue biaisée"):
            raise ValueError("La sélection doit être Tournoi ou Roue biaisée.")
        if self.mutation_rate not in (0.2, 0.5, 0.8):
            raise ValueError("Le taux de mutation doit être 0.2, 0.5 ou 0.8.")
        if self.crossover_rate not in (0.2, 0.5, 0.8):
            raise ValueError("Le taux de crossover doit être 0.2, 0.5 ou 0.8.")
        if self.generations not in (100, 500, 1000):
            raise ValueError("Le nombre d'itérations doit être 100, 500 ou 1000.")


@dataclass(frozen=True)
class GAResult:
    route: tuple[int, ...]
    distance_km: float
    convergence_km: tuple[float, ...]


ProgressCallback = Callable[[int, int, float], None]


def make_demo_stops() -> list[Stop]:
    """Create 40 reproducible sample stops in central, dense Agadir."""
    rng = random.Random(2026)
    center_lat, center_lon = 30.4275, -9.5981
    stops = [Stop("01", center_lat, center_lon, "Dépôt")]
    for index in range(2, 41):
        stops.append(
            Stop(
                f"{index:02d}",
                center_lat + rng.uniform(-0.008, 0.008),
                center_lon + rng.uniform(-0.011, 0.011),
                f"Point {index:02d}",
            )
        )
    return stops


def parse_stops_csv(csv_text: str) -> list[Stop]:
    """Read exactly 40 stops; the first row is treated as the depot."""
    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("Le CSV doit contenir les colonnes id, latitude et longitude.")

    columns = {name.strip().lower(): name for name in reader.fieldnames}
    id_column = columns.get("id") or columns.get("stop_id")
    latitude_column = columns.get("latitude") or columns.get("lat")
    longitude_column = columns.get("longitude") or columns.get("lon")
    name_column = columns.get("name") or columns.get("nom")
    if not id_column or not latitude_column or not longitude_column:
        raise ValueError("Colonnes requises : id, latitude, longitude (name/nom facultatif).")

    stops: list[Stop] = []
    try:
        for row_number, row in enumerate(reader, start=2):
            stop_id = (row.get(id_column) or "").strip()
            latitude = float(row[latitude_column])
            longitude = float(row[longitude_column])
            name = (row.get(name_column) or "").strip() if name_column else ""
            if not stop_id:
                raise ValueError(f"Identifiant manquant à la ligne {row_number}.")
            if not math.isfinite(latitude) or not -90 <= latitude <= 90:
                raise ValueError(f"Latitude invalide à la ligne {row_number}.")
            if not math.isfinite(longitude) or not -180 <= longitude <= 180:
                raise ValueError(f"Longitude invalide à la ligne {row_number}.")
            stops.append(Stop(stop_id, latitude, longitude, name or f"Point {stop_id}"))
    except (TypeError, KeyError) as error:
        raise ValueError("Coordonnées CSV manquantes ou invalides.") from error
    except ValueError as error:
        if str(error).startswith(("Identifiant", "Latitude", "Longitude")):
            raise
        raise ValueError("Les colonnes latitude et longitude doivent être numériques.") from error

    if len(stops) != 40:
        raise ValueError(f"Le fichier doit contenir exactement 40 points (reçu : {len(stops)}).")
    if len({stop.stop_id for stop in stops}) != len(stops):
        raise ValueError("Les identifiants des points doivent être uniques.")
    return stops


def haversine_matrix(stops: Sequence[Stop]) -> list[list[float]]:
    """Return straight-line distances in kilometres."""
    if len(stops) < 2:
        raise ValueError("Il faut au moins deux points pour calculer des distances.")
    earth_radius_km = 6371.0088
    matrix: list[list[float]] = []
    for origin in stops:
        row = []
        lat1 = math.radians(origin.latitude)
        for destination in stops:
            lat2 = math.radians(destination.latitude)
            delta_lat = lat2 - lat1
            delta_lon = math.radians(destination.longitude - origin.longitude)
            haversine = (
                math.sin(delta_lat / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
            )
            row.append(2 * earth_radius_km * math.asin(math.sqrt(haversine)))
        matrix.append(row)
    return matrix


def _route_distance(route: Sequence[int], distances: Sequence[Sequence[float]]) -> float:
    return sum(
        distances[route[index]][route[(index + 1) % len(route)]]
        for index in range(len(route))
    )


def _order_crossover(
    first: Sequence[int], second: Sequence[int], rng: random.Random
) -> list[int]:
    if len(first) < 2:
        return list(first)
    left, right = sorted(rng.sample(range(len(first)), 2))
    child: list[int | None] = [None] * len(first)
    child[left : right + 1] = first[left : right + 1]
    remaining = [gene for gene in second if gene not in child]
    positions = list(range(right + 1, len(first))) + list(range(0, left))
    for position, gene in zip(positions, remaining):
        child[position] = gene
    return [gene for gene in child if gene is not None]


def _mutate(chromosome: Sequence[int], rng: random.Random) -> list[int]:
    mutated = list(chromosome)
    if len(mutated) >= 2:
        left, right = sorted(rng.sample(range(len(mutated)), 2))
        if rng.random() < 0.5:
            mutated[left], mutated[right] = mutated[right], mutated[left]
        else:
            mutated[left : right + 1] = reversed(mutated[left : right + 1])
    return mutated


def run_genetic_algorithm(
    distances: Sequence[Sequence[float]],
    config: GAConfig,
    progress_callback: ProgressCallback | None = None,
) -> GAResult:
    """Find a depot-rooted cycle. Matrix distances are expressed in kilometres."""
    size = len(distances)
    if size < 3 or any(len(row) != size for row in distances):
        raise ValueError("La matrice des distances doit être carrée et contenir au moins 3 points.")

    rng = random.Random(config.seed)
    genes = list(range(1, size))
    population = []
    for _ in range(config.population_size):
        chromosome = genes.copy()
        rng.shuffle(chromosome)
        population.append(chromosome)

    costs = [_route_distance((0, *chromosome), distances) for chromosome in population]
    best_index = min(range(len(population)), key=costs.__getitem__)
    best_chromosome = population[best_index].copy()
    best_distance = costs[best_index]
    convergence = [best_distance]

    for generation in range(1, config.generations + 1):
        elite_index = min(range(len(population)), key=costs.__getitem__)
        next_population = [population[elite_index].copy()]

        if config.selection == "Roue biaisée":
            fitness = [1 / max(cost, 1e-12) for cost in costs]
            cumulative_fitness = []
            total_fitness = 0.0
            for value in fitness:
                total_fitness += value
                cumulative_fitness.append(total_fitness)

            def choose_parent() -> list[int]:
                target = rng.random() * total_fitness
                selected = min(
                    len(cumulative_fitness) - 1,
                    next(
                        (
                            index
                            for index, cumulative in enumerate(cumulative_fitness)
                            if cumulative >= target
                        ),
                        len(cumulative_fitness) - 1,
                    ),
                )
                return population[selected]

        else:

            def choose_parent() -> list[int]:
                contestants = rng.sample(range(len(population)), min(3, len(population)))
                selected = min(contestants, key=costs.__getitem__)
                return population[selected]

        while len(next_population) < config.population_size:
            first = choose_parent()
            second = choose_parent()
            if rng.random() < config.crossover_rate:
                child_a = _order_crossover(first, second, rng)
                child_b = _order_crossover(second, first, rng)
            else:
                child_a, child_b = first.copy(), second.copy()

            if rng.random() < config.mutation_rate:
                child_a = _mutate(child_a, rng)
            if rng.random() < config.mutation_rate:
                child_b = _mutate(child_b, rng)
            next_population.append(child_a)
            if len(next_population) < config.population_size:
                next_population.append(child_b)

        population = next_population
        costs = [_route_distance((0, *chromosome), distances) for chromosome in population]
        generation_best = min(range(len(population)), key=costs.__getitem__)
        if costs[generation_best] < best_distance:
            best_chromosome = population[generation_best].copy()
            best_distance = costs[generation_best]
        convergence.append(best_distance)
        if progress_callback is not None:
            progress_callback(generation, config.generations, best_distance)

    return GAResult((0, *best_chromosome), best_distance, tuple(convergence))
