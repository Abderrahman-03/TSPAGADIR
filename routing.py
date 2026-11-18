"""OSRM road-distance and route requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import requests

from tsp_core import Stop, haversine_matrix

OSRM_BASE_URL = "https://router.project-osrm.org"


class OSRMError(RuntimeError):
    """Raised when the public OSRM service cannot provide a valid response."""


@dataclass(frozen=True)
class RoadRoute:
    coordinates: tuple[tuple[float, float], ...]
    distance_km: float
    duration_minutes: float


def get_osrm_distance_matrix(
    stops: Sequence[Stop], base_url: str = OSRM_BASE_URL
) -> list[list[float]]:
    coordinates = ";".join(f"{stop.longitude},{stop.latitude}" for stop in stops)
    try:
        response = requests.get(
            f"{base_url}/table/v1/driving/{coordinates}",
            params={"annotations": "distance"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise OSRMError(f"Impossible de joindre OSRM pour la matrice : {error}") from error

    if payload.get("code") != "Ok":
        raise OSRMError(f"OSRM a refusé la matrice : {payload.get('message', payload.get('code'))}")
    raw_distances = payload.get("distances")
    if not isinstance(raw_distances, list) or len(raw_distances) != len(stops):
        raise OSRMError("OSRM a retourné une matrice de dimensions inattendues.")
    if any(not isinstance(row, list) or len(row) != len(stops) for row in raw_distances):
        raise OSRMError("OSRM a retourné une matrice de dimensions inattendues.")
    if any(value is None for row in raw_distances for value in row):
        raise OSRMError("OSRM n'a pas trouvé de route pour chaque paire de points.")
    return [[float(value) / 1000 for value in row] for row in raw_distances]


def get_distance_matrix(
    stops: Sequence[Stop], prefer_osrm: bool = True
) -> tuple[list[list[float]], str, str | None]:
    if prefer_osrm:
        try:
            return get_osrm_distance_matrix(stops), "OSRM (distance routière)", None
        except OSRMError as error:
            return (
                haversine_matrix(stops),
                "Haversine (distance à vol d'oiseau)",
                str(error),
            )
    return haversine_matrix(stops), "Haversine (distance à vol d'oiseau)", None


def get_osrm_route(
    stops: Sequence[Stop], route: Sequence[int], base_url: str = OSRM_BASE_URL
) -> RoadRoute:
    ordered_stops = [stops[index] for index in (*route, route[0])]
    coordinates = ";".join(
        f"{stop.longitude},{stop.latitude}" for stop in ordered_stops
    )
    try:
        response = requests.get(
            f"{base_url}/route/v1/driving/{coordinates}",
            params={
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
                "continue_straight": "false",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise OSRMError(f"Impossible de récupérer le tracé OSRM : {error}") from error

    if payload.get("code") != "Ok" or not payload.get("routes"):
        raise OSRMError(f"OSRM n'a pas calculé le tracé : {payload.get('message', payload.get('code'))}")
    route_data = payload["routes"][0]
    try:
        coordinates_lon_lat = route_data["geometry"]["coordinates"]
        coordinates_lat_lon = tuple(
            (float(latitude), float(longitude))
            for longitude, latitude in coordinates_lon_lat
        )
        distance_km = float(route_data["distance"]) / 1000
        duration_minutes = float(route_data["duration"]) / 60
    except (KeyError, TypeError, ValueError) as error:
        raise OSRMError("Le tracé retourné par OSRM est incomplet ou invalide.") from error
    return RoadRoute(coordinates_lat_lon, distance_km, duration_minutes)
