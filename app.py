"""Tkinter desktop dashboard for the 40-stop Agadir TSP demo."""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import folium
import optuna
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from agents import AgentRecommendation, run_research_agents
from routing import OSRMError, RoadRoute, get_distance_matrix, get_osrm_route
from tsp_core import (
    GAConfig,
    GAResult,
    Stop,
    make_demo_stops,
    parse_stops_csv,
    run_genetic_algorithm,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TSPDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TSP Agadir — zone dense")
        self.root.geometry("1440x920")
        self.root.minsize(1120, 760)

        self.stops = make_demo_stops()
        self.distance_matrix: list[list[float]] | None = None
        self.distance_source = ""
        self.result: GAResult | None = None
        self.config = GAConfig()
        self.road_route: RoadRoute | None = None
        self.route_error: str | None = None
        self.map_path: Path | None = None
        self.busy = False
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self.population_var = tk.StringVar(value="100")
        self.selection_var = tk.StringVar(value="Tournoi")
        self.mutation_var = tk.StringVar(value="0.2")
        self.crossover_var = tk.StringVar(value="0.8")
        self.generations_var = tk.StringVar(value="500")
        self.trials_var = tk.StringVar(value="5")
        self.status_var = tk.StringVar(value="Prêt — sélectionnez B1, B2 ou B3.")
        self.distance_var = tk.StringVar(value="Distance : —")
        self.source_var = tk.StringVar(value="Métrique : —")
        self.map_status_var = tk.StringVar(
            value="Après optimisation, ouvrez la carte interactive Folium dans le navigateur."
        )

        self._build_layout()
        self._draw_empty_charts()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        heading = ttk.Frame(self.root, padding=(12, 8))
        heading.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            heading,
            text="TSP Agadir — optimisation de tournée",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            heading,
            text=(
                "40 points en zone dense · OSRM / OpenStreetMap · "
                "algorithme génétique, Optuna et agents de recherche"
            ),
        ).pack(anchor="w", pady=(3, 0))

        grid = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        grid.grid(row=1, column=0, sticky="nsew")
        grid.columnconfigure((0, 1), weight=1, uniform="quadrants")
        grid.rowconfigure((0, 1), weight=1, uniform="quadrants")

        self.problem_frame = ttk.LabelFrame(
            grid, text="1 · Définition du problème", padding=10
        )
        self.solution_frame = ttk.LabelFrame(
            grid, text="2 · Convergence et solution", padding=10
        )
        self.cartesian_frame = ttk.LabelFrame(
            grid, text="3 · Plan cartésien TSP", padding=10
        )
        self.map_frame = ttk.LabelFrame(
            grid, text="4 · Routage réel — OSRM / OpenStreetMap", padding=10
        )
        self.problem_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self.solution_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        self.cartesian_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self.map_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=5)
        for frame in (
            self.problem_frame,
            self.solution_frame,
            self.cartesian_frame,
            self.map_frame,
        ):
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)

        self._build_problem_controls()
        self._build_solution_view()
        self._build_cartesian_view()
        self._build_map_controls()

        footer = ttk.Frame(self.root, padding=(12, 4, 12, 10))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(
            row=0, column=0, sticky="w"
        )
        self.progress = ttk.Progressbar(
            footer, orient="horizontal", mode="determinate", maximum=100, length=320
        )
        self.progress.grid(row=0, column=1, sticky="e", padx=(12, 0))

    def _build_problem_controls(self) -> None:
        frame = self.problem_frame
        ttk.Label(frame, text="Ville : Agadir").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text=f"Nombre de points : {len(self.stops)}").grid(
            row=0, column=1, sticky="w", padx=(20, 0)
        )
        ttk.Label(
            frame,
            text=(
                "Le premier point est le dépôt. Les points intégrés sont synthétiques ; "
                "un CSV terrain peut les remplacer."
            ),
            wraplength=650,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 2))
        ttk.Label(
            frame,
            text=(
                "Les coordonnées sont envoyées au service public OSRM pour calculer les routes. "
                "N'importez pas de coordonnées sensibles."
            ),
            wraplength=650,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 7))
        ttk.Button(
            frame, text="Charger un CSV de 40 points…", command=self._load_csv
        ).grid(row=3, column=0, sticky="w")
        ttk.Label(
            frame,
            text="Colonnes : id, latitude, longitude, name (facultatif) · WGS84",
        ).grid(row=3, column=1, sticky="w", padx=(10, 0))

        controls = ttk.Frame(frame)
        controls.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        definitions = (
            ("Population", self.population_var, ("50", "100", "500")),
            ("Sélection", self.selection_var, ("Roue biaisée", "Tournoi")),
            ("Mutation", self.mutation_var, ("0.2", "0.5", "0.8")),
            ("Crossover", self.crossover_var, ("0.2", "0.5", "0.8")),
            ("Itérations", self.generations_var, ("100", "500", "1000")),
            ("Essais Optuna", self.trials_var, ("3", "5", "10")),
        )
        for column, (label, variable, values) in enumerate(definitions):
            controls.columnconfigure(column, weight=1)
            ttk.Label(controls, text=label).grid(
                row=0, column=column, sticky="w", padx=3
            )
            ttk.Combobox(
                controls,
                textvariable=variable,
                values=values,
                state="readonly",
                width=13,
            ).grid(row=1, column=column, sticky="ew", padx=3, pady=(2, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        buttons.columnconfigure((0, 1, 2), weight=1)
        self.run_buttons = []
        for column, (text, mode) in enumerate(
            (
                ("B1 · Réglage manuel", "manual"),
                ("B2 · Optuna", "optuna"),
                ("B3 · Agents IA", "agents"),
            )
        ):
            button = ttk.Button(
                buttons,
                text=text,
                command=lambda selected_mode=mode: self._start(selected_mode),
            )
            button.grid(row=0, column=column, sticky="ew", padx=3)
            self.run_buttons.append(button)

        self.agent_text = tk.Text(frame, height=4, wrap="word", state="disabled")
        self.agent_text.grid(
            row=6, column=0, columnspan=2, sticky="nsew", pady=(5, 0)
        )
        frame.rowconfigure(6, weight=1)

    def _build_solution_view(self) -> None:
        frame = self.solution_frame
        ttk.Label(frame, textvariable=self.distance_var, font=("Segoe UI", 11, "bold")).pack(
            anchor="w"
        )
        ttk.Label(frame, textvariable=self.source_var).pack(anchor="w", pady=(1, 3))
        self.solution_figure = Figure(figsize=(6, 2.5), dpi=90, tight_layout=True)
        self.solution_axis = self.solution_figure.add_subplot(111)
        self.solution_canvas = FigureCanvasTkAgg(
            self.solution_figure, master=frame
        )
        self.solution_canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Label(frame, text="Chromosome solution — retour au dépôt inclus").pack(
            anchor="w", pady=(4, 0)
        )
        self.chromosome_text = tk.Text(frame, height=3, wrap="word", state="disabled")
        self.chromosome_text.pack(fill="x", expand=False)

    def _build_cartesian_view(self) -> None:
        self.cartesian_figure = Figure(figsize=(6, 3.2), dpi=90, tight_layout=True)
        self.cartesian_axis = self.cartesian_figure.add_subplot(111)
        self.cartesian_canvas = FigureCanvasTkAgg(
            self.cartesian_figure, master=self.cartesian_frame
        )
        self.cartesian_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_map_controls(self) -> None:
        frame = self.map_frame
        ttk.Label(
            frame,
            text=(
                "La carte Folium interactive s'ouvre dans le navigateur. "
                "Les marqueurs indiquent l'ordre de visite ; le dépôt est étoilé."
            ),
            wraplength=650,
        ).pack(anchor="w", pady=(2, 12))
        ttk.Label(
            frame,
            textvariable=self.map_status_var,
            wraplength=650,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="w")
        ttk.Button(
            buttons,
            text="Ouvrir la carte Folium",
            command=self._open_map,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            buttons,
            text="Enregistrer la carte HTML…",
            command=self._save_map,
        ).pack(side="left")
        ttk.Label(
            frame,
            text="Fond cartographique © OpenStreetMap · itinéraire routier © OSRM",
        ).pack(anchor="w", side="bottom", pady=(12, 0))

    def _draw_empty_charts(self) -> None:
        self.solution_axis.set_title("Lancez B1, B2 ou B3 pour optimiser")
        self.solution_axis.set_xlabel("Itération")
        self.solution_axis.set_ylabel("Distance (km)")
        self.solution_canvas.draw_idle()
        self.cartesian_axis.set_title("Itinéraire TSP — coordonnées Agadir")
        self.cartesian_axis.set_xlabel("Longitude")
        self.cartesian_axis.set_ylabel("Latitude")
        self.cartesian_canvas.draw_idle()

    def _load_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Choisir les coordonnées des 40 points",
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
                stops = parse_stops_csv(csv_file.read())
        except (OSError, UnicodeDecodeError, ValueError) as error:
            messagebox.showerror("CSV invalide", str(error))
            return

        self.stops = stops
        self.distance_matrix = None
        self.result = None
        self.road_route = None
        self.route_error = None
        self._remove_map_file()
        self.status_var.set(f"40 points chargés depuis {Path(path).name}.")
        self.distance_var.set("Distance : —")
        self.source_var.set("Métrique : —")
        self.map_status_var.set("Les points sont prêts ; lancez l'optimisation.")
        self._set_text(self.chromosome_text, "")
        self._clear_agent_report()
        self._draw_empty_charts()

    def _selected_config(self) -> GAConfig:
        return GAConfig(
            population_size=int(self.population_var.get()),
            selection=self.selection_var.get(),
            mutation_rate=float(self.mutation_var.get()),
            crossover_rate=float(self.crossover_var.get()),
            generations=int(self.generations_var.get()),
        )

    def _start(self, mode: str) -> None:
        if self.busy:
            return
        try:
            config = self._selected_config()
        except ValueError as error:
            messagebox.showerror("Paramètres invalides", str(error))
            return

        self.busy = True
        self.progress.configure(value=0)
        self.status_var.set("Préparation du calcul…")
        for button in self.run_buttons:
            button.configure(state="disabled")
        stops = tuple(self.stops)
        cached_matrix = self.distance_matrix
        trials = int(self.trials_var.get())
        threading.Thread(
            target=self._worker,
            args=(mode, config, stops, cached_matrix, trials),
            daemon=True,
        ).start()
        self.root.after(100, self._poll_events)

    def _worker(
        self,
        mode: str,
        manual_config: GAConfig,
        stops: tuple[Stop, ...],
        cached_matrix: list[list[float]] | None,
        trials: int,
    ) -> None:
        try:
            config = manual_config
            recommendation: AgentRecommendation | None = None
            if mode == "agents":
                self.events.put(("status", "Agents Web et Semantic Scholar en recherche…"))
                recommendation = run_research_agents()
                config = recommendation.config
                self.events.put(("agents", recommendation))

            if cached_matrix is None:
                self.events.put(("status", "Calcul de la matrice des distances OSRM…"))
                matrix, source, matrix_warning = get_distance_matrix(stops)
            else:
                matrix, source, matrix_warning = (
                    cached_matrix,
                    self.distance_source,
                    None,
                )
            if matrix_warning:
                self.events.put(("warning", matrix_warning))

            if mode == "optuna":
                result, config = self._optimize_with_optuna(matrix, trials)
            else:
                self.events.put(("progress", (0, "Optimisation génétique : 0 %")))

                def report_progress(generation: int, total: int, distance: float) -> None:
                    percent = min(95, 5 + int(generation * 90 / total))
                    self.events.put(
                        (
                            "progress",
                            (
                                percent,
                                f"Itération {generation}/{total} · "
                                f"meilleure distance : {distance:.2f} km",
                            ),
                        )
                    )

                result = run_genetic_algorithm(matrix, config, report_progress)

            self.events.put(("progress", (96, "Récupération du tracé routier OSRM…")))
            try:
                road_route = get_osrm_route(stops, result.route)
                route_error = None
            except OSRMError as error:
                road_route = None
                route_error = str(error)
            self.events.put(
                (
                    "complete",
                    {
                        "stops": stops,
                        "matrix": matrix,
                        "source": source,
                        "matrix_warning": matrix_warning,
                        "result": result,
                        "config": config,
                        "road_route": road_route,
                        "route_error": route_error,
                        "recommendation": recommendation,
                    },
                )
            )
        except Exception as error:
            logging.exception("Échec du calcul TSP")
            self.events.put(("error", str(error)))

    def _optimize_with_optuna(
        self,
        matrix: list[list[float]],
        trials: int,
    ) -> tuple[GAResult, GAConfig]:
        study = optuna.create_study(direction="minimize")
        results: dict[int, GAResult] = {}

        def objective(trial: optuna.Trial) -> float:
            config = GAConfig(
                population_size=trial.suggest_categorical(
                    "population_size", [50, 100, 500]
                ),
                selection=trial.suggest_categorical(
                    "selection", ["Roue biaisée", "Tournoi"]
                ),
                mutation_rate=trial.suggest_categorical(
                    "mutation_rate", [0.2, 0.5, 0.8]
                ),
                crossover_rate=trial.suggest_categorical(
                    "crossover_rate", [0.2, 0.5, 0.8]
                ),
                generations=trial.suggest_categorical(
                    "generations", [100, 500, 1000]
                ),
                seed=42 + trial.number,
            )

            def report_progress(generation: int, total: int, distance: float) -> None:
                fraction = (trial.number + generation / total) / trials
                percent = min(95, 5 + int(fraction * 85))
                self.events.put(
                    (
                        "progress",
                        (
                            percent,
                            f"Optuna · essai {trial.number + 1}/{trials} · "
                            f"meilleure distance : {distance:.2f} km",
                        ),
                    )
                )

            result = run_genetic_algorithm(matrix, config, report_progress)
            results[trial.number] = result
            return result.distance_km

        self.events.put(("status", f"Optuna lance {trials} essais…"))
        study.optimize(objective, n_trials=trials)
        best_trial = study.best_trial
        best_config = GAConfig(**best_trial.params, seed=42 + best_trial.number)
        self.events.put(
            ("status", f"Optuna terminé · meilleure distance : {study.best_value:.2f} km")
        )
        return results[best_trial.number], best_config

    def _poll_events(self) -> None:
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if event == "status":
                self.status_var.set(str(payload))
            elif event == "progress":
                percent, text = payload
                self.progress.configure(value=percent)
                self.status_var.set(text)
            elif event == "warning":
                self.status_var.set(f"Repli de distance : {payload}")
            elif event == "agents":
                self._show_agent_report(payload)
            elif event == "complete":
                self._show_result(payload)
                self.busy = False
                for button in self.run_buttons:
                    button.configure(state="normal")
            elif event == "error":
                self.busy = False
                for button in self.run_buttons:
                    button.configure(state="normal")
                self.progress.configure(value=0)
                self.status_var.set(f"Échec : {payload}")
                messagebox.showerror("Erreur pendant l'optimisation", str(payload))

        if self.busy:
            self.root.after(100, self._poll_events)

    def _show_result(self, payload: dict[str, object]) -> None:
        self.stops = list(payload["stops"])
        self.distance_matrix = payload["matrix"]
        self.distance_source = str(payload["source"])
        matrix_warning = payload["matrix_warning"]
        self.result = payload["result"]
        self.config = payload["config"]
        self.road_route = payload["road_route"]
        self.route_error = payload["route_error"]
        self.population_var.set(str(self.config.population_size))
        self.selection_var.set(self.config.selection)
        self.mutation_var.set(str(self.config.mutation_rate))
        self.crossover_var.set(str(self.config.crossover_rate))
        self.generations_var.set(str(self.config.generations))

        self.progress.configure(value=100)
        self.distance_var.set(f"Distance optimisée : {self.result.distance_km:.2f} km")
        self.source_var.set(f"Métrique : {self.distance_source}")
        if matrix_warning:
            self.status_var.set(
                f"Optimisation terminée — repli Haversine : {matrix_warning}"
            )
        else:
            self.status_var.set("Optimisation terminée — 100 %.")
        self._draw_results()
        self._create_map_file()

    def _draw_results(self) -> None:
        if self.result is None:
            return
        result = self.result
        stops = self.stops
        self.solution_axis.clear()
        self.solution_axis.plot(
            range(len(result.convergence_km)),
            result.convergence_km,
            color="#ff6b35",
            linewidth=2,
        )
        self.solution_axis.set_title("Courbe de convergence")
        self.solution_axis.set_xlabel("Itération")
        self.solution_axis.set_ylabel("Distance (km)")
        self.solution_axis.grid(alpha=0.25)
        self.solution_figure.tight_layout()
        self.solution_canvas.draw_idle()

        chromosome_ids = [stops[index].stop_id for index in result.route]
        chromosome = " → ".join(chromosome_ids + [chromosome_ids[0]])
        self._set_text(self.chromosome_text, chromosome)

        self.cartesian_axis.clear()
        route = [stops[index] for index in (*result.route, result.route[0])]
        self.cartesian_axis.plot(
            [stop.longitude for stop in route],
            [stop.latitude for stop in route],
            color="#ff6b35",
            linewidth=1.5,
            zorder=1,
        )
        self.cartesian_axis.scatter(
            [stop.longitude for stop in stops[1:]],
            [stop.latitude for stop in stops[1:]],
            color="#1f77b4",
            s=28,
            zorder=2,
            label="Points",
        )
        for stop in stops[1:]:
            self.cartesian_axis.annotate(
                stop.stop_id,
                (stop.longitude, stop.latitude),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
            )
        depot = stops[0]
        self.cartesian_axis.scatter(
            [depot.longitude],
            [depot.latitude],
            marker="*",
            color="red",
            edgecolor="black",
            s=180,
            zorder=3,
            label="Dépôt",
        )
        self.cartesian_axis.set_title("Ordre de visite — dépôt en étoile")
        self.cartesian_axis.set_xlabel("Longitude")
        self.cartesian_axis.set_ylabel("Latitude")
        self.cartesian_axis.grid(alpha=0.2)
        self.cartesian_axis.legend(loc="best")
        self.cartesian_figure.tight_layout()
        self.cartesian_canvas.draw_idle()

    def _create_map_file(self) -> None:
        if self.result is None:
            return
        stops = self.stops
        center = [stops[0].latitude, stops[0].longitude]
        route_map = folium.Map(location=center, zoom_start=14, control_scale=True)
        visit_order = {
            stop_index: order + 1 for order, stop_index in enumerate(self.result.route)
        }
        for index, stop in enumerate(stops):
            if index == 0:
                marker = folium.Marker(
                    [stop.latitude, stop.longitude],
                    tooltip=f"★ Dépôt · {stop.name}",
                    popup=f"Point {stop.stop_id} — {stop.name}",
                    icon=folium.Icon(color="red", icon="star", prefix="fa"),
                )
            else:
                order = visit_order[index]
                marker = folium.Marker(
                    [stop.latitude, stop.longitude],
                    tooltip=f"Étape {order} · {stop.name}",
                    popup=f"Étape {order} — point {stop.stop_id}",
                    icon=folium.DivIcon(
                        html=(
                            '<div style="background:#1769aa;color:white;'
                            'border-radius:50%;width:26px;height:26px;'
                            'text-align:center;line-height:26px;font-weight:bold;'
                            f'border:2px solid white">{order}</div>'
                        )
                    ),
                )
            marker.add_to(route_map)

        if self.road_route:
            route_map.add_child(
                folium.PolyLine(
                    self.road_route.coordinates,
                    color="#ff6b35",
                    weight=4,
                    opacity=0.9,
                    tooltip=(
                        f"OSRM · {self.road_route.distance_km:.2f} km · "
                        f"{self.road_route.duration_minutes:.0f} min"
                    ),
                )
            )
            route_map.fit_bounds(self.road_route.coordinates)
            self.map_status_var.set(
                f"Tracé routier OSRM : {self.road_route.distance_km:.2f} km, "
                f"{self.road_route.duration_minutes:.0f} min estimées."
            )
        else:
            path = [stops[index] for index in (*self.result.route, 0)]
            route_map.add_child(
                folium.PolyLine(
                    [(stop.latitude, stop.longitude) for stop in path],
                    color="#ff6b35",
                    weight=3,
                    opacity=0.8,
                    tooltip="Segments droits — tracé OSRM indisponible",
                )
            )
            route_map.fit_bounds([(stop.latitude, stop.longitude) for stop in stops])
            self.map_status_var.set(
                "OSRM indisponible ; la carte montre l'ordre des points en segments droits. "
                f"{self.route_error or ''}"
            )

        self._remove_map_file()
        with tempfile.NamedTemporaryFile(
            prefix="tsp_agadir_", suffix=".html", delete=False
        ) as temp_file:
            self.map_path = Path(temp_file.name)
        route_map.save(str(self.map_path))

    def _open_map(self) -> None:
        if self.map_path is None:
            messagebox.showinfo("Carte Folium", "Lancez d'abord l'optimisation.")
            return
        webbrowser.open(self.map_path.as_uri())

    def _save_map(self) -> None:
        if self.map_path is None:
            messagebox.showinfo("Carte Folium", "Lancez d'abord l'optimisation.")
            return
        destination = filedialog.asksaveasfilename(
            title="Enregistrer la carte interactive",
            defaultextension=".html",
            initialfile="tsp_agadir_route.html",
            filetypes=[("Page HTML", "*.html")],
        )
        if destination:
            try:
                with open(self.map_path, "rb") as source, open(destination, "wb") as target:
                    target.write(source.read())
            except OSError as error:
                messagebox.showerror("Enregistrement impossible", str(error))
            else:
                self.status_var.set(f"Carte Folium enregistrée : {destination}")

    def _show_agent_report(self, recommendation: AgentRecommendation) -> None:
        lines = [recommendation.rationale, ""]
        for finding in recommendation.findings:
            if finding.error:
                lines.append(f"{finding.agent} — indisponible : {finding.error}")
            else:
                lines.append(f"{finding.agent} — {finding.summary}")
                if finding.source_url:
                    lines.append(f"Source : {finding.source_url}")
        lines.extend(
            (
                "",
                "Le raisonneur utilise des règles heuristiques, sans modèle LLM.",
            )
        )
        self._set_text(self.agent_text, "\n".join(lines))

    def _clear_agent_report(self) -> None:
        self._set_text(
            self.agent_text,
            "B3 consulte les agents Web et Semantic Scholar puis affiche l'arbitrage.",
        )

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _remove_map_file(self) -> None:
        if self.map_path is not None:
            try:
                os.unlink(self.map_path)
            except FileNotFoundError:
                pass
            self.map_path = None

    def _close(self) -> None:
        self._remove_map_file()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TSPDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
