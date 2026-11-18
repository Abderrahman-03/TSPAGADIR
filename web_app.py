"""Four-quadrant Streamlit interface for the Agadir TSP application."""

from __future__ import annotations

from html import escape

import streamlit as st
import folium
import matplotlib.pyplot as plt
import optuna
import streamlit.components.v1 as components

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

optuna.logging.set_verbosity(optuna.logging.WARNING)

st.set_page_config(
    page_title="TSP Agadir | Optimisation de tournée",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #15324b;
        --teal: #138a83;
        --teal-soft: #e6f5f2;
        --orange: #e88745;
        --ink: #22394d;
        --muted: #64788a;
        --line: #dce6ee;
        --canvas: #f1f5f8;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(241,245,248,.94); }
    .block-container { max-width: 1600px; padding-top: 1.2rem; padding-bottom: 2rem; }
    .hero {
        position: relative; overflow: hidden;
        padding: 1.45rem 1.7rem; border-radius: 18px; margin-bottom: 1.15rem;
        color: white;
        background: linear-gradient(112deg,#12324b 0%,#155c67 60%,#19877c 100%);
        box-shadow: 0 10px 26px rgba(21,50,75,.16);
    }
    .hero h1 { margin: 0; font-size: 1.9rem; letter-spacing: -.025em; }
    .hero p { margin: .4rem 0 0; color: #d9eeed; font-size: 1rem; }
    h2, h3 { color: var(--navy); letter-spacing: -.015em; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff; border-radius: 16px; border-color: var(--line);
        box-shadow: 0 4px 14px rgba(21,50,75,.045);
    }
    [data-testid="stVerticalBlockBorderWrapper"] h3 {
        padding-bottom: .6rem; border-bottom: 1px solid #e8eef3;
    }
    div[data-testid="stMetric"] {
        min-height: 94px; background: #f7fafb; border: 1px solid #e3ebef;
        padding: .72rem .8rem; border-radius: 12px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--navy); }
    [data-testid="stMetricDelta"] { color: var(--teal); }
    .stButton > button {
        border-radius: 10px; min-height: 2.65rem; font-weight: 650;
        border-color: #c9d8e2; color: var(--navy); transition: all .15s ease;
    }
    .stButton > button:hover { border-color: var(--teal); color: var(--teal); }
    .stButton > button[kind="primary"] {
        background: var(--teal); border-color: var(--teal); color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background: #0e746f; border-color: #0e746f; color: white;
    }
    [data-testid="stProgressBar"] > div > div { background: var(--teal); }
    .result-note {
        color: var(--muted); font-size: .84rem; margin: -.25rem 0 .45rem;
    }
    .route-strip {
        display: flex; flex-wrap: wrap; align-items: center; gap: .35rem;
        padding: .8rem; background: #f7fafb; border: 1px solid #e3ebef;
        border-radius: 12px; line-height: 1.6;
    }
    .route-chip {
        display: inline-block; padding: .12rem .5rem; border-radius: 999px;
        color: #14544f; background: var(--teal-soft); border: 1px solid #c9e8e2;
        font: 600 .78rem ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .route-chip.depot {
        color: #75431f; background: #fff1e5; border-color: #f4d0b1;
    }
    .route-arrow { color: #92a3ae; font-size: .78rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: .1rem;
    }
    @media (max-width: 900px) {
        .hero h1 { font-size: 1.5rem; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _initialize_state() -> None:
    defaults = {
        "tsp_stops": make_demo_stops(),
        "tsp_matrix": None,
        "tsp_distance_source": None,
        "tsp_result": None,
        "tsp_config": None,
        "tsp_road_route": None,
        "tsp_route_error": None,
        "tsp_matrix_warning": None,
        "tsp_recommendation": None,
        "tsp_agent_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_csv(uploaded_file) -> None:
    if uploaded_file is None:
        return
    try:
        stops = parse_stops_csv(uploaded_file.getvalue().decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as error:
        st.error(f"Fichier CSV invalide : {error}")
        return
    if stops != st.session_state.tsp_stops:
        st.session_state.tsp_stops = stops
        st.session_state.tsp_matrix = None
        st.session_state.tsp_result = None
        st.session_state.tsp_road_route = None
        st.session_state.tsp_route_error = None
        st.success("Les 40 coordonnées du CSV sont chargées.")


def _progress_callback(progress, status, generations: int, trial_offset: int = 0, trials: int = 1):
    def update(generation: int, _total: int, distance: float) -> None:
        if generation == 1 or generation % 5 == 0 or generation == generations:
            percentage = min(
                94,
                5 + int(((trial_offset + generation / generations) / trials) * 85),
            )
            progress.progress(percentage, text=f"Progression : {percentage} %")
            status.caption(
                f"Itération {generation}/{generations} · meilleure distance : {distance:.2f} km"
            )

    return update


def _run_optimization(mode: str, manual_config: GAConfig, trials: int = 5) -> None:
    progress = st.progress(0, text="Préparation : 0 %")
    status = st.empty()
    stops: list[Stop] = st.session_state.tsp_stops
    recommendation: AgentRecommendation | None = None

    try:
        config = manual_config
        if mode == "agents":
            status.caption("Les agents Web et Semantic Scholar recherchent des recommandations…")
            recommendation = run_research_agents()
            st.session_state.tsp_recommendation = recommendation
            config = recommendation.config

        matrix = st.session_state.tsp_matrix
        if matrix is None:
            status.caption("Calcul de la matrice des distances OSRM…")
            matrix, source, warning = get_distance_matrix(stops)
            st.session_state.tsp_matrix = matrix
            st.session_state.tsp_distance_source = source
            st.session_state.tsp_matrix_warning = warning
        else:
            source = st.session_state.tsp_distance_source
            warning = st.session_state.tsp_matrix_warning

        if warning:
            st.warning(
                f"OSRM indisponible pour la matrice : {warning} "
                "Les distances à vol d’oiseau sont utilisées."
            )

        if mode == "optuna":
            study = optuna.create_study(direction="minimize")
            trial_results: dict[int, tuple[GAResult, GAConfig]] = {}

            def objective(trial: optuna.Trial) -> float:
                trial_config = GAConfig(
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
                result = run_genetic_algorithm(
                    matrix,
                    trial_config,
                    _progress_callback(
                        progress, status, trial_config.generations, trial.number, trials
                    ),
                )
                trial_results[trial.number] = (result, trial_config)
                return result.distance_km

            status.caption(f"Optuna lance {trials} essais…")
            study.optimize(objective, n_trials=trials)
            result, config = trial_results[study.best_trial.number]
            st.session_state.tsp_optuna_params = study.best_trial.params
        else:
            status.caption("Optimisation génétique en cours…")
            result = run_genetic_algorithm(
                matrix,
                config,
                _progress_callback(progress, status, config.generations),
            )

        progress.progress(96, text="Récupération du trajet réel OSRM : 96 %")
        try:
            road_route = get_osrm_route(stops, result.route)
            route_error = None
        except OSRMError as error:
            road_route = None
            route_error = str(error)

        st.session_state.tsp_result = result
        st.session_state.tsp_config = config
        st.session_state.tsp_road_route = road_route
        st.session_state.tsp_route_error = route_error
        progress.progress(100, text="Terminé : 100 %")
        status.caption("Optimisation terminée.")
    except (ValueError, RuntimeError) as error:
        st.error(f"Impossible de terminer l’optimisation : {error}")
    except Exception as error:
        st.error(f"Erreur inattendue pendant l’optimisation : {error}")
        raise


def _show_agents() -> None:
    recommendation: AgentRecommendation | None = st.session_state.tsp_recommendation
    if recommendation is None:
        return
    st.info(recommendation.rationale)
    for finding in recommendation.findings:
        if finding.error:
            st.warning(f"{finding.agent} indisponible : {finding.error}")
        else:
            st.markdown(f"**{finding.agent}** — {finding.summary}")
            if finding.source_url:
                st.caption(f"Source : {finding.source_url}")


def _show_convergence(result: GAResult) -> None:
    figure, axis = plt.subplots(figsize=(7, 2.7), dpi=110)
    iterations = range(len(result.convergence_km))
    axis.set_facecolor("#f8fafb")
    figure.patch.set_facecolor("white")
    axis.fill_between(iterations, result.convergence_km, color="#138a83", alpha=0.11)
    axis.plot(iterations, result.convergence_km, color="#138a83", linewidth=2.2)
    axis.scatter(
        [0, len(result.convergence_km) - 1],
        [result.convergence_km[0], result.convergence_km[-1]],
        color=["#e88745", "#138a83"],
        edgecolors="white",
        linewidths=1.2,
        s=42,
        zorder=3,
    )
    axis.set_xlabel("Itération", color="#64788a")
    axis.set_ylabel("Distance (km)", color="#64788a")
    axis.tick_params(colors="#64788a", labelsize=8)
    axis.grid(color="#dce6ee", linewidth=.7, alpha=.8)
    axis.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#dce6ee")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def _show_cartesian(stops: list[Stop], result: GAResult) -> None:
    figure, axis = plt.subplots(figsize=(7, 4), dpi=110)
    axis.set_facecolor("#f8fafb")
    figure.patch.set_facecolor("white")
    route = [stops[index] for index in (*result.route, result.route[0])]
    axis.plot(
        [stop.longitude for stop in route],
        [stop.latitude for stop in route],
        color="#e88745",
        linewidth=1.9,
        alpha=.9,
        zorder=1,
    )
    axis.scatter(
        [stop.longitude for stop in stops[1:]],
        [stop.latitude for stop in stops[1:]],
        color="#138a83",
        edgecolors="white",
        linewidths=.6,
        s=30,
        zorder=2,
    )
    for stop in stops[1:]:
        axis.annotate(
            stop.stop_id,
            (stop.longitude, stop.latitude),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
            color="#385268",
        )
    depot = stops[0]
    axis.scatter(
        depot.longitude,
        depot.latitude,
        marker="*",
        color="#e88745",
        edgecolor="#75431f",
        linewidth=1,
        s=200,
        zorder=3,
        label="★ Entrepôt",
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.tick_params(colors="#64788a", labelsize=8)
    axis.grid(color="#dce6ee", linewidth=.7, alpha=.8)
    axis.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#dce6ee")
    axis.legend(loc="best")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def _show_folium_map(stops: list[Stop], result: GAResult, road_route: RoadRoute | None):
    route_map = folium.Map(
        location=[stops[0].latitude, stops[0].longitude],
        zoom_start=14,
        control_scale=True,
    )
    visit_order = {
        index: order + 1 for order, index in enumerate(result.route)
    }
    for index, stop in enumerate(stops):
        if index == 0:
            folium.Marker(
                [stop.latitude, stop.longitude],
                tooltip=f"★ Entrepôt · {stop.name}",
                popup=f"Point {stop.stop_id} — {stop.name}",
                icon=folium.Icon(color="orange", icon="star", prefix="fa"),
            ).add_to(route_map)
        else:
            order = visit_order[index]
            folium.Marker(
                [stop.latitude, stop.longitude],
                tooltip=f"Étape {order} · {stop.name}",
                popup=f"Étape {order} — point {stop.stop_id}",
                icon=folium.DivIcon(
                    html=(
                        '<div style="background:#138a83;color:white;'
                        'border-radius:50%;width:26px;height:26px;'
                        'text-align:center;line-height:26px;font-weight:bold;'
                        f'border:2px solid white">{order}</div>'
                    )
                ),
            ).add_to(route_map)

    if road_route:
        folium.PolyLine(
            road_route.coordinates,
            color="#e88745",
            weight=5,
            opacity=0.9,
            tooltip=(
                f"OSRM · {road_route.distance_km:.2f} km · "
                f"{road_route.duration_minutes:.0f} min"
            ),
        ).add_to(route_map)
        route_map.fit_bounds(road_route.coordinates)
    else:
        route = [stops[index] for index in (*result.route, result.route[0])]
        folium.PolyLine(
            [(stop.latitude, stop.longitude) for stop in route],
            color="#e88745",
            weight=3,
            tooltip="Segments droits — tracé OSRM indisponible",
        ).add_to(route_map)
        route_map.fit_bounds([(stop.latitude, stop.longitude) for stop in stops])
    return route_map


def main() -> None:
    _initialize_state()
    st.markdown(
        """
        <div class="hero">
          <h1>🚚 TSP Agadir — optimisation de tournée</h1>
          <p>40 points · algorithme génétique · optimisation OSRM · carte OpenStreetMap</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upper_left, upper_right = st.columns(2, gap="medium")
    with upper_left:
        with st.container(border=True):
            st.subheader("1 · Définition du problème")
            city_col, count_col = st.columns(2)
            city_col.text_input("Ville", value="Agadir", disabled=True)
            count_col.metric("Nombre de points", len(st.session_state.tsp_stops))
            st.caption(
                "Le premier point (ou la première ligne du CSV) est l’entrepôt. "
                "Les 40 coordonnées intégrées sont synthétiques."
            )
            uploaded_file = st.file_uploader(
                "Importer 40 points CSV",
                type=["csv"],
                help="Colonnes requises : id, latitude, longitude ; name est facultative.",
            )
            _load_csv(uploaded_file)
            st.caption(
                "Les coordonnées sont envoyées au service public OSRM. "
                "N’importez pas de données de localisation sensibles."
            )

            settings_left, settings_right = st.columns(2)
            with settings_left:
                population = st.selectbox("Taille de population", [50, 100, 500], index=1)
                selection = st.selectbox(
                    "Sélection", ["Roue biaisée", "Tournoi"], index=1
                )
                mutation = st.selectbox("Taux de mutation", [0.2, 0.5, 0.8], index=0)
            with settings_right:
                crossover = st.selectbox("Taux de crossover", [0.2, 0.5, 0.8], index=2)
                generations = st.selectbox(
                    "Nombre d’itérations", [100, 500, 1000], index=1
                )
                trials = st.selectbox("Essais Optuna (B2)", [3, 5, 10], index=1)

            manual_config = GAConfig(
                population_size=population,
                selection=selection,
                mutation_rate=mutation,
                crossover_rate=crossover,
                generations=generations,
            )
            b1, b2, b3 = st.columns(3)
            manual_clicked = b1.button(
                "⚡ B1 · Manuel", key="web_b1", type="primary", use_container_width=True
            )
            optuna_clicked = b2.button(
                "🧪 B2 · Optuna", key="web_b2", use_container_width=True
            )
            agents_clicked = b3.button(
                "🤖 B3 · Agents IA", key="web_b3", use_container_width=True
            )

            if manual_clicked:
                _run_optimization("manual", manual_config)
            elif optuna_clicked:
                _run_optimization("optuna", manual_config, trials=trials)
            elif agents_clicked:
                _run_optimization("agents", manual_config)

            _show_agents()

    with upper_right:
        with st.container(border=True):
            st.subheader("2 · Convergence, métrique et chromosome")
            result: GAResult | None = st.session_state.tsp_result
            if result is None:
                st.info("Lancez B1, B2 ou B3 pour calculer la tournée.")
            else:
                config: GAConfig = st.session_state.tsp_config
                initial_distance = result.convergence_km[0]
                gain = (
                    (initial_distance - result.distance_km) / initial_distance * 100
                    if initial_distance > 0
                    else 0
                )
                distance_metric, time_metric, gain_metric, iteration_metric = st.columns(4)
                distance_metric.metric("Distance", f"{result.distance_km:.2f} km")
                road_route: RoadRoute | None = st.session_state.tsp_road_route
                time_metric.metric(
                    "Temps estimé",
                    f"{road_route.duration_minutes:.0f} min" if road_route else "—",
                )
                gain_metric.metric("Amélioration", f"{gain:.1f} %")
                iteration_metric.metric("Itérations", f"{config.generations:,}".replace(",", " "))
                st.markdown(
                    f'<div class="result-note">Métrique : '
                    f'<strong>{escape(st.session_state.tsp_distance_source or "—")}</strong>'
                    " · Le chromosome retourne au dépôt.</div>",
                    unsafe_allow_html=True,
                )
                if st.session_state.tsp_matrix_warning:
                    st.warning(
                        "Métrique de repli : distance géodésique à vol d’oiseau, "
                        "OSRM indisponible."
                    )
                _show_convergence(result)
                st.markdown("**Chromosome solution**")
                stop_ids = [
                    st.session_state.tsp_stops[index].stop_id
                    for index in result.route
                ]
                route_items = []
                last_index = len(stop_ids) - 1
                for position, stop_id in enumerate(stop_ids + [stop_ids[0]]):
                    depot_class = " depot" if position in (0, last_index + 1) else ""
                    prefix = "★ " if position == 0 else ""
                    route_items.append(
                        f'<span class="route-chip{depot_class}">'
                        f'{prefix}{escape(stop_id)}</span>'
                    )
                    if position < len(stop_ids):
                        route_items.append('<span class="route-arrow">→</span>')
                st.markdown(
                    f'<div class="route-strip">{"".join(route_items)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="result-note" style="margin-top:.55rem">'
                    f'Population <strong>{config.population_size}</strong> · '
                    f'{escape(config.selection)} · mutation <strong>{config.mutation_rate:.1f}</strong> · '
                    f'crossover <strong>{config.crossover_rate:.1f}</strong> · '
                    f'graine <strong>{config.seed}</strong></div>',
                    unsafe_allow_html=True,
                )
                params = st.session_state.get("tsp_optuna_params")
                if params:
                    with st.expander("Meilleurs paramètres Optuna"):
                        st.json(params)

    lower_left, lower_right = st.columns(2, gap="medium")
    with lower_left:
        with st.container(border=True):
            st.subheader("3 · Plan cartésien TSP")
            result = st.session_state.tsp_result
            if result is None:
                st.info("Le plan apparaîtra après l’optimisation.")
            else:
                _show_cartesian(st.session_state.tsp_stops, result)

    with lower_right:
        with st.container(border=True):
            st.subheader("4 · Routage réel — OSRM / OpenStreetMap")
            result = st.session_state.tsp_result
            if result is None:
                st.info("La carte routière apparaîtra après l’optimisation.")
            else:
                road_route: RoadRoute | None = st.session_state.tsp_road_route
                if road_route:
                    st.caption(
                        f"OSRM · {road_route.distance_km:.2f} km · "
                        f"{road_route.duration_minutes:.0f} min estimées"
                    )
                else:
                    st.warning(
                        f"Tracé OSRM indisponible : {st.session_state.tsp_route_error}. "
                        "La carte affiche des segments droits, pas les rues."
                    )
                route_map = _show_folium_map(
                    st.session_state.tsp_stops, result, road_route
                )
                components.html(route_map.get_root().render(), height=470, scrolling=False)
                st.caption(
                    "Dépôt étoilé · étapes numérotées · carte © OpenStreetMap · "
                    "routage © OSRM"
                )


if __name__ == "__main__":
    main()
