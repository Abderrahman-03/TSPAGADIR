"""Four-quadrant Streamlit interface for the Agadir TSP application."""

from __future__ import annotations

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
    .stApp { background: #f5f7fb; }
    [data-testid="stHeader"] { background: rgba(245,247,251,.9); }
    .block-container { max-width: 1560px; padding-top: 1.4rem; }
    .hero {
        padding: 1.2rem 1.5rem; border-radius: 16px; margin-bottom: 1.15rem;
        color: white; background: linear-gradient(120deg,#102a43,#1769aa);
    }
    .hero h1 { margin: 0; font-size: 1.8rem; }
    .hero p { margin: .35rem 0 0; color: #dceaf6; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: white; border-radius: 14px; border-color: #e2e8f0;
    }
    div[data-testid="stMetric"] {
        background: #f5f8fc; border: 1px solid #e8eef5;
        padding: .6rem .8rem; border-radius: 10px;
    }
    .small-note { color: #526579; font-size: .88rem; }
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
    figure, axis = plt.subplots(figsize=(7, 2.8), dpi=110)
    axis.plot(result.convergence_km, color="#f36b3f", linewidth=2)
    axis.set_xlabel("Itération")
    axis.set_ylabel("Distance (km)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def _show_cartesian(stops: list[Stop], result: GAResult) -> None:
    figure, axis = plt.subplots(figsize=(7, 4), dpi=110)
    route = [stops[index] for index in (*result.route, result.route[0])]
    axis.plot(
        [stop.longitude for stop in route],
        [stop.latitude for stop in route],
        color="#f36b3f",
        linewidth=1.5,
        zorder=1,
    )
    axis.scatter(
        [stop.longitude for stop in stops[1:]],
        [stop.latitude for stop in stops[1:]],
        color="#1769aa",
        s=22,
        zorder=2,
    )
    for stop in stops[1:]:
        axis.annotate(
            stop.stop_id,
            (stop.longitude, stop.latitude),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
        )
    depot = stops[0]
    axis.scatter(
        depot.longitude,
        depot.latitude,
        marker="*",
        color="#df2525",
        edgecolor="black",
        s=190,
        zorder=3,
        label="★ Entrepôt",
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.2)
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
                icon=folium.Icon(color="red", icon="star", prefix="fa"),
            ).add_to(route_map)
        else:
            order = visit_order[index]
            folium.Marker(
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
            ).add_to(route_map)

    if road_route:
        folium.PolyLine(
            road_route.coordinates,
            color="#f36b3f",
            weight=4,
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
            color="#f36b3f",
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
                "B1 · Manuel", key="web_b1", use_container_width=True
            )
            optuna_clicked = b2.button(
                "B2 · Optuna", key="web_b2", use_container_width=True
            )
            agents_clicked = b3.button(
                "B3 · Agents IA", key="web_b3", use_container_width=True
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
                metric, source = st.columns(2)
                metric.metric("Distance optimisée", f"{result.distance_km:.2f} km")
                source.metric("Métrique", st.session_state.tsp_distance_source)
                if st.session_state.tsp_matrix_warning:
                    st.warning(
                        "Métrique de repli : distance géodésique à vol d’oiseau, "
                        "OSRM indisponible."
                    )
                _show_convergence(result)
                stop_ids = [
                    st.session_state.tsp_stops[index].stop_id
                    for index in result.route
                ]
                st.caption("Chromosome solution — retour à l’entrepôt inclus")
                st.code(" → ".join(stop_ids + [stop_ids[0]]), language=None)
                config: GAConfig = st.session_state.tsp_config
                st.caption(
                    f"Population {config.population_size} · {config.selection} · "
                    f"mutation {config.mutation_rate} · crossover {config.crossover_rate} · "
                    f"{config.generations} itérations"
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
