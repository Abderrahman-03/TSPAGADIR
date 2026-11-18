# TSP Agadir — zone dense

Application de bureau Tkinter pour optimiser une tournée cyclique de 40 points à Agadir. Les graphiques sont intégrés dans la fenêtre ; la carte interactive Folium s'ouvre dans le navigateur.

Une interface Web en quatre quadrants est également disponible via `web_app.py`.

## Fonctionnalités

- **B1 — Manuel** : population, sélection (tournoi ou roue biaisée), mutation, crossover et nombre d'itérations.
- **B2 — Optuna** : recherche dans les valeurs configurées et affichage du meilleur jeu de paramètres.
- **B3 — Agents IA** : recherche web Wikipédia et Semantic Scholar, puis arbitrage par un raisonneur heuristique explicable.
- Matrice des distances routières OSRM ; repli signalé sur les distances à vol d'oiseau si le service est indisponible.
- Courbe de convergence, chromosome solution, plan cartésien et carte OpenStreetMap avec le dépôt étoilé et les étapes numérotées.
- Import facultatif d'un CSV de 40 lignes avec `id,latitude,longitude` et `name` facultatif. La première ligne est le dépôt. Les coordonnées doivent être en WGS84.

Les 40 coordonnées intégrées sont des points de démonstration reproductibles répartis dans le centre dense d'Agadir ; elles ne représentent pas des adresses vérifiées. Utiliser le CSV pour des points terrain.

## Installation et lancement (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Pour lancer l'interface Web dans le navigateur :

```powershell
python -m streamlit run web_app.py
```

### Hébergement public avec Streamlit Community Cloud

Publier le projet dans un dépôt GitHub public, puis créer une application Streamlit Cloud en sélectionnant ce dépôt et le fichier principal `web_app.py`. Garder `requirements.txt` à la racine pour l'installation automatique des dépendances.

Le calcul routier et les agents de recherche nécessitent un accès Internet. Les coordonnées sont envoyées au service public OSRM pour établir les distances et les trajets. Le service OSRM et les APIs de recherche peuvent appliquer des limites ou être temporairement indisponibles.
