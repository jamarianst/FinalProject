import html
from typing import Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from textwrap import dedent


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ScoutAI Recruit",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --card-bg: rgba(25, 29, 38, 0.96);
        --soft-bg: rgba(255, 255, 255, 0.045);
        --border: rgba(255, 255, 255, 0.09);
        --muted: #9ca3af;
        --accent: #ff4b4b;
        --accent-soft: rgba(255, 75, 75, 0.13);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    .scoutai-title {
        font-size: clamp(2.3rem, 4vw, 3.4rem);
        font-weight: 850;
        letter-spacing: -0.06em;
        line-height: 1.05;
        margin-bottom: 0.3rem;
    }

    .scoutai-subtitle {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 900px;
        margin-bottom: 2rem;
    }

    .eyebrow {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.11rem;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }

    .player-card {
        background:
            linear-gradient(
                135deg,
                rgba(37, 42, 53, 0.98),
                rgba(18, 21, 28, 0.98)
            );
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.6rem 1.75rem;
        box-shadow: 0 14px 38px rgba(0, 0, 0, 0.22);
        margin-bottom: 1.8rem;
    }

    .player-name {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }

    .player-header-row {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .player-photo {
        width: 72px;
        height: 72px;
        border-radius: 16px;
        object-fit: cover;
        border: 1px solid var(--border);
        background: var(--soft-bg);
    }

    .club-logo {
        width: 24px;
        height: 24px;
        object-fit: contain;
        vertical-align: middle;
        margin-right: 0.4rem;
    }

    .player-club {
        color: var(--muted);
        font-size: 1rem;
        margin-top: 0.2rem;
    }

    .role-badge,
    .match-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .role-badge {
        margin-top: 0.8rem;
        background: var(--accent-soft);
        border: 1px solid rgba(255, 75, 75, 0.34);
        color: #ff8f8f;
    }

    .match-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .match-elite {
        background: rgba(38, 208, 124, 0.16);
        border: 1px solid rgba(38, 208, 124, 0.42);
        color: #8cf0ba;
    }

    .match-excellent {
        background: rgba(72, 187, 120, 0.12);
        border: 1px solid rgba(72, 187, 120, 0.28);
        color: #7de2a8;
    }

    .match-strong {
        background: rgba(250, 204, 21, 0.13);
        border: 1px solid rgba(250, 204, 21, 0.32);
        color: #fde68a;
    }

    .match-good {
        background: rgba(251, 146, 60, 0.13);
        border: 1px solid rgba(251, 146, 60, 0.32);
        color: #fdba74;
    }

    .match-possible {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: #cbd5e1;
    }

    .profile-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(150px, 1fr));
        gap: 0.9rem;
        margin-top: 1.35rem;
    }

    .profile-stat {
        background: var(--soft-bg);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 13px;
        padding: 0.9rem 1rem;
    }

    .profile-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.06rem;
        text-transform: uppercase;
    }

    .profile-value {
        font-size: 1.04rem;
        font-weight: 700;
        margin-top: 0.32rem;
        overflow-wrap: anywhere;
    }

    .recommendation-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.45rem 1.4rem;
        min-height: 285px;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18);
        transition: transform 0.18s ease, border-color 0.18s ease;
    }

    .recommendation-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.16);
    }

    .recommendation-name {
        font-size: 1.28rem;
        font-weight: 820;
        margin-bottom: 0.18rem;
        letter-spacing: -0.02em;
    }

    .recommendation-club {
        color: var(--muted);
        font-size: 0.9rem;
        min-height: 2.3rem;
    }

    .similarity-number {
        font-size: 2.55rem;
        font-weight: 880;
        letter-spacing: -0.055em;
        margin-top: 0.9rem;
        line-height: 1;
    }

    .mini-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
        margin-top: 1rem;
    }

    .mini-stat {
        background: var(--soft-bg);
        border-radius: 10px;
        padding: 0.65rem 0.75rem;
    }

    .mini-label {
        color: var(--muted);
        font-size: 0.68rem;
        text-transform: uppercase;
        font-weight: 750;
    }

    .mini-value {
        margin-top: 0.18rem;
        font-size: 0.9rem;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    .why-section {
        margin-top: 0.9rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
    }

    .why-text {
        color: #d1d5db;
        font-size: 0.86rem;
        line-height: 1.55;
        margin-top: 0.42rem;
    }

    .why-item {
        display: block;
        margin-bottom: 0.28rem;
    }

    .category-section {
        margin-top: 0.9rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
    }

    .category-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.45rem;
        margin-top: 0.5rem;
    }

    .category-item {
        background: var(--soft-bg);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 9px;
        padding: 0.5rem 0.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
    }

    .category-name {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
    }

    .category-score {
        font-size: 0.8rem;
        font-weight: 800;
        white-space: nowrap;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
    }

    @media (max-width: 900px) {
        .profile-grid {
            grid-template-columns: repeat(2, minmax(140px, 1fr));
        }
    }

    @media (max-width: 600px) {
        .profile-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("player_replacement_data.csv")


@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans.pkl")
    model_features = joblib.load("model_features.pkl")
    return scaler, kmeans, model_features


try:
    df = load_data()
    scaler, kmeans, model_features = load_models()
except FileNotFoundError as error:
    st.error(
        "A required project file could not be found. Make sure "
        "`player_replacement_data.csv`, `scaler.pkl`, `kmeans.pkl`, and "
        "`model_features.pkl` are in the app directory."
    )
    st.exception(error)
    st.stop()
except Exception as error:
    st.error("The app could not load the dataset or saved model files.")
    st.exception(error)
    st.stop()


required_columns = [
    "player_name",
    "club_name",
    "position_name",
    "cluster",
    "role",
    "market_value",
    "market_value_numeric",
]

missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    st.error(
        "The dataset is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


available_features = [
    feature for feature in list(model_features) if feature in df.columns
]

if not available_features:
    st.error("None of the saved model features were found in the dataset.")
    st.stop()


X = (
    df[available_features]
    .replace([np.inf, -np.inf], np.nan)
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
)

try:
    X_scaled = scaler.transform(X)
except Exception as error:
    st.error(
        "The scaler could not transform the dataset. Make sure the CSV and "
        "saved model files were created from the same notebook."
    )
    st.exception(error)
    st.stop()


# Phase 5C: precompute alternative distance spaces on the same standardized features.
similarity_matrix = cosine_similarity(X_scaled)
euclidean_matrix = euclidean_distances(X_scaled)
manhattan_matrix = manhattan_distances(X_scaled)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def safe_text(value, fallback: str = "Unknown") -> str:
    if pd.isna(value):
        return fallback

    text = str(value).strip()

    if text.lower() in {"", "none", "nan"}:
        return fallback

    return text


def safe_html(value, fallback: str = "Unknown") -> str:
    return html.escape(safe_text(value, fallback))


def compact_html(markup: str) -> str:
    """Remove Markdown-breaking indentation and blank lines from HTML."""
    return "".join(
        line.strip()
        for line in dedent(markup).splitlines()
        if line.strip()
    )


def format_market_value(value) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric_value):
        return "Unknown"

    return f"€{numeric_value:.2f}m"


def format_difference(value) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric_value):
        return "Unknown"

    if numeric_value > 0:
        return f"+€{numeric_value:.2f}m"

    if numeric_value < 0:
        return f"-€{abs(numeric_value):.2f}m"

    return "€0.00m"


def get_display_market_value(player: pd.Series) -> str:
    stored_value = safe_text(player.get("market_value"))

    if stored_value != "Unknown":
        return stored_value

    return format_market_value(player.get("market_value_numeric"))


def get_match_label(similarity_percent: float) -> Tuple[str, str]:
    if similarity_percent >= 95:
        return "Elite Match", "match-elite"
    if similarity_percent >= 90:
        return "Excellent Match", "match-excellent"
    if similarity_percent >= 85:
        return "Strong Match", "match-strong"
    if similarity_percent >= 75:
        return "Good Match", "match-good"
    return "Possible Match", "match-possible"


def short_role_name(role) -> str:
    role_text = safe_text(role)

    replacements = {
        "Defensive Core Players": "Defensive Core",
        "Creative Playmakers": "Creative Midfielder",
        "Attacking Fullbacks": "Attacking Fullback",
        "Ball-winning Midfielders": "Ball Winner",
        "Ball Winning Midfielders": "Ball Winner",
    }

    if role_text in replacements:
        return replacements[role_text]

    cleaned = role_text.replace(" Players", "").replace(" Player", "")
    return cleaned if cleaned else "Unknown"


def first_available_value(player: pd.Series, columns, fallback="Unknown") -> str:
    for column in columns:
        if column in player.index:
            value = safe_text(player.get(column), fallback="")
            if value:
                return value

    return fallback


def feature_category(feature: str) -> str:
    feature_name = feature.lower()

    if any(term in feature_name for term in ["goal", "shot", "xg"]):
        return "Comparable attacking output"
    if any(term in feature_name for term in ["assist", "chance", "xa", "creative"]):
        return "Similar creative contribution"
    if any(term in feature_name for term in ["tackle", "interception", "defensive"]):
        return "Similar defensive workload"
    if any(term in feature_name for term in ["recover", "press", "duel"]):
        return "Comparable ball-winning activity"
    if any(term in feature_name for term in ["carry", "progress", "dribble"]):
        return "Similar ball-progression profile"
    if any(term in feature_name for term in ["pass", "touch", "possession"]):
        return "Comparable possession involvement"
    if any(term in feature_name for term in ["clean_sheet", "clean sheets"]):
        return "Similar defensive reliability"
    if any(term in feature_name for term in ["start", "minute", "appearance"]):
        return "Comparable playing availability"

    return f"Similar {readable_feature_name(feature).lower()}"


def explanation_labels(features, top_n: int = 3):
    labels = []

    for feature in features:
        label = feature_category(feature)
        if label not in labels:
            labels.append(label)

        if len(labels) == top_n:
            break

    return labels


def get_recommendations(
    player_name: str,
    top_n: int = 5,
    same_position: bool = True,
    allowed_positions=None,
    min_market_value: Optional[float] = None,
    max_market_value: Optional[float] = None,
    excluded_clubs=None,
    exclude_current_club: bool = False,
    min_similarity: float = 0.0,
    ranking_mode: str = "Style Similarity",
    similarity_weight: float = 0.75,
    similarity_method: str = "Cosine",
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    matches = df[
        df["player_name"]
        .astype(str)
        .str.lower()
        .eq(player_name.lower())
    ]

    if matches.empty:
        return None, pd.DataFrame()

    player_index = matches.index[0]
    selected_player = df.loc[player_index]

    candidate_mask = (
        (df["cluster"] == selected_player["cluster"])
        & (df.index != player_index)
    )

    if same_position:
        candidate_mask &= (
            df["position_name"] == selected_player["position_name"]
        )
    elif allowed_positions:
        candidate_mask &= df["position_name"].isin(allowed_positions)

    if min_market_value is not None or max_market_value is not None:
        numeric_market_values = pd.to_numeric(
            df["market_value_numeric"], errors="coerce"
        )
        candidate_mask &= numeric_market_values.notna()

        if min_market_value is not None:
            candidate_mask &= numeric_market_values >= float(min_market_value)

        if max_market_value is not None:
            candidate_mask &= numeric_market_values <= float(max_market_value)

    excluded_clubs = [
        str(club).strip() for club in (excluded_clubs or []) if str(club).strip()
    ]

    if exclude_current_club:
        current_club = safe_text(selected_player.get("club_name"), fallback="")
        if current_club:
            excluded_clubs.append(current_club)

    if excluded_clubs:
        candidate_mask &= ~df["club_name"].astype(str).isin(set(excluded_clubs))

    candidates = df[candidate_mask].copy()

    if candidates.empty:
        return selected_player, pd.DataFrame()

    candidate_indices = candidates.index.to_numpy()

    # Phase 5C: calculate similarity with the selected metric in the exact
    # standardized feature space used by the saved model. Cosine is the
    # original baseline. Distance metrics are converted to bounded 0-100
    # similarity indices using a data-adaptive exponential transform.
    method = safe_text(similarity_method, fallback="Cosine")
    if method == "Euclidean":
        distances = euclidean_matrix[player_index, candidate_indices]
        positive = distances[distances > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        scale = max(scale, 1e-9)
        scores = np.exp(-distances / scale)
    elif method == "Manhattan":
        distances = manhattan_matrix[player_index, candidate_indices]
        positive = distances[distances > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        scale = max(scale, 1e-9)
        scores = np.exp(-distances / scale)
    else:
        scores = similarity_matrix[player_index, candidate_indices]

    candidates["similarity_score"] = scores
    candidates["similarity_percent"] = (
        np.clip(candidates["similarity_score"], 0.0, 1.0) * 100
    ).round(1)

    if min_similarity > 0:
        candidates = candidates[
            candidates["similarity_percent"] >= float(min_similarity)
        ].copy()

    if candidates.empty:
        return selected_player, pd.DataFrame()

    selected_market_value = pd.to_numeric(
        selected_player["market_value_numeric"],
        errors="coerce",
    )

    candidates["value_difference"] = (
        pd.to_numeric(
            candidates["market_value_numeric"],
            errors="coerce",
        )
        - selected_market_value
    )

    candidates["value_difference_display"] = (
        candidates["value_difference"].apply(format_difference)
    )

    # ------------------------------------------------------------------
    # Phase 5A: transparent recruitment ranking
    # ------------------------------------------------------------------
    # Style Similarity keeps the original research baseline unchanged.
    # Balanced Recruitment Score combines cosine similarity with a
    # pool-relative affordability score. Lower market value = higher value
    # score among the candidates that survived the recruitment filters.
    candidates["value_score"] = np.nan
    candidates["recruitment_score"] = candidates["similarity_percent"].astype(float)

    ranking_mode = safe_text(ranking_mode, fallback="Style Similarity")

    if ranking_mode == "Balanced Recruitment Score":
        numeric_values = pd.to_numeric(
            candidates["market_value_numeric"], errors="coerce"
        )

        # A value-aware score cannot be calculated responsibly when the
        # candidate's market value is missing, so those rows are excluded only
        # in this optional ranking mode. They remain available in the original
        # Style Similarity mode.
        candidates = candidates[numeric_values.notna()].copy()

        if candidates.empty:
            return selected_player, pd.DataFrame()

        numeric_values = pd.to_numeric(
            candidates["market_value_numeric"], errors="coerce"
        )

        # Percentile rank within the current filtered candidate pool.
        # Cheapest candidate receives the highest affordability score.
        candidates["value_score"] = (
            numeric_values.rank(method="average", pct=True, ascending=False) * 100
        )

        similarity_weight = float(np.clip(similarity_weight, 0.0, 1.0))
        value_weight = 1.0 - similarity_weight

        candidates["recruitment_score"] = (
            similarity_weight * candidates["similarity_percent"].astype(float)
            + value_weight * candidates["value_score"].astype(float)
        ).round(1)

        candidates = candidates.sort_values(
            ["recruitment_score", "similarity_percent"],
            ascending=[False, False],
        )
    else:
        candidates = candidates.sort_values(
            "similarity_score",
            ascending=False,
        )

    return selected_player, candidates.head(top_n).copy()


def get_top_matching_features(
    selected_player: pd.Series,
    comparison_player: pd.Series,
    features,
    top_n: int = 4,
):
    selected_vector = pd.to_numeric(
        selected_player[features],
        errors="coerce",
    ).fillna(0)

    comparison_vector = pd.to_numeric(
        comparison_player[features],
        errors="coerce",
    ).fillna(0)

    differences = (selected_vector - comparison_vector).abs()

    return differences.sort_values().head(top_n).index.tolist()



def get_feature_match_details(
    selected_player: pd.Series,
    comparison_player: pd.Series,
    features,
    top_n: int = 3,
    difference_n: int = 1,
):
    """Return interpretable feature-level matches in the model's scaled space.

    The overall recommendation score remains cosine similarity. This helper is
    only for explanation. It removes low-information zero-vs-zero matches and
    de-emphasizes composite/FPL-style metrics so the displayed reasons are
    driven by interpretable football statistics.
    """
    valid_features = [
        feature for feature in features if feature in available_features
    ]

    if not valid_features:
        return [], []

    # These can still remain in the recommendation model, but they are less
    # useful as human-facing explanations than direct on-pitch statistics.
    explanation_exclusions = {
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "ict index",
        "total_points",
        "total points",
        "bonus",
        "bps",
    }

    selected_position = df.index.get_loc(selected_player.name)
    comparison_position = df.index.get_loc(comparison_player.name)
    feature_positions = {
        feature: position
        for position, feature in enumerate(available_features)
    }

    details = []

    for feature in valid_features:
        feature_key = feature.lower().strip()

        if feature_key in explanation_exclusions:
            continue

        # Use raw values only to decide whether a comparison is informative.
        selected_raw = pd.to_numeric(
            selected_player.get(feature), errors="coerce"
        )
        comparison_raw = pd.to_numeric(
            comparison_player.get(feature), errors="coerce"
        )

        if pd.isna(selected_raw):
            selected_raw = 0.0
        if pd.isna(comparison_raw):
            comparison_raw = 0.0

        selected_raw = float(selected_raw)
        comparison_raw = float(comparison_raw)

        # A shared zero is mathematically identical but usually not a useful
        # scouting reason (e.g. both defenders having 0 goals/90).
        if np.isclose(selected_raw, 0.0, atol=1e-10) and np.isclose(
            comparison_raw, 0.0, atol=1e-10
        ):
            continue

        feature_position = feature_positions[feature]

        selected_scaled = float(
            X_scaled[selected_position, feature_position]
        )
        comparison_scaled = float(
            X_scaled[comparison_position, feature_position]
        )

        feature_spread = float(
            np.nanstd(X_scaled[:, feature_position])
        )

        # Constant or nearly constant features cannot meaningfully distinguish
        # players, so do not use them in the explanation.
        if not np.isfinite(feature_spread) or feature_spread < 1e-12:
            continue

        normalized_gap = (
            abs(selected_scaled - comparison_scaled) / feature_spread
        )

        match_percent = float(
            np.clip(
                100.0 * np.exp(-0.5 * normalized_gap ** 2),
                0.0,
                100.0,
            )
        )

        # Prefer features that are both close and actually active for at least
        # one of the players. The activity term only breaks ties among very
        # similar features; it does not change the displayed match percentage.
        population_raw = pd.to_numeric(
            df[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        population_scale = float(np.nanpercentile(np.abs(population_raw), 75))
        if not np.isfinite(population_scale) or population_scale < 1e-12:
            population_scale = 1.0

        activity = min(
            1.0,
            max(abs(selected_raw), abs(comparison_raw)) / population_scale,
        )

        explanation_score = normalized_gap - (0.08 * activity)

        details.append(
            {
                "feature": feature,
                "match_percent": round(match_percent, 1),
                "normalized_gap": normalized_gap,
                "explanation_score": explanation_score,
            }
        )

    # Safety fallback: if every available feature was filtered out, use any
    # non-constant model feature so the card never renders an empty explanation.
    if not details:
        for feature in valid_features:
            feature_position = feature_positions[feature]
            feature_spread = float(np.nanstd(X_scaled[:, feature_position]))

            if not np.isfinite(feature_spread) or feature_spread < 1e-12:
                continue

            selected_scaled = float(X_scaled[selected_position, feature_position])
            comparison_scaled = float(X_scaled[comparison_position, feature_position])
            normalized_gap = abs(selected_scaled - comparison_scaled) / feature_spread
            match_percent = float(
                np.clip(100.0 * np.exp(-0.5 * normalized_gap ** 2), 0.0, 100.0)
            )

            details.append(
                {
                    "feature": feature,
                    "match_percent": round(match_percent, 1),
                    "normalized_gap": normalized_gap,
                    "explanation_score": normalized_gap,
                }
            )

    strongest_matches = sorted(
        details,
        key=lambda item: (
            item["explanation_score"],
            -item["match_percent"],
        ),
    )[:top_n]

    strongest_feature_names = {
        item["feature"] for item in strongest_matches
    }

    difference_pool = [
        item for item in details
        if item["feature"] not in strongest_feature_names
    ]

    if not difference_pool:
        difference_pool = details

    biggest_differences = sorted(
        difference_pool,
        key=lambda item: item["normalized_gap"],
        reverse=True,
    )[:difference_n]

    return strongest_matches, biggest_differences

def get_feature_category_name(feature: str):
    """Map a model feature to an interpretable football dimension.

    This mapping affects only the explanation layer. It does not alter K-Means,
    cosine similarity, candidate filtering, or recommendation ranking.
    """
    name = feature.lower().replace("-", "_").replace(" ", "_")

    # Order matters: creative passing/crossing features should be classified as
    # Creation before the broader Possession/Progression rules catch "pass".
    if any(term in name for term in [
        "assist", "expected_assist", "xa", "chance", "key_pass",
        "cross", "creative", "shot_creating", "goal_creating"
    ]):
        return "Creation"

    if any(term in name for term in [
        "goal", "expected_goal", "xg", "shot", "finishing",
        "penalty_area", "box_touch"
    ]):
        return "Attacking"

    if any(term in name for term in [
        "recover", "duel", "press", "pressure", "possession_won",
        "ball_won", "ball_winning"
    ]):
        return "Ball Winning"

    if any(term in name for term in [
        "tackle", "interception", "clearance", "block", "clean_sheet",
        "defensive", "aerial", "error", "conceded"
    ]):
        return "Defending"

    if any(term in name for term in [
        "progress", "carry", "dribble", "take_on", "pass", "touch",
        "possession", "receive", "retention", "turnover"
    ]):
        return "Progression / Possession"

    return None


def get_category_similarity_scores(
    selected_player: pd.Series,
    comparison_player: pd.Series,
    features,
):
    """Calculate category-level similarity in the model's scaled feature space.

    For each football category, standardized feature gaps are aggregated with a
    root-mean-square distance and converted to a 0-100 similarity score using
    the same Gaussian-style transformation as the feature explainer.

    Shared zero features and constant features are removed so a category cannot
    look artificially perfect simply because both players recorded no activity.
    """
    category_order = [
        "Defending",
        "Ball Winning",
        "Progression / Possession",
        "Creation",
        "Attacking",
    ]

    feature_positions = {
        feature: position
        for position, feature in enumerate(available_features)
    }

    selected_position = df.index.get_loc(selected_player.name)
    comparison_position = df.index.get_loc(comparison_player.name)

    category_gaps = {category: [] for category in category_order}

    for feature in features:
        if feature not in feature_positions or feature not in df.columns:
            continue

        category = get_feature_category_name(feature)
        if category is None:
            continue

        selected_raw = pd.to_numeric(selected_player.get(feature), errors="coerce")
        comparison_raw = pd.to_numeric(comparison_player.get(feature), errors="coerce")

        if pd.isna(selected_raw):
            selected_raw = 0.0
        if pd.isna(comparison_raw):
            comparison_raw = 0.0

        # Prevent inactive 0-vs-0 statistics from inflating a category score.
        if np.isclose(float(selected_raw), 0.0, atol=1e-10) and np.isclose(
            float(comparison_raw), 0.0, atol=1e-10
        ):
            continue

        feature_position = feature_positions[feature]
        spread = float(np.nanstd(X_scaled[:, feature_position]))

        if not np.isfinite(spread) or spread < 1e-12:
            continue

        selected_scaled = float(X_scaled[selected_position, feature_position])
        comparison_scaled = float(X_scaled[comparison_position, feature_position])
        normalized_gap = abs(selected_scaled - comparison_scaled) / spread

        if np.isfinite(normalized_gap):
            category_gaps[category].append(float(normalized_gap))

    results = []

    for category in category_order:
        gaps = category_gaps[category]
        if not gaps:
            continue

        # RMS penalizes one large mismatch more than a simple mean while still
        # summarizing all available model features within the football category.
        rms_gap = float(np.sqrt(np.mean(np.square(gaps))))
        score = float(np.clip(100.0 * np.exp(-0.5 * rms_gap ** 2), 0.0, 100.0))

        results.append(
            {
                "category": category,
                "score": round(score, 1),
                "feature_count": len(gaps),
            }
        )

    return results


def category_icon(category: str) -> str:
    icons = {
        "Defending": "🛡️",
        "Ball Winning": "⚔️",
        "Progression / Possession": "⚡",
        "Creation": "🎨",
        "Attacking": "⚽",
    }
    return icons.get(category, "📊")


def readable_feature_name(feature: str) -> str:
    return (
        feature.replace("_per_90", " per 90")
        .replace("_", " ")
        .title()
    )


def is_lower_better_feature(feature: str) -> bool:
    """Return True for metrics where a lower raw value is generally preferable.

    This is used only for the percentile-style comparison radar. It does not
    alter the recommendation engine, K-Means clustering, or cosine similarity.
    """
    name = feature.lower().replace("-", "_").replace(" ", "_")
    return any(term in name for term in [
        "error", "conceded", "turnover", "dispossess", "miscontrol",
        "yellow", "red_card", "foul_committed", "own_goal",
    ])


def get_player_category_profile(player: pd.Series, features):
    """Build a 0-100 percentile profile across ScoutAI football dimensions.

    Each model feature is converted to a percentile rank relative to the full
    dataset, then percentiles are averaged within the football category. This
    makes metrics with different raw scales comparable on one radar chart.
    """
    category_order = [
        "Defending",
        "Ball Winning",
        "Progression / Possession",
        "Creation",
        "Attacking",
    ]

    category_values = {category: [] for category in category_order}

    for feature in features:
        if feature not in df.columns:
            continue

        category = get_feature_category_name(feature)
        if category is None:
            continue

        population = (
            pd.to_numeric(df[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

        valid_population = population.dropna()
        if valid_population.empty or valid_population.nunique() <= 1:
            continue

        player_value = pd.to_numeric(player.get(feature), errors="coerce")
        if pd.isna(player_value):
            continue

        # Percentile is calculated against the dataset distribution. Ties use
        # the average rank, which prevents identical values from being treated
        # as artificially unique.
        percentile_series = population.rank(method="average", pct=True) * 100.0
        percentile = percentile_series.loc[player.name]

        if pd.isna(percentile):
            continue

        percentile = float(percentile)
        if is_lower_better_feature(feature):
            percentile = 100.0 - percentile

        category_values[category].append(percentile)

    results = []
    for category in category_order:
        values = category_values[category]
        if not values:
            continue
        results.append(
            {
                "category": category,
                "score": round(float(np.mean(values)), 1),
                "feature_count": len(values),
            }
        )

    return results


def create_category_radar_chart(
    selected_player: pd.Series,
    comparison_player: pd.Series,
    features,
):
    """Create a normalized player-vs-player radar chart by football dimension."""
    selected_profile = get_player_category_profile(selected_player, features)
    comparison_profile = get_player_category_profile(comparison_player, features)

    selected_lookup = {
        item["category"]: item["score"] for item in selected_profile
    }
    comparison_lookup = {
        item["category"]: item["score"] for item in comparison_profile
    }

    category_order = [
        "Defending",
        "Ball Winning",
        "Progression / Possession",
        "Creation",
        "Attacking",
    ]

    categories = [
        category
        for category in category_order
        if category in selected_lookup and category in comparison_lookup
    ]

    if len(categories) < 3:
        return None, pd.DataFrame()

    selected_values = [selected_lookup[category] for category in categories]
    comparison_values = [comparison_lookup[category] for category in categories]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    selected_plot = selected_values + selected_values[:1]
    comparison_plot = comparison_values + comparison_values[:1]

    fig, ax = plt.subplots(figsize=(8.5, 7.2), subplot_kw={"polar": True})

    ax.plot(
        angles,
        selected_plot,
        linewidth=2.4,
        marker="o",
        label=safe_text(selected_player.get("player_name")),
    )
    ax.fill(angles, selected_plot, alpha=0.12)

    ax.plot(
        angles,
        comparison_plot,
        linewidth=2.4,
        marker="o",
        label=safe_text(comparison_player.get("player_name")),
    )
    ax.fill(angles, comparison_plot, alpha=0.12)

    display_labels = [
        "Progression /\nPossession" if category == "Progression / Possession"
        else category
        for category in categories
    ]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_title(
        "Player Profile Comparison",
        pad=24,
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.24, 1.14), frameon=True)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    profile_table = pd.DataFrame(
        {
            "Dimension": categories,
            safe_text(selected_player.get("player_name")): selected_values,
            safe_text(comparison_player.get("player_name")): comparison_values,
        }
    )

    return fig, profile_table


def build_detailed_comparison_table(
    selected_player: pd.Series,
    comparison_player: pd.Series,
    features,
):
    """Return raw statistical evidence behind the visual comparison."""
    rows = []

    for feature in features:
        selected_value = pd.to_numeric(selected_player.get(feature), errors="coerce")
        comparison_value = pd.to_numeric(comparison_player.get(feature), errors="coerce")

        if pd.isna(selected_value) and pd.isna(comparison_value):
            continue

        selected_value = np.nan if pd.isna(selected_value) else float(selected_value)
        comparison_value = np.nan if pd.isna(comparison_value) else float(comparison_value)

        difference = (
            np.nan
            if np.isnan(selected_value) or np.isnan(comparison_value)
            else comparison_value - selected_value
        )

        rows.append(
            {
                "Statistic": readable_feature_name(feature),
                safe_text(selected_player.get("player_name")): selected_value,
                safe_text(comparison_player.get("player_name")): comparison_value,
                "Difference": difference,
            }
        )

    comparison_table = pd.DataFrame(rows)

    if not comparison_table.empty:
        numeric_columns = comparison_table.columns[1:]
        comparison_table[numeric_columns] = comparison_table[numeric_columns].round(2)

    return comparison_table


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    compact_html(
        """
        <div class="scoutai-title">⚽ ScoutAI Recruit</div>
        <div class="scoutai-subtitle">
            Role-aware football recruitment powered by machine learning.
            Identify statistically similar replacement options based on playing
            style, position, performance profile, and financial constraints.
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ScoutAI Search")
    st.caption("Configure the recruitment brief.")

    player_options = sorted(
        df["player_name"].dropna().astype(str).unique()
    )

    selected_name = st.selectbox(
        "Player to replace",
        player_options,
    )

    selected_sidebar_matches = df[
        df["player_name"].astype(str).str.lower().eq(selected_name.lower())
    ]

    if not selected_sidebar_matches.empty:
        selected_sidebar_player = selected_sidebar_matches.iloc[0]
        selected_sidebar_role = short_role_name(selected_sidebar_player.get("role"))
        st.caption(f"Statistical role pool: **{selected_sidebar_role}**")

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=15,
        value=5,
    )

    st.markdown("#### Playing profile")

    same_position = st.checkbox(
        "Require same listed position",
        value=True,
    )

    allowed_positions = None

    if not same_position:
        role_position_options = []

        if not selected_sidebar_matches.empty:
            selected_cluster = selected_sidebar_matches.iloc[0]["cluster"]
            role_position_options = sorted(
                df.loc[df["cluster"] == selected_cluster, "position_name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        if not role_position_options:
            role_position_options = sorted(
                df["position_name"].dropna().astype(str).unique().tolist()
            )

        allowed_positions = st.multiselect(
            "Allowed candidate positions",
            role_position_options,
            default=role_position_options,
            help=(
                "Used only when same-position matching is turned off. "
                "Candidates still remain inside the selected player's statistical role cluster."
            ),
        )

    min_similarity = st.slider(
        "Minimum style similarity (%)",
        min_value=0,
        max_value=100,
        value=75,
        step=1,
        help="Hide candidates whose cosine-similarity score falls below this threshold.",
    )

    st.markdown("#### Financial constraints")

    use_budget = st.checkbox(
        "Apply market value range",
        value=False,
    )

    min_market_value = None
    max_market_value = None

    if use_budget:
        market_values = pd.to_numeric(
            df["market_value_numeric"],
            errors="coerce",
        ).dropna()

        if market_values.empty:
            st.warning("No numeric market values are available.")
        else:
            maximum_value = max(
                1,
                int(np.ceil(market_values.max())),
            )

            default_upper = min(50, maximum_value)
            value_range = st.slider(
                "Market value range (€m)",
                min_value=0,
                max_value=maximum_value,
                value=(0, default_upper),
                step=1,
            )
            min_market_value, max_market_value = value_range

    st.markdown("#### Club constraints")

    exclude_current_club = st.checkbox(
        "Exclude player's current club",
        value=False,
    )

    club_options = sorted(
        df["club_name"].dropna().astype(str).unique().tolist()
    )

    excluded_clubs = st.multiselect(
        "Exclude specific clubs",
        club_options,
        default=[],
        help="Players from selected clubs will be removed from the candidate pool.",
    )

    st.markdown("#### Similarity method")
    similarity_method = st.selectbox(
        "Compare player profiles with",
        ["Cosine", "Euclidean", "Manhattan"],
        index=0,
        help=(
            "Cosine is the original ScoutAI baseline. Euclidean and Manhattan "
            "provide alternative comparisons on the same standardized features."
        ),
    )
    if similarity_method != "Cosine":
        st.caption(
            "Distance is converted to a 0–100 similarity index with an exponential "
            "transform scaled to the median candidate distance. Treat scores across "
            "different methods as method-specific indices, not probabilities."
        )

    st.markdown("#### Ranking strategy")

    ranking_mode = st.selectbox(
        "Rank recommendations by",
        ["Style Similarity", "Balanced Recruitment Score"],
        index=0,
        help=(
            "Style Similarity preserves the original cosine-similarity ranking. "
            "Balanced Recruitment Score combines style similarity with market-value affordability."
        ),
    )

    similarity_weight = 0.75

    if ranking_mode == "Balanced Recruitment Score":
        similarity_weight_percent = st.slider(
            "Similarity weight (%)",
            min_value=50,
            max_value=100,
            value=75,
            step=5,
            help=(
                "The remaining weight is assigned to affordability. For example, "
                "75% similarity means 25% affordability."
            ),
        )
        similarity_weight = similarity_weight_percent / 100.0
        st.caption(
            f"Recruitment Score = {similarity_weight_percent}% style similarity + "
            f"{100 - similarity_weight_percent}% affordability."
        )
        st.caption(
            "Players without a numeric market value are excluded only from this value-aware ranking mode."
        )

    st.divider()
    st.caption(
        "Role-aware search remains locked to the selected player's K-Means "
        "statistical role cluster. Position, similarity, market value, and club "
        "filters narrow the candidate pool; they do not change the cosine-similarity calculation. "
        "The optional Balanced Recruitment Score changes ranking only, not player similarity."
    )


# -----------------------------------------------------------------------------
# Recommendations
# -----------------------------------------------------------------------------
selected_player, recommendations = get_recommendations(
    selected_name,
    top_n=top_n,
    same_position=same_position,
    allowed_positions=allowed_positions,
    min_market_value=min_market_value,
    max_market_value=max_market_value,
    excluded_clubs=excluded_clubs,
    exclude_current_club=exclude_current_club,
    min_similarity=min_similarity,
    ranking_mode=ranking_mode,
    similarity_weight=similarity_weight,
    similarity_method=similarity_method,
)

if selected_player is None:
    st.error("Selected player was not found.")
    st.stop()


# -----------------------------------------------------------------------------
# Phase 5B/5C/5D: Research diagnostics
# -----------------------------------------------------------------------------
def _research_candidate_pool(player_row, same_position=True, allowed_positions=None):
    idx = player_row.name
    mask = (df["cluster"] == player_row["cluster"]) & (df.index != idx)
    if same_position:
        mask &= df["position_name"] == player_row["position_name"]
    elif allowed_positions:
        mask &= df["position_name"].isin(allowed_positions)
    return df[mask].copy()

def _metric_scores(player_index, candidate_indices, method):
    if method == "Cosine":
        return np.clip(similarity_matrix[player_index, candidate_indices], 0, 1) * 100
    matrix = euclidean_matrix if method == "Euclidean" else manhattan_matrix
    d = matrix[player_index, candidate_indices]
    positive = d[d > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return np.exp(-d / max(scale, 1e-9)) * 100

def build_research_diagnostics(player_row, k=5):
    pool = _research_candidate_pool(player_row, same_position=same_position, allowed_positions=allowed_positions)
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    idx = player_row.name
    ids = pool.index.to_numpy()
    metric_rows = []
    top_sets = {}
    for method in ["Cosine", "Euclidean", "Manhattan"]:
        vals = _metric_scores(idx, ids, method)
        temp = pd.DataFrame({"player": pool["player_name"].values, "score": vals}).sort_values("score", ascending=False)
        top = temp.head(k).copy()
        top_sets[method] = set(top["player"])
        for rank, row in enumerate(top.itertuples(index=False), 1):
            metric_rows.append({"Method": method, "Rank": rank, "Player": row.player, "Similarity Index": round(float(row.score), 1)})
    method_table = pd.DataFrame(metric_rows)
    overlap_rows=[]
    baseline=top_sets.get("Cosine", set())
    for method in ["Euclidean", "Manhattan"]:
        overlap=len(baseline & top_sets.get(method,set()))
        overlap_rows.append({"Comparison": f"Cosine vs {method}", f"Top-{k} overlap": f"{overlap}/{k}", "Agreement": round(100*overlap/max(k,1),1)})
    overlap_table=pd.DataFrame(overlap_rows)

    # 5B sensitivity: how the value-aware top-k changes as weights move.
    numeric = pd.to_numeric(pool["market_value_numeric"], errors="coerce")
    value_pool = pool[numeric.notna()].copy()
    weight_rows=[]
    if not value_pool.empty:
        vids=value_pool.index.to_numpy()
        sim=_metric_scores(idx, vids, similarity_method)
        vals=pd.to_numeric(value_pool["market_value_numeric"], errors="coerce")
        affordability=vals.rank(method="average", pct=True, ascending=False).to_numpy()*100
        for sw in [1.00, .90, .75, .60, .50]:
            total=sw*sim+(1-sw)*affordability
            order=np.argsort(-total)[:k]
            names=value_pool.iloc[order]["player_name"].astype(str).tolist()
            weight_rows.append({"Similarity Weight": f"{int(sw*100)}%", "Value Weight": f"{int((1-sw)*100)}%", "Top Recommendations": ", ".join(names)})
    return method_table, overlap_table, pd.DataFrame(weight_rows)

research_methods, research_overlap, research_weights = build_research_diagnostics(selected_player, k=min(top_n, 5))


player_name = safe_html(selected_player.get("player_name"))
club_name = safe_html(selected_player.get("club_name"))
position_name = safe_html(selected_player.get("position_name"))
player_role = safe_html(short_role_name(selected_player.get("role")))
market_value = safe_html(get_display_market_value(selected_player))

age_value = first_available_value(
    selected_player,
    ["age", "player_age"],
    fallback="",
)
foot_value = first_available_value(
    selected_player,
    ["preferred_foot", "foot"],
    fallback="",
)
nationality_value = first_available_value(
    selected_player,
    ["nationality", "country", "nation"],
    fallback="",
)

optional_profile_cards = ""

if age_value:
    optional_profile_cards += (
        '<div class="profile-stat">'
        '<div class="profile-label">🎂 Age</div>'
        f'<div class="profile-value">{safe_html(age_value)}</div>'
        '</div>'
    )

if foot_value:
    optional_profile_cards += (
        '<div class="profile-stat">'
        '<div class="profile-label">🦶 Preferred Foot</div>'
        f'<div class="profile-value">{safe_html(foot_value)}</div>'
        '</div>'
    )

if nationality_value:
    optional_profile_cards += (
        '<div class="profile-stat">'
        '<div class="profile-label">🌍 Nationality</div>'
        f'<div class="profile-value">{safe_html(nationality_value)}</div>'
        '</div>'
    )
player_photo_url = first_available_value(
    selected_player,
    ["player_image_url", "photo_url", "image_url"],
    fallback="",
)
club_logo_url = first_available_value(
    selected_player,
    ["club_logo_url", "team_logo_url", "logo_url"],
    fallback="",
)

player_photo_html = (
    f'<img class="player-photo" src="{safe_html(player_photo_url)}" '
    f'alt="{player_name}">'
    if player_photo_url
    else ""
)

club_logo_html = (
    f'<img class="club-logo" src="{safe_html(club_logo_url)}" alt="">'
    if club_logo_url
    else ""
)


st.markdown(
    '<div class="eyebrow">Selected Player Profile</div>',
    unsafe_allow_html=True,
)

st.markdown(
    compact_html(
        f"""
        <div class="player-card">
            <div class="player-header-row">
                {player_photo_html}
                <div>
                    <div class="player-name">{player_name}</div>
                    <div class="player-club">{club_logo_html}{club_name}</div>
                    <div class="role-badge">{player_role}</div>
                </div>
            </div>

            <div class="profile-grid">
                <div class="profile-stat">
                    <div class="profile-label">📍 Position</div>
                    <div class="profile-value">{position_name}</div>
                </div>

                <div class="profile-stat">
                    <div class="profile-label">🧠 Role</div>
                    <div class="profile-value">{player_role}</div>
                </div>

                <div class="profile-stat">
                    <div class="profile-label">💰 Market Value</div>
                    <div class="profile-value">{market_value}</div>
                </div>

                {optional_profile_cards}
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="eyebrow">Recommended Replacements</div>',
    unsafe_allow_html=True,
)

if recommendations.empty:
    st.warning(
        "No players matched the current recruitment brief. Try lowering the "
        "minimum similarity, widening the market value range, allowing more "
        "positions, removing club exclusions, or switching back to Style Similarity "
        "if too many candidates have missing market values."
    )
else:
    card_columns = st.columns(min(len(recommendations), 3))

    for card_index, (_, player) in enumerate(recommendations.iterrows()):
        column = card_columns[card_index % len(card_columns)]

        similarity = float(player["similarity_percent"])
        recruitment_score = float(player.get("recruitment_score", similarity))
        value_score_numeric = pd.to_numeric(player.get("value_score"), errors="coerce")
        display_value = get_display_market_value(player)
        value_difference = safe_text(
            player.get("value_difference_display")
        )
        match_label, match_class = get_match_label(similarity)

        category_scores = get_category_similarity_scores(
            selected_player,
            player,
            available_features,
        )

        category_html = "".join(
            '<div class="category-item">'
            f'<span class="category-name">{category_icon(item["category"])} '
            f'{safe_html(item["category"])}</span>'
            f'<span class="category-score">{item["score"]:.1f}%</span>'
            '</div>'
            for item in category_scores
        )

        if not category_html:
            category_html = (
                '<div class="why-text">Not enough categorized model features '
                'to calculate category similarity.</div>'
            )

        strongest_matches, biggest_differences = get_feature_match_details(
            selected_player,
            player,
            available_features,
            top_n=min(3, len(available_features)),
            difference_n=1,
        )

        matching_parts = []

        for item in strongest_matches:
            matching_parts.append(
                '<span class="why-item">'
                f'• {safe_html(readable_feature_name(item["feature"]))} '
                f'— {item["match_percent"]:.1f}% feature match'
                '</span>'
            )

        if biggest_differences:
            difference = biggest_differences[0]
            matching_parts.append(
                '<span class="why-item">'
                f'⚠ Biggest difference: '
                f'{safe_html(readable_feature_name(difference["feature"]))} '
                f'({difference["match_percent"]:.1f}% feature match)'
                '</span>'
            )

        matching_text = "".join(matching_parts)

        recruitment_score_html = ""
        if ranking_mode == "Balanced Recruitment Score":
            value_score_display = (
                f"{float(value_score_numeric):.1f}%"
                if pd.notna(value_score_numeric)
                else "Unknown"
            )
            recruitment_score_html = (
                '<div class="mini-stat">'
                '<div class="mini-label">⭐ Recruitment Score</div>'
                f'<div class="mini-value">{recruitment_score:.1f}/100</div>'
                '</div>'
                '<div class="mini-stat">'
                '<div class="mini-label">💎 Affordability Score</div>'
                f'<div class="mini-value">{safe_html(value_score_display)}</div>'
                '</div>'
            )

        with column:
            st.markdown(
                compact_html(
                    f"""
                    <div class="recommendation-card">
                        <div class="recommendation-name">
                            {safe_html(player.get("player_name"))}
                        </div>

                        <div class="recommendation-club">
                            {safe_html(player.get("club_name"))}
                        </div>

                        <div class="match-badge {match_class}">{match_label}</div>

                        <div class="similarity-number">{similarity:.1f}%</div>
                        <div class="profile-label">Style Similarity</div>

                        <div class="mini-grid">
                            <div class="mini-stat">
                                <div class="mini-label">🧠 Role</div>
                                <div class="mini-value">
                                    {safe_html(short_role_name(player.get("role")))}
                                </div>
                            </div>

                            <div class="mini-stat">
                                <div class="mini-label">💰 Market Value</div>
                                <div class="mini-value">
                                    {safe_html(display_value)}
                                </div>
                            </div>

                            <div class="mini-stat">
                                <div class="mini-label">📍 Position</div>
                                <div class="mini-value">
                                    {safe_html(player.get("position_name"))}
                                </div>
                            </div>

                            <div class="mini-stat">
                                <div class="mini-label">📉 Value Difference</div>
                                <div class="mini-value">
                                    {safe_html(value_difference)}
                                </div>
                            </div>
                            {recruitment_score_html}
                        </div>

                        <div class="category-section">
                            <div class="mini-label">Similarity Breakdown</div>
                            <div class="category-grid">
                                {category_html}
                            </div>
                        </div>

                        <div class="why-section">
                            <div class="mini-label">Why ScoutAI Recommends Him</div>
                            <div class="why-text">
                                {matching_text}
                            </div>
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    st.markdown("### Recommendation Table")

    table_columns = [
        "player_name",
        "club_name",
        "position_name",
        "role",
        "similarity_percent",
        "market_value",
        "value_difference_display",
    ]

    if ranking_mode == "Balanced Recruitment Score":
        table_columns.extend(["value_score", "recruitment_score"])

    display_table = recommendations[table_columns].copy()

    display_table["market_value"] = display_table.apply(
        get_display_market_value,
        axis=1,
    )
    display_table["role"] = display_table["role"].apply(short_role_name)

    display_column_names = [
        "Player",
        "Club",
        "Position",
        "Role",
        "Similarity (%)",
        "Market Value",
        "Value Difference",
    ]

    if ranking_mode == "Balanced Recruitment Score":
        display_column_names.extend(["Affordability Score", "Recruitment Score"])
        display_table["value_score"] = pd.to_numeric(
            display_table["value_score"], errors="coerce"
        ).round(1)
        display_table["recruitment_score"] = pd.to_numeric(
            display_table["recruitment_score"], errors="coerce"
        ).round(1)

    display_table.columns = display_column_names

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Phase 3A + 3B: Player comparison visualizations
# -----------------------------------------------------------------------------
if not recommendations.empty:
    st.markdown("## Player Comparison")
    st.caption(
        "Compare the selected player with one recommended replacement. "
        "The radar uses percentile-style category profiles so statistics with "
        "different raw scales can be compared fairly."
    )

    comparison_name = st.selectbox(
        "Choose a recommended player to compare",
        recommendations["player_name"].tolist(),
    )

    comparison_player = recommendations[
        recommendations["player_name"] == comparison_name
    ].iloc[0]

    comparison_features = [
        feature
        for feature in [
            "goals_per_90",
            "assists_per_90",
            "expected_goals_per_90",
            "expected_assists_per_90",
            "expected_goal_involvements_per_90",
            "tackles_per_90",
            "recoveries_per_90",
            "defensive_contribution_per_90",
            "clearances_blocks_interceptions_per_90",
            "clean_sheets_per_90",
            "starts_per_90",
        ]
        if feature in df.columns
    ]

    # Use all categorized model features for the radar when possible. This keeps
    # the visual aligned with the same feature universe used by ScoutAI.
    radar_features = [
        feature
        for feature in available_features
        if get_feature_category_name(feature) is not None
    ]

    comparison_similarity = float(comparison_player["similarity_percent"])
    match_label, _ = get_match_label(comparison_similarity)

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Style Similarity", f"{comparison_similarity:.1f}%")
    summary_col2.metric("Match Tier", match_label)
    summary_col3.metric(
        "Role",
        short_role_name(comparison_player.get("role")),
    )

    st.markdown("### 3A · Player Profile Radar")

    radar_chart, profile_table = create_category_radar_chart(
        selected_player,
        comparison_player,
        radar_features,
    )

    if radar_chart is not None:
        radar_col, profile_col = st.columns([1.35, 1])

        with radar_col:
            st.pyplot(radar_chart, use_container_width=True)
            plt.close(radar_chart)

        with profile_col:
            st.markdown("#### Dimension Percentiles")
            st.caption(
                "Scores show each player's relative statistical profile within "
                "the dataset. They are descriptive percentiles, not the overall "
                "cosine-similarity score."
            )
            st.dataframe(
                profile_table,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "Not enough categorized model features were available to create "
            "a reliable radar chart."
        )

    st.markdown("### 3B · Detailed Statistical Comparison")
    st.caption(
        "Raw per-90 values remain visible here so the normalized radar never "
        "hides the underlying statistical evidence. Difference is calculated "
        "as recommended player minus selected player."
    )

    if comparison_features:
        comparison_table = build_detailed_comparison_table(
            selected_player,
            comparison_player,
            comparison_features,
        )

        st.dataframe(
            comparison_table,
            use_container_width=True,
            hide_index=True,
        )

        strongest_matches, biggest_differences = get_feature_match_details(
            selected_player,
            comparison_player,
            comparison_features,
            top_n=min(3, len(comparison_features)),
            difference_n=1,
        )

        if strongest_matches:
            st.markdown("#### Statistical Takeaways")
            takeaway_columns = st.columns(len(strongest_matches))
            for takeaway_column, item in zip(takeaway_columns, strongest_matches):
                takeaway_column.metric(
                    readable_feature_name(item["feature"]),
                    f'{item["match_percent"]:.1f}% match',
                )

        if biggest_differences:
            difference = biggest_differences[0]
            st.caption(
                "Largest meaningful statistical gap: "
                f'**{readable_feature_name(difference["feature"])}** '
                f'({difference["match_percent"]:.1f}% feature match).'
            )
    else:
        st.info("No detailed comparison statistics were found in the dataset.")


# -----------------------------------------------------------------------------
# Phase 3C: Player Style Map 2.0
# -----------------------------------------------------------------------------
st.markdown("## Player Style Map")
st.caption(
    "PCA projects the model features into two dimensions for visualization. "
    "It is not used to calculate recommendations. The selected player's role "
    "cluster is emphasized while unrelated players are faded into the background."
)

try:
    pca = PCA(n_components=2)
    pca_values = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pca_values,
        columns=["PC1", "PC2"],
        index=df.index,
    )

    pca_df["cluster"] = df["cluster"]
    pca_df["player_name"] = df["player_name"]
    pca_df["role"] = df["role"]

    selected_cluster = selected_player["cluster"]
    selected_role_label = short_role_name(selected_player.get("role"))

    same_role_mask = pca_df["cluster"] == selected_cluster
    other_role_mask = ~same_role_mask

    fig, ax = plt.subplots(figsize=(11.5, 7.2))

    # Context first: all players outside the selected role are intentionally
    # de-emphasized so they do not compete visually with recruitment candidates.
    if other_role_mask.any():
        ax.scatter(
            pca_df.loc[other_role_mask, "PC1"],
            pca_df.loc[other_role_mask, "PC2"],
            s=28,
            alpha=0.12,
            label="Other role clusters",
        )

    # Highlight the statistical role from which recommendations are generated.
    ax.scatter(
        pca_df.loc[same_role_mask, "PC1"],
        pca_df.loc[same_role_mask, "PC2"],
        s=38,
        alpha=0.38,
        label=f"{selected_role_label} role cluster",
    )

    selected_row = pca_df.loc[selected_player.name]

    ax.scatter(
        selected_row["PC1"],
        selected_row["PC2"],
        s=260,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label=safe_text(selected_player.get("player_name")),
    )

    ax.annotate(
        safe_text(selected_player.get("player_name")),
        (selected_row["PC1"], selected_row["PC2"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        zorder=6,
    )

    if not recommendations.empty:
        recommendation_rows = []

        for _, recommended_player in recommendations.head(5).iterrows():
            recommended_row = pca_df.loc[recommended_player.name]
            recommendation_rows.append(recommended_row)

        if recommendation_rows:
            recommendation_x = [row["PC1"] for row in recommendation_rows]
            recommendation_y = [row["PC2"] for row in recommendation_rows]

            ax.scatter(
                recommendation_x,
                recommendation_y,
                s=95,
                marker="o",
                edgecolors="black",
                linewidths=0.9,
                zorder=4,
                label="Recommended replacements",
            )

            label_offsets = [
                (8, 8),
                (8, -15),
                (-72, 8),
                (-72, -15),
                (10, 16),
            ]

            for label_index, (_, recommended_player) in enumerate(
                recommendations.head(5).iterrows()
            ):
                recommended_row = pca_df.loc[recommended_player.name]
                offset = label_offsets[label_index % len(label_offsets)]

                ax.annotate(
                    safe_text(recommended_player.get("player_name")),
                    (recommended_row["PC1"], recommended_row["PC2"]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8.5,
                    zorder=6,
                )

    explained_variance = pca.explained_variance_ratio_.sum() * 100

    ax.set_xlabel("PCA Style Dimension 1")
    ax.set_ylabel("PCA Style Dimension 2")
    ax.set_title(
        f"{selected_role_label} Recruitment Neighborhood",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.15)

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "The two displayed PCA dimensions explain "
        f"{explained_variance:.1f}% of the variation in the model features. "
        "Distances on this chart are visual context only; ScoutAI still ranks "
        "recommendations with cosine similarity in the full scaled feature space."
    )

except Exception as error:
    st.warning("The Player Style Map could not be created.")
    st.exception(error)


# -----------------------------------------------------------------------------
# Methodology
# -----------------------------------------------------------------------------
with st.expander("Methodology and Limitations"):
    st.write(
        """
        Players are grouped into statistical playing roles using K-Means
        clustering. Cosine similarity is then used to compare a selected
        player with candidates from the same cluster. The optional position
        filter restricts recommendations to the selected player's listed
        position.

        Market value is displayed as recruitment-cost context. It is not used
        to calculate similarity. The optional budget filter removes candidates
        whose estimated market value exceeds the selected limit.

        PCA is included only as a visualization tool to show how players and
        clusters are distributed across two reduced dimensions. PCA is not the
        recommendation engine.

        Recommendation quality depends on the available data. Additional
        event-level statistics such as progressive passing, pressing, ball
        carrying, chance creation, and detailed defensive actions could improve
        role separation and player comparisons.
        """
    )


# -----------------------------------------------------------------------------
# Phase 5 Research Lab
# -----------------------------------------------------------------------------
st.divider()
st.markdown("## Phase 5 · Recommendation Research Lab")
st.caption(
    "These diagnostics test sensitivity and method agreement. They do not prove that a "
    "recommendation is objectively correct; external football outcomes or expert labels "
    "would be required for ground-truth validation."
)

with st.expander("5B · Weight Sensitivity", expanded=False):
    st.write(
        "Shows whether value-aware recommendations remain stable when the similarity/affordability "
        "weights change. Large shortlist changes indicate a weight-sensitive ranking."
    )
    if research_weights.empty:
        st.info("Not enough numeric market-value data is available for this diagnostic.")
    else:
        st.dataframe(research_weights, use_container_width=True, hide_index=True)

with st.expander("5C · Similarity Method Comparison", expanded=False):
    st.write(
        "Compares Cosine, Euclidean, and Manhattan methods on the same standardized model features. "
        "The distance-based scores are method-specific similarity indices, so ranking agreement is "
        "more informative than comparing their raw percentages directly."
    )
    if research_methods.empty:
        st.info("No eligible candidates are available for method comparison.")
    else:
        st.dataframe(research_methods, use_container_width=True, hide_index=True)
        if not research_overlap.empty:
            st.markdown("**Top-k agreement with the cosine baseline**")
            st.dataframe(research_overlap, use_container_width=True, hide_index=True)

with st.expander("5D · Evaluation Framework", expanded=False):
    st.markdown("""
**Current internal evaluation**

- **Method agreement:** do different similarity measures identify the same candidates?
- **Weight sensitivity:** does the shortlist remain stable when recruitment weights change?
- **Constraint validity:** every result must satisfy the active role, position, value, club, and threshold filters.
- **Explainability consistency:** displayed strengths/gaps are calculated from the same standardized feature space.

**Still required for research-grade external validation**

Expert scout ratings, historical replacement/transfer outcomes, or another defensible ground-truth label. Without one of those, ScoutAI can evaluate robustness and consistency but should not claim that a recommendation is objectively correct.
""")