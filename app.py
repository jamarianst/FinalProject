import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Player Replacement Analytics",
    page_icon="⚽",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv("player_replacement_data.csv")


@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans.pkl")
    model_features = joblib.load("model_features.pkl")

    return scaler, kmeans, model_features


df = load_data()
scaler, kmeans, model_features = load_models()


required_columns = [
    "player_name",
    "club_name",
    "position_name",
    "cluster",
    "role",
    "market_value",
    "market_value_numeric"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


available_features = [
    feature
    for feature in model_features
    if feature in df.columns
]

if not available_features:
    st.error(
        "None of the saved model features were found in the dataset."
    )
    st.stop()


X = (
    df[available_features]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


try:
    X_scaled = scaler.transform(X)
except Exception as error:
    st.error(
        "The scaler could not transform the dataset. "
        "Make sure the CSV and saved model files were created "
        "from the same notebook."
    )
    st.exception(error)
    st.stop()


similarity_matrix = cosine_similarity(X_scaled)


def safe_text(value, fallback="Unknown"):
    if pd.isna(value):
        return fallback

    return str(value)


def format_market_value(value):
    if pd.isna(value):
        return "Unknown"

    return f"€{value:.2f}m"


def get_recommendations(
    player_name,
    top_n=5,
    same_position=True,
    max_market_value=None
):
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
            df["position_name"]
            == selected_player["position_name"]
        )

    if max_market_value is not None:
        candidate_mask &= (
            df["market_value_numeric"].notna()
            & (
                df["market_value_numeric"]
                <= max_market_value
            )
        )

    candidates = df[candidate_mask].copy()

    if candidates.empty:
        return selected_player, pd.DataFrame()

    candidate_indices = candidates.index.to_numpy()

    candidates["similarity_score"] = (
        similarity_matrix[
            player_index,
            candidate_indices
        ]
    )

    candidates["similarity_percent"] = (
        candidates["similarity_score"] * 100
    ).round(1)

    selected_market_value = selected_player[
        "market_value_numeric"
    ]

    candidates["value_difference"] = (
        candidates["market_value_numeric"]
        - selected_market_value
    )

    def format_difference(value):
        if pd.isna(value):
            return "Unknown"

        if value > 0:
            return f"+€{value:.2f}m"

        if value < 0:
            return f"-€{abs(value):.2f}m"

        return "€0.00m"

    candidates["value_difference_display"] = (
        candidates["value_difference"]
        .apply(format_difference)
    )

    recommendations = (
        candidates
        .sort_values(
            "similarity_score",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    return selected_player, recommendations


def create_comparison_chart(
    selected_player,
    comparison_player,
    features
):
    selected_values = [
        pd.to_numeric(
            selected_player[feature],
            errors="coerce"
        )
        for feature in features
    ]

    comparison_values = [
        pd.to_numeric(
            comparison_player[feature],
            errors="coerce"
        )
        for feature in features
    ]

    selected_values = [
        0 if pd.isna(value) else value
        for value in selected_values
    ]

    comparison_values = [
        0 if pd.isna(value) else value
        for value in comparison_values
    ]

    positions = np.arange(len(features))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        positions - bar_width / 2,
        selected_values,
        bar_width,
        label=selected_player["player_name"]
    )

    ax.bar(
        positions + bar_width / 2,
        comparison_values,
        bar_width,
        label=comparison_player["player_name"]
    )

    readable_labels = [
        feature
        .replace("_per_90", " per 90")
        .replace("_", " ")
        .title()
        for feature in features
    ]

    ax.set_xticks(positions)
    ax.set_xticklabels(
        readable_labels,
        rotation=45,
        ha="right"
    )

    ax.set_ylabel("Performance Value")
    ax.set_title("Player Statistical Comparison")
    ax.legend()

    fig.tight_layout()

    return fig


st.title("⚽ Player Replacement Analytics")

st.write(
    "Select a player to identify statistically similar "
    "replacement options based on playing role, position, "
    "performance profile, and market value."
)


with st.sidebar:
    st.header("Recruitment Filters")

    player_options = sorted(
        df["player_name"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_name = st.selectbox(
        "Select player",
        player_options
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=15,
        value=5
    )

    same_position = st.checkbox(
        "Require same position",
        value=True
    )

    use_budget = st.checkbox(
        "Apply budget limit",
        value=False
    )

    max_market_value = None

    if use_budget:
        market_values = (
            df["market_value_numeric"]
            .dropna()
        )

        if market_values.empty:
            st.warning(
                "No numeric market values are available."
            )
        else:
            maximum_value = max(
                1,
                int(np.ceil(market_values.max()))
            )

            default_value = min(
                50,
                maximum_value
            )

            max_market_value = st.slider(
                "Maximum market value (€m)",
                min_value=1,
                max_value=maximum_value,
                value=default_value
            )


selected_player, recommendations = get_recommendations(
    selected_name,
    top_n=top_n,
    same_position=same_position,
    max_market_value=max_market_value
)


if selected_player is None:
    st.error("Selected player was not found.")
    st.stop()


st.subheader("Selected Player")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Player",
    safe_text(selected_player["player_name"])
)

col2.metric(
    "Club",
    safe_text(selected_player["club_name"])
)

col3.metric(
    "Position",
    safe_text(selected_player["position_name"])
)

col4.metric(
    "Role",
    safe_text(selected_player["role"])
)

selected_market_value = selected_player[
    "market_value"
]

if pd.isna(selected_market_value):
    selected_market_value = format_market_value(
        selected_player[
            "market_value_numeric"
        ]
    )

col5.metric(
    "Market Value",
    safe_text(selected_market_value)
)


st.subheader("Recommended Replacements")

if recommendations.empty:
    st.warning(
        "No players matched the selected filters. "
        "Try removing the position or budget restriction."
    )

else:
    display_columns = [
        "player_name",
        "club_name",
        "position_name",
        "role",
        "similarity_percent",
        "market_value",
        "value_difference_display"
    ]

    display_table = recommendations[
        display_columns
    ].copy()

    display_table.columns = [
        "Player",
        "Club",
        "Position",
        "Role",
        "Similarity (%)",
        "Market Value",
        "Value Difference"
    ]

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Player Comparison")

    comparison_name = st.selectbox(
        "Choose a recommended player to compare",
        recommendations[
            "player_name"
        ].tolist()
    )

    comparison_player = recommendations[
        recommendations["player_name"]
        == comparison_name
    ].iloc[0]

    comparison_features = [
        feature
        for feature in [
            "goals_per_90",
            "assists_per_90",
            "expected_goals_per_90",
            "expected_assists_per_90",
            "tackles_per_90",
            "recoveries_per_90",
            "defensive_contribution_per_90",
            "clean_sheets_per_90",
            "starts_per_90"
        ]
        if feature in df.columns
    ]

    if comparison_features:
        comparison_table = pd.DataFrame(
            {
                "Statistic": [
                    feature
                    .replace("_per_90", " per 90")
                    .replace("_", " ")
                    .title()
                    for feature in comparison_features
                ],
                selected_player["player_name"]: [
                    selected_player[feature]
                    for feature in comparison_features
                ],
                comparison_player["player_name"]: [
                    comparison_player[feature]
                    for feature in comparison_features
                ]
            }
        )

        numeric_columns = comparison_table.columns[1:]

        comparison_table[numeric_columns] = (
            comparison_table[numeric_columns]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .round(2)
        )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
    )

        comparison_chart = create_comparison_chart(
            selected_player,
            comparison_player,
            comparison_features
        )

        st.pyplot(comparison_chart)

        plt.close(comparison_chart)

    else:
        st.info(
            "No comparison statistics were found "
            "in the dataset."
        )


st.subheader("PCA Cluster Visualization")

try:
    pca = PCA(n_components=2)

    pca_values = pca.fit_transform(
        X_scaled
    )

    pca_df = pd.DataFrame(
        pca_values,
        columns=["PC1", "PC2"],
        index=df.index
    )

    pca_df["cluster"] = df["cluster"]
    pca_df["player_name"] = df[
        "player_name"
    ]

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["cluster"],
        cmap="tab10",
        alpha=0.65
    )

    selected_row = pca_df.loc[
        selected_player.name
    ]

    ax.scatter(
        selected_row["PC1"],
        selected_row["PC2"],
        s=180,
        marker="*",
        edgecolors="black",
        linewidths=1.2
    )

    ax.annotate(
        selected_player["player_name"],
        (
            selected_row["PC1"],
            selected_row["PC2"]
        ),
        xytext=(8, 8),
        textcoords="offset points"
    )

    if not recommendations.empty:
        for _, recommended_player in (
            recommendations.head(5).iterrows()
        ):
            recommended_row = pca_df.loc[
                recommended_player.name
            ]

            ax.scatter(
                recommended_row["PC1"],
                recommended_row["PC2"],
                s=65,
                marker="o",
                edgecolors="black",
                linewidths=0.7
            )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.set_title(
        "Player Role Clusters"
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Cluster"
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    explained_variance = (
        pca.explained_variance_ratio_.sum()
        * 100
    )

    st.caption(
        "The two displayed principal components "
        f"explain {explained_variance:.1f}% of the "
        "variation in the model features."
    )

except Exception as error:
    st.warning(
        "The PCA visualization could not be created."
    )
    st.exception(error)


with st.expander(
    "Methodology and Limitations"
):
    st.write(
        """
        Players are grouped into statistical playing roles
        using K-Means clustering. Cosine similarity is then
        used to compare a selected player with candidates
        from the same cluster. The optional position filter
        restricts recommendations to the selected player's
        listed position.

        Market value is displayed as recruitment-cost context.
        It is not used to calculate similarity. The optional
        budget filter removes candidates whose estimated market
        value exceeds the selected limit.

        PCA is included as a visualization tool to show how
        players and clusters are distributed across two reduced
        dimensions. PCA is not the recommendation engine.

        The recommendation quality depends on the available
        data. Additional event-level statistics such as
        progressive passing, pressing, ball carrying, chance
        creation, and detailed defensive actions could improve
        role separation and player comparisons.
        """
    )
