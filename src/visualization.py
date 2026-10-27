"""
Visualization Module for Kerala Tourism Outlier Analysis
=========================================================
Generates analytical charts, time-series anomaly overlays,
multivariate scatter projections, and district comparison heatmaps.
"""

from typing import List, Optional
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def set_plot_style():
    """Configures modern styling for tourism analytics plots."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11


def plot_timeseries_outliers(
    df: pd.DataFrame,
    districts: Optional[List[str]] = None,
    outlier_col: str = "Ensemble_Outlier",
    save_path: Optional[str] = None,
):
    """
    Plots multi-district tourist arrival trajectories over time with overlaid anomaly flags.
    Highlights historic disruptions like August 2018 Floods and 2020 COVID lockdowns.
    """
    set_plot_style()
    if districts is None:
        # Highlight top diverse representative districts
        districts = ["Ernakulam", "Idukki", "Alappuzha", "Wayanad", "Thiruvananthapuram"]

    data = df[df["District"].isin(districts)].copy()
    data = data.sort_values(by="Date")

    fig, axes = plt.subplots(len(districts), 1, figsize=(14, 3.2 * len(districts)), sharex=True)
    if len(districts) == 1:
        axes = [axes]

    for ax, dist in zip(axes, districts):
        sub = data[data["District"] == dist]
        normal_pts = sub[sub[outlier_col] == 0]
        outlier_pts = sub[sub[outlier_col] == 1]

        # Trajectory line
        ax.plot(
            sub["Date"],
            sub["Total_Tourists"],
            color="#2b5c8f",
            linewidth=1.8,
            alpha=0.85,
            label="Arrivals",
        )

        # Inlier markers
        ax.scatter(
            normal_pts["Date"],
            normal_pts["Total_Tourists"],
            color="#4682b4",
            s=22,
            alpha=0.6,
        )

        # Outlier markers
        ax.scatter(
            outlier_pts["Date"],
            outlier_pts["Total_Tourists"],
            color="#d9381e",
            s=65,
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
            label="Detected Outlier",
        )

        ax.set_title(f"District: {dist} - Tourist Arrival Trajectory & Anomalies")
        ax.set_ylabel("Total Tourists")
        ax.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("Timeline (Year - Month)")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Visualization] Saved timeseries plot: {save_path}")
    else:
        plt.show()


def plot_multivariate_scatter(
    df: pd.DataFrame,
    outlier_col: str = "Ensemble_Outlier",
    save_path: Optional[str] = None,
):
    """
    Generates bivariate scatter plots (Domestic vs Foreign tourists, and Occupancy vs Revenue)
    differentiating inliers from anomalies.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    palette = {0: "#3470a3", 1: "#e03b24"}

    # 1. Domestic vs Foreign tourists
    sns.scatterplot(
        data=df,
        x="Domestic_Tourists",
        y="Foreign_Tourists",
        hue=outlier_col,
        palette=palette,
        alpha=0.75,
        s=50,
        ax=ax1,
    )
    ax1.set_title("Domestic vs. Foreign Tourist Footfalls")
    ax1.set_xlabel("Domestic Tourists")
    ax1.set_ylabel("Foreign Tourists")
    ax1.legend(title="Anomaly", labels=["Normal", "Outlier"])

    # 2. Hotel Occupancy vs Tourism Revenue
    sns.scatterplot(
        data=df,
        x="Hotel_Occupancy_Rate",
        y="Tourism_Revenue_Cr",
        hue=outlier_col,
        palette=palette,
        alpha=0.75,
        s=50,
        ax=ax2,
    )
    ax2.set_title("Hotel Occupancy (%) vs. Tourism Revenue (₹ Crores)")
    ax2.set_xlabel("Hotel Occupancy Rate (%)")
    ax2.set_ylabel("Tourism Revenue (₹ Cr)")
    ax2.legend(title="Anomaly", labels=["Normal", "Outlier"])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Visualization] Saved multivariate scatter plot: {save_path}")
    else:
        plt.show()


def plot_district_boxplots(
    df: pd.DataFrame,
    outlier_col: str = "Ensemble_Outlier",
    save_path: Optional[str] = None,
):
    """
    Generates horizontal box plots of tourist volumes across all 14 districts
    with highlighted anomalies.
    """
    set_plot_style()
    plt.figure(figsize=(13, 8))

    order = (
        df.groupby("District")["Total_Tourists"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    # Base boxplot
    sns.boxplot(
        data=df,
        y="District",
        x="Total_Tourists",
        order=order,
        color="#cce2f5",
        fliersize=0,
        linewidth=1.2,
    )

    # Overlay individual data points, coloring outliers
    palette = {0: "#4a7c59", 1: "#d9381e"}
    sns.stripplot(
        data=df,
        y="District",
        x="Total_Tourists",
        order=order,
        hue=outlier_col,
        palette=palette,
        dodge=False,
        jitter=0.25,
        size=4.5,
        alpha=0.7,
    )

    plt.title("Kerala Tourism Distribution & Detected Outliers by District (14 Districts)")
    plt.xlabel("Total Monthly Tourists")
    plt.ylabel("District")
    plt.legend(title="Anomaly Flag", loc="lower right", labels=["Normal", "Outlier"])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Visualization] Saved district boxplot: {save_path}")
    else:
        plt.show()


def plot_model_agreement(
    similarity_matrix: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Renders a heatmap displaying pairwise Jaccard agreement among outlier detection algorithms.
    """
    set_plot_style()
    plt.figure(figsize=(9, 7))

    # Clean display labels
    clean_labels = [c.replace("Outlier_", "") for c in similarity_matrix.columns]

    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar=True,
        square=True,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        linewidths=0.5,
    )

    plt.title("Inter-Algorithm Agreement (Jaccard Similarity Matrix)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Visualization] Saved model agreement heatmap: {save_path}")
    else:
        plt.show()


def plot_seasonal_heatmap(
    df: pd.DataFrame,
    outlier_col: str = "Ensemble_Outlier",
    save_path: Optional[str] = None,
):
    """
    Renders a heatmap showing the distribution of detected anomalies across
    Months (1-12) and Districts to reveal seasonal vulnerability patterns.
    """
    set_plot_style()
    pivot = df.pivot_table(
        index="District",
        columns="Month",
        values=outlier_col,
        aggfunc="sum",
        fill_value=0,
    )

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Reindex columns to ensure 1-12 order
    pivot = pivot.reindex(columns=range(1, 13), fill_value=0)
    pivot.columns = month_names

    plt.figure(figsize=(12, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap="Reds",
        linewidths=0.5,
        cbar_kws={"label": "Anomaly Count"},
    )

    plt.title("Seasonal Vulnerability Matrix: Outlier Frequency by Month & District")
    plt.xlabel("Month of the Year")
    plt.ylabel("District")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Visualization] Saved seasonal heatmap: {save_path}")
    else:
        plt.show()


def generate_all_visualizations(
    df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    output_dir: str = "results",
    outlier_col: str = "Ensemble_Outlier",
):
    """
    Orchestrates generation of all analytical plots and saves them to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_timeseries_outliers(
        df,
        outlier_col=outlier_col,
        save_path=os.path.join(output_dir, "timeseries_anomalies.png"),
    )

    plot_multivariate_scatter(
        df,
        outlier_col=outlier_col,
        save_path=os.path.join(output_dir, "multivariate_outliers_scatter.png"),
    )

    plot_district_boxplots(
        df,
        outlier_col=outlier_col,
        save_path=os.path.join(output_dir, "district_boxplots.png"),
    )

    plot_model_agreement(
        similarity_matrix,
        save_path=os.path.join(output_dir, "model_agreement_heatmap.png"),
    )

    plot_seasonal_heatmap(
        df,
        outlier_col=outlier_col,
        save_path=os.path.join(output_dir, "seasonal_anomaly_matrix.png"),
    )

"""
src/visualization.py
--------------------
Kerala Tourism Trend Outlier Analysis Using Outlier Detection
Role / Responsibility: Nifa (Visualization & Comparison Visuals)

This module implements reusable Matplotlib visualizations for:
1. LOF Anomaly Score Distribution (density, distribution, threshold).
2. Algorithm Detection Comparison (outlier counts, percentages, overlap breakdown).
3. 2D Feature Scatter Comparison (consensus 'Both', 'LOF Only', 'IF Only', 'Inlier').
4. Temporal / Trend Outlier Analysis (visualizing anomalies along time or index).
5. Continuous Anomaly Score Relationship (LOF vs. Isolation Forest score correlation).

All functions are designed to work seamlessly in interactive notebooks (returning
figure/axis objects) as well as automated batch scripts (saving high-resolution PNGs).
"""

import os
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set consistent academic / report visual styling
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

# Standard color palette for comparison categories
CATEGORY_PALETTE = {
    "Inlier": "#b0bec5",                 # Subtle gray-blue
    "LOF Only": "#f57c00",               # Vivid Amber / Orange
    "Isolation Forest Only": "#1976d2",  # Deep Royal Blue
    "Both": "#d32f2f",                   # Bold Crimson Red (Consensus Outliers)
}

CATEGORY_MARKERS = {
    "Inlier": "o",
    "LOF Only": "^",
    "Isolation Forest Only": "s",
    "Both": "X",
}


def _save_figure(fig: plt.Figure, save_path: Optional[str], description: str) -> None:
    """Helper to safely create parent directory and save high-resolution figure."""
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] Saved {description} to: {save_path}")


# =====================================================================
# 1. LOF SCORE DISTRIBUTION PLOT
# =====================================================================

def plot_lof_score_distribution(
    df_or_scores: Union[pd.DataFrame, np.ndarray, list],
    score_col: str = "lof_score",
    threshold: Optional[float] = None,
    bins: int = 30,
    title: str = "Distribution of Local Outlier Factor (LOF) Anomaly Scores",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the histogram and distribution of LOF anomaly scores.

    In LOF:
    - Scores close to 1.0 indicate observations with density similar to their neighbors (inliers).
    - Scores significantly greater than 1.0 (e.g. > 1.2 or > 1.5) indicate lower local density (outliers).

    Parameters
    ----------
    df_or_scores : pd.DataFrame, np.ndarray, or list
        DataFrame containing `score_col` or raw 1D array of LOF scores.
    score_col : str, default='lof_score'
        Column name if df_or_scores is a DataFrame.
    threshold : float, optional
        Threshold line to display on the plot.
    bins : int, default=30
        Number of histogram bins.
    title : str
        Plot title.
    save_path : str, optional
        If provided, saves figure to this path.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects.
    """
    if isinstance(df_or_scores, pd.DataFrame):
        if score_col not in df_or_scores.columns:
            raise KeyError(f"Column '{score_col}' not found in DataFrame.")
        scores = df_or_scores[score_col].dropna().values
    else:
        scores = np.asarray(df_or_scores).flatten()

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=100)

    # Histogram
    n, bins_edges, patches = ax.hist(
        scores, bins=bins, color="#455a64", edgecolor="white", alpha=0.85, density=False
    )

    # Median and mean lines
    median_val = np.median(scores)
    mean_val = np.mean(scores)
    ax.axvline(median_val, color="#00897b", linestyle="--", linewidth=1.8, label=f"Median LOF: {median_val:.2f}")
    ax.axvline(mean_val, color="#3949ab", linestyle=":", linewidth=1.8, label=f"Mean LOF: {mean_val:.2f}")

    if threshold is not None:
        ax.axvline(threshold, color="#d32f2f", linestyle="-", linewidth=2.0, label=f"Outlier Cutoff: {threshold:.2f}")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Local Outlier Factor (LOF Score, higher = lower local density)", fontsize=11)
    ax.set_ylabel("Frequency (Count)", fontsize=11)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="upper right")

    plt.tight_layout()
    _save_figure(fig, save_path, "LOF score distribution")

    return fig, ax


# =====================================================================
# 2. COMPARISON SUMMARY BAR CHARTS
# =====================================================================

def plot_comparison_summary(
    summary: Dict[str, Union[int, float, str, dict]],
    title: str = "Outlier Detection Comparison: LOF vs. Isolation Forest",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Visualizes the comparative findings between LOF and Isolation Forest:
    - Left Panel: Overall outlier rate detected by each algorithm.
    - Right Panel: Breakdown of detections into Both (Consensus), LOF Only, and IF Only.

    Parameters
    ----------
    summary : dict
        Comparison summary dictionary generated by `src.evaluation.compare_outlier_algorithms`.
    title : str
        Figure suptitle.
    save_path : str, optional
        Destination filepath for saved figure.

    Returns
    -------
    fig, (ax1, ax2) : tuple
        Matplotlib figure and pair of axes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=100)

    # ----------------- PANEL 1: Outlier Detection Rates -----------------
    models = ["Local Outlier Factor (LOF)", "Isolation Forest (IF)"]
    pcts = [summary["lof_outliers_pct"], summary["if_outliers_pct"]]
    counts = [summary["lof_outliers_count"], summary["if_outliers_count"]]
    colors = ["#f57c00", "#1976d2"]

    bars1 = ax1.bar(models, pcts, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Outliers Detected (% of Total Dataset)", fontsize=11)
    ax1.set_title("Overall Detection Rate by Algorithm", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylim(0, max(pcts) * 1.35 if max(pcts) > 0 else 10)
    ax1.grid(axis="y")

    for bar, count, pct in zip(bars1, counts, pcts):
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (max(pcts) * 0.03),
            f"{pct:.1f}%\n({count} records)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # ----------------- PANEL 2: Overlap Breakdown -----------------
    categories = ["Consensus\n(Both Flagged)", "LOF Only\n(Local Density)", "IF Only\n(Global Tree)"]
    cat_counts = [summary["both_outliers_count"], summary["lof_only_count"], summary["if_only_count"]]
    cat_pcts = [summary["both_outliers_pct"], summary["lof_only_pct"], summary["if_only_pct"]]
    cat_colors = [CATEGORY_PALETTE["Both"], CATEGORY_PALETTE["LOF Only"], CATEGORY_PALETTE["Isolation Forest Only"]]

    bars2 = ax2.bar(categories, cat_counts, color=cat_colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Number of Outlier Records", fontsize=11)
    ax2.set_title("Consensus vs. Algorithm-Specific Detections", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylim(0, max(cat_counts) * 1.35 if max(cat_counts) > 0 else 10)
    ax2.grid(axis="y")

    for bar, count, pct in zip(bars2, cat_counts, cat_pcts):
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (max(cat_counts) * 0.03 if max(cat_counts) > 0 else 0.5),
            f"{count} records\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Footer annotation of Jaccard similarity and concordance
    jaccard = summary.get("jaccard_similarity", 0.0)
    concordance = summary.get("concordance_rate_pct", 0.0)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        -0.02,
        f"Jaccard Similarity Index on Outliers: {jaccard:.3f}   |   Overall Inlier/Outlier Concordance: {concordance:.1f}%\n"
        "Note: In unsupervised outlier analysis, disagreement indicates complementary detection modes, not model errors.",
        ha="center",
        fontsize=9.5,
        fontstyle="italic",
        color="#37474f",
    )

    plt.tight_layout()
    _save_figure(fig, save_path, "comparison summary plot")

    return fig, (ax1, ax2)


# =====================================================================
# 3. 2D FEATURE SCATTER COMPARISON PLOT
# =====================================================================

def plot_feature_scatter_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    status_col: str = "comparison_status",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a 2D feature scatter plot highlighting inliers vs consensus outliers
    and algorithm-specific outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and `status_col`.
    x_col : str
        Column name for X axis.
    y_col : str
        Column name for Y axis.
    status_col : str, default='comparison_status'
        Column categorizing each record ('Inlier', 'LOF Only', 'Isolation Forest Only', 'Both').
    title : str, optional
        Custom title.
    save_path : str, optional
        Destination filepath for saved figure.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Specified feature columns '{x_col}' or '{y_col}' not found in DataFrame.")

    if status_col not in df.columns:
        raise KeyError(f"Status column '{status_col}' not found in DataFrame.")

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

    # Plot order: Inliers first (background), then specific algorithms, then consensus (top)
    plot_order = ["Inlier", "Isolation Forest Only", "LOF Only", "Both"]

    for category in plot_order:
        subset = df[df[status_col] == category]
        if subset.empty:
            continue

        color = CATEGORY_PALETTE.get(category, "#757575")
        marker = CATEGORY_MARKERS.get(category, "o")
        alpha = 0.4 if category == "Inlier" else 0.9
        size = 25 if category == "Inlier" else (90 if category == "Both" else 65)
        edgecolor = "none" if category == "Inlier" else "black"
        linewidth = 0.8 if category != "Inlier" else 0

        ax.scatter(
            subset[x_col],
            subset[y_col],
            c=color,
            marker=marker,
            alpha=alpha,
            s=size,
            edgecolors=edgecolor,
            linewidths=linewidth,
            label=f"{category} (n={len(subset)})",
            zorder=1 if category == "Inlier" else (4 if category == "Both" else 3),
        )

    plot_title = (
        title
        if title
        else f"Kerala Tourism Outlier Distribution: {x_col.replace('_', ' ').title()} vs. {y_col.replace('_', ' ').title()}"
    )
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=11)
    ax.grid(True)
    ax.legend(
        title="Detection Status",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        loc="best",
        fontsize=9.5,
    )

    plt.tight_layout()
    _save_figure(fig, save_path, "feature scatter comparison")

    return fig, ax


# =====================================================================
# 4. TEMPORAL / TREND OUTLIER ANALYSIS PLOT
# =====================================================================

def plot_temporal_outlier_trend(
    df: pd.DataFrame,
    date_or_index_col: str,
    metric_col: str,
    status_col: str = "comparison_status",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualizes tourism metric trends over time or sequence index, highlighting
    where LOF, Isolation Forest, or both flagged anomalies.

    Helps analyze seasonal shockpoints, off-season unexpected spikes, and festival peaks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time/sequence column, metric column, and status column.
    date_or_index_col : str
        Column for horizontal axis (e.g., date, year, month, or sequential index).
    metric_col : str
        Continuous metric column (e.g., total tourist footfall).
    status_col : str, default='comparison_status'
        Category column.
    title : str, optional
        Custom title.
    save_path : str, optional
        Destination filepath for saved figure.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    """
    if date_or_index_col not in df.columns or metric_col not in df.columns:
        raise KeyError(f"Columns '{date_or_index_col}' or '{metric_col}' not found in DataFrame.")

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=100)

    # Sort if date or chronological
    df_sorted = df.sort_values(by=date_or_index_col).reset_index(drop=True)

    # Baseline trend line
    ax.plot(
        df_sorted[date_or_index_col],
        df_sorted[metric_col],
        color="#78909c",
        linewidth=1.5,
        alpha=0.7,
        label="Observed Tourism Trend",
        zorder=1,
    )

    # Highlight anomalous points
    outlier_categories = ["Isolation Forest Only", "LOF Only", "Both"]
    for category in outlier_categories:
        subset = df_sorted[df_sorted[status_col] == category]
        if subset.empty:
            continue

        color = CATEGORY_PALETTE.get(category, "#757575")
        marker = CATEGORY_MARKERS.get(category, "o")
        size = 110 if category == "Both" else 75
        zorder = 5 if category == "Both" else 4

        ax.scatter(
            subset[date_or_index_col],
            subset[metric_col],
            color=color,
            marker=marker,
            s=size,
            edgecolors="black",
            linewidths=0.9,
            label=f"{category} ({len(subset)})",
            zorder=zorder,
        )

    plot_title = (
        title
        if title
        else f"Kerala Tourism Trend Outlier Timeline: {metric_col.replace('_', ' ').title()}"
    )
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(date_or_index_col.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=11)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

    plt.tight_layout()
    _save_figure(fig, save_path, "temporal outlier trend plot")

    return fig, ax


# =====================================================================
# 5. CONTINUOUS ANOMALY SCORE CORRELATION PLOT
# =====================================================================

def plot_score_correlation(
    df: pd.DataFrame,
    lof_score_col: str = "lof_score",
    if_score_col: Optional[str] = None,
    title: str = "Anomaly Score Correlation: LOF vs. Isolation Forest",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the relationship between continuous anomaly scores from LOF and
    Isolation Forest.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with score columns.
    lof_score_col : str, default='lof_score'
        Name of LOF score column.
    if_score_col : str, optional
        Name of Isolation Forest score column (e.g., 'if_score' or 'anomaly_score').
        If None, attempts auto-detection.
    title : str
        Plot title.
    save_path : str, optional
        Output filepath.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    """
    if lof_score_col not in df.columns:
        raise KeyError(f"LOF score column '{lof_score_col}' not found.")

    if if_score_col is None:
        candidates = ["if_score", "anomaly_score", "score", "decision_function"]
        for c in candidates:
            if c in df.columns:
                if_score_col = c
                break

    if if_score_col is None or if_score_col not in df.columns:
        print("[WARN] Continuous Isolation Forest score column not available. Skipping score correlation plot.")
        return None, None

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=100)

    # Compute Spearman rank correlation
    rank_corr = df[[lof_score_col, if_score_col]].corr(method="spearman").iloc[0, 1]

    scatter = ax.scatter(
        df[if_score_col],
        df[lof_score_col],
        c="#37474f",
        alpha=0.6,
        s=35,
        edgecolors="none",
    )

    ax.set_title(f"{title}\n(Spearman Rank Correlation: {rank_corr:.3f})", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(f"Isolation Forest Score ({if_score_col})", fontsize=11)
    ax.set_ylabel(f"Local Outlier Factor Score ({lof_score_col})", fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    _save_figure(fig, save_path, "score correlation plot")

    return fig, ax
