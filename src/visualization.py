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
