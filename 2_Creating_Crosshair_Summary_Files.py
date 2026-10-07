import pandas as pd
import glob
from pathlib import Path

files = glob.glob("*_perceptual_results.csv")

for file in files:
    print(f"Processing: {file}")

    # Load each file into dataframe
    df = pd.read_csv(file)

    # Ensure correct types
    df["Cluster Weight"] = df["Cluster Weight"].astype(float)
    df["DeltaE"] = df["DeltaE"].astype(float)
    df["APCA_Lc"] = df["APCA_Lc"].astype(float)
    df["APCA_Pass"] = df["APCA_Pass"].astype(bool)

    # Absolute Perceptual Contrast (APCA)
    df["APCA_abs"] = df["APCA_Lc"].abs()

    # Precompute weighted perceptual contrast and colour difference metrics
    # Weight represents the proportion of pixels associated with each cluster colour
    df["Weighted_DeltaE"] = df["Cluster Weight"] * df["DeltaE"]
    df["Weighted_APCA"] = df["Cluster Weight"] * df["APCA_abs"]
    df["Weighted_Pass"] = df["Cluster Weight"] * df["APCA_Pass"].astype(int)

    # Grouped summary
    results = (
        df.groupby("Crosshair RGB", as_index=False)
          .agg(
              Average_DeltaE=("DeltaE", "mean"),
              Average_APCA=("APCA_abs", "mean"),
              Sum_Weighted_DeltaE=("Weighted_DeltaE", "sum"),
              Sum_Weighted_APCA=("Weighted_APCA", "sum"),
              Sum_Weighted_Pass=("Weighted_Pass", "sum"),
              Total_Weight=("Cluster Weight", "sum"),
          )
    )

    # Weighted means
    results["Weighted_Mean_DeltaE"] = (
        results["Sum_Weighted_DeltaE"] / results["Total_Weight"]
    )

    results["Weighted_Mean_APCA"] = (
        results["Sum_Weighted_APCA"] / results["Total_Weight"]
    )

    # Line Thickness Approximation
    # Base thickness derived from perceptual APCA relationship
    # Thickness = (Lc(w) / 100)^(-1.5)
    # Prevent division issues for extremely small APCA values
    epsilon = 1e-6
    Lc_w = results["Weighted_Mean_APCA"].clip(lower=epsilon)

    results["Thickness_Base"] = (Lc_w / 100.0) ** (-1.5)

    # Apply empirically derived scaling constants
    results["Thickness_1920x1080_71cm"] = results["Thickness_Base"] * 0.954
    results["Thickness_1280x960_71cm"]  = results["Thickness_Base"] * 0.636
    results["Thickness_1920x1080_30cm"] = results["Thickness_Base"] * 0.403
    results["Thickness_1280x960_30cm"]  = results["Thickness_Base"] * 0.269

    # Percent pass rate
    results["Percent_Pass_Rate"] = (
        results["Sum_Weighted_Pass"] / results["Total_Weight"]
    ) * 100

    # Normalisation
    APCA_MAX = 108

    results["Norm_Average_APCA"] = (results["Average_APCA"] / APCA_MAX).clip(0, 1)
    results["Norm_Weighted_Mean_APCA"] = (results["Weighted_Mean_APCA"] / APCA_MAX).clip(0, 1)
    results["Norm_Sum_Weighted_APCA"] = (results["Sum_Weighted_APCA"] / APCA_MAX).clip(0, 1)
    results["Norm_Pass_Rate"] = results["Percent_Pass_Rate"] / 100.0

    # Composite visibility score calculation which combines the normalised weighted APCA contrast and the
    # weighted proportion of clusters passing the APCA threshold of 15. Both components contribute equally to the score.
    results["Composite_WeightedMeanAPCA_Pass"] = (
        results["Norm_Weighted_Mean_APCA"] + results["Norm_Pass_Rate"]
    ) / 2

    # Save file with map name
    map_name = Path(file).stem.replace("_perceptual_results", "")
    out_file = f"{map_name}_crosshair_perceptual_summary_with_composites.csv"

    results.to_csv(out_file, index=False)
    print(f"Saved: {out_file}\n")