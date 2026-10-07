import pandas as pd
import glob
import re
from pathlib import Path

# Standardise RGB strings so they can be matched consistently across the cluster and perceptual results CSV files.
def normalise_rgb_string(s):
    r, g, b = map(int, re.findall(r"\d+", str(s)))
    return f"({r}, {g}, {b})"

# Load merged clusters
merge_df = pd.read_csv("colour_clusters_threshold_8.csv")

def parse_rgb_list(rgb_string):
    """Extract list of RGB tuples from '(r,g,b); (r,g,b)' string."""
    matches = re.findall(r"\((\d+),\s*(\d+),\s*(\d+)\)", rgb_string)
    return [f"({r}, {g}, {b})" for r, g, b in matches]

# Build a lookup from each original RGB colour to its merged colour group.
# This allows the map-level perceptual results to be assigned to the corresponding perceptual colour cluster.
rgb_to_group = {}

# Store the medoid RGB value for each merged colour group. The medoid is used as the representative colour for the group.
group_to_medoid = {
    idx + 1: row["Medoid RGB"]
    for idx, row in merge_df.iterrows()
}

for idx, row in merge_df.iterrows():
    group_id = idx + 1  # 1,2,3,...
    rgb_list = parse_rgb_list(row["RGB Colors"])
    for rgb in rgb_list:
        rgb_to_group[normalise_rgb_string(rgb)] = group_id

# Process each map file
map_files = glob.glob("*_crosshair_perceptual_summary_with_composites.csv")

master_rows = []

for file in map_files:
    df = pd.read_csv(file)

    # Retain only the measures required for the merged colour summaries.
    df = df[["Crosshair RGB",
             "Composite_WeightedMeanAPCA_Pass",
             "Average_APCA",
             "Weighted_Mean_APCA",
             "Sum_Weighted_Pass",
             "Percent_Pass_Rate",
             "Average_DeltaE",
             "Weighted_Mean_DeltaE",
             "Thickness_1920x1080_71cm",
             "Thickness_1280x960_71cm",
             "Thickness_1920x1080_30cm",
             "Thickness_1280x960_30cm"
             ]
    ].copy()

    # Standardise RGB formatting and assign each colour to its merged perceptual colour group.
    df["Crosshair RGB"] = df["Crosshair RGB"].apply(normalise_rgb_string)
    df["Merge_Group"] = df["Crosshair RGB"].map(rgb_to_group)

    df["Medoid_RGB"] = df["Merge_Group"].map(group_to_medoid)

    # Calculate the mean and standard deviation of each perceptual measure
    # for the colours belonging to each merged colour group within the specific map.
    group_avg = (
        df.groupby("Merge_Group")
        .agg(
            Group_Medoid_RGB=("Merge_Group", lambda g: group_to_medoid[g.iloc[0]]),

            # Number of colours in each merged group
            Number_of_Colours=("Merge_Group", "size"),

            # Composite
            Group_Mean_Composite=("Composite_WeightedMeanAPCA_Pass", "mean"),
            Group_SD_Composite=("Composite_WeightedMeanAPCA_Pass", "std"),

            # Perceptual contrast (APCA)
            Group_Mean_APCA=("Average_APCA", "mean"),
            Group_SD_APCA=("Average_APCA", "std"),

            # Weighted perceptual contrast (APCA)
            Group_Mean_Weighted_APCA=("Weighted_Mean_APCA", "mean"),
            Group_SD_Weighted_APCA=("Weighted_Mean_APCA", "std"),

            # Weighted Pass measure
            Group_Mean_SumWeightedPass=("Sum_Weighted_Pass", "mean"),
            Group_SD_SumWeightedPass=("Sum_Weighted_Pass", "std"),

            # Percent Pass Rate
            Group_Mean_Percent_Pass_Rate=("Percent_Pass_Rate", "mean"),
            Group_SD_Percent_Pass_Rate=("Percent_Pass_Rate", "std"),

            # DeltaE
            Group_Mean_DeltaE=("Average_DeltaE", "mean"),
            Group_SD_DeltaE=("Average_DeltaE", "std"),

            # Weighted DeltaE
            Group_Mean_Weighted_DeltaE=("Weighted_Mean_DeltaE", "mean"),
            Group_SD_Weighted_DeltaE=("Weighted_Mean_DeltaE", "std"),

            # Thicknesses
            Group_Mean_Thickness_1920x1080_71cm=("Thickness_1920x1080_71cm", "mean"),
            Group_SD_Thickness_1920x1080_71cm=("Thickness_1920x1080_71cm", "std"),

            Group_Mean_Thickness_1280x960_71cm=("Thickness_1280x960_71cm", "mean"),
            Group_SD_Thickness_1280x960_71cm=("Thickness_1280x960_71cm", "std"),

            Group_Mean_Thickness_1920x1080_30cm=("Thickness_1920x1080_30cm", "mean"),
            Group_SD_Thickness_1920x1080_30cm=("Thickness_1920x1080_30cm", "std"),

            Group_Mean_Thickness_1280x960_30cm=("Thickness_1280x960_30cm", "mean"),
            Group_SD_Thickness_1280x960_30cm=("Thickness_1280x960_30cm", "std"),
        )
        .reset_index()
    )

    # Add the merged-group summary statistics back to each individual crosshair colour in the map-specific results.
    df = df.merge(group_avg, on="Merge_Group", how="left")

    # Derive the map name from the input filename and save the map-specific merged crosshair results.
    map_name = Path(file).stem.replace(
        "_crosshair_perceptual_summary_with_composites", ""
    )

    out_name = f"{map_name}_merged_crosshair_summary.csv"
    df.to_csv(out_name, index=False)

    # Collect for master file
    group_avg["Map"] = map_name
    master_rows.append(group_avg)

# Combine the group summaries from all maps so each merged crosshair colour group can be summarised across the full dataset.
master_df = pd.concat(master_rows, ignore_index=True)

# Calculate across-map means and standard deviations for each merged crosshair colour group.
# Across-map SD describes variation in the group mean between maps.
# Average within-group SD describes variation between individual crosshair colours within the merged group, averaged across maps.
master_summary = (
    master_df.groupby("Merge_Group")
    .agg(
        Medoid_RGB=("Group_Medoid_RGB", "first"),

        #Number of crosshair colours in each group
        Number_of_Colours=("Number_of_Colours", "first"),

        # Map
        Number_of_Maps=("Map", "nunique"),

        # Composite score
        Mean_Composite_Across_Maps=("Group_Mean_Composite", "mean"),
        SD_Composite_Across_Maps=("Group_Mean_Composite", "std"),

        # Perceptual contrast (APCA)
        Mean_APCA_Across_Maps=("Group_Mean_APCA", "mean"),
        SD_APCA_Across_Maps=("Group_Mean_APCA", "std"),
        Average_WithinGroup_APCA_SD=("Group_SD_APCA", "mean"),

        # Weighted perceptual contrast (APCA)
        Mean_Weighted_APCA_Across_Maps=("Group_Mean_Weighted_APCA", "mean"),
        SD_Weighted_APCA_Across_Maps=("Group_Mean_Weighted_APCA", "std"),
        Average_WithinGroup_Weighted_APCA_SD=("Group_SD_Weighted_APCA", "mean"),

        # Weighted Pass measure
        Mean_SumWeightedPass_Across_Maps=("Group_Mean_SumWeightedPass", "mean"),
        SD_SumWeightedPass_Across_Maps=("Group_Mean_SumWeightedPass", "std"),

        # Percent Pass Rate
        Mean_Percent_Pass_Rate_Across_Maps=("Group_Mean_Percent_Pass_Rate", "mean"),
        SD_Percent_Pass_Rate_Across_Maps=("Group_Mean_Percent_Pass_Rate", "std"),

        # DeltaE
        Mean_DeltaE_Across_Maps=("Group_Mean_DeltaE", "mean"),
        SD_DeltaE_Across_Maps=("Group_Mean_DeltaE", "std"),
        Average_WithinGroups_DeltaE_SD=("Group_SD_DeltaE", "mean"),

        # Weighted DeltaE
        Mean_Weighted_DeltaE_Across_Maps=("Group_Mean_Weighted_DeltaE", "mean"),
        SD_Weighted_DeltaE_Across_Maps=("Group_Mean_Weighted_DeltaE", "std"),
        Average_WithinGroups_Weighted_DeltaE_SD=("Group_SD_Weighted_DeltaE", "mean"),

        # Thicknesses
        Mean_Thickness_1920x1080_71cm_Across_Maps=("Group_Mean_Thickness_1920x1080_71cm", "mean"),
        SD_Thickness_1920x1080_71cm_Across_Maps=("Group_Mean_Thickness_1920x1080_71cm", "std"),

        Mean_Thickness_1280x960_71cm_Across_Maps=("Group_Mean_Thickness_1280x960_71cm", "mean"),
        SD_Thickness_1280x960_71cm_Across_Maps=("Group_Mean_Thickness_1280x960_71cm", "std"),

        Mean_Thickness_1920x1080_30cm_Across_Maps=("Group_Mean_Thickness_1920x1080_30cm", "mean"),
        SD_Thickness_1920x1080_30cm_Across_Maps=("Group_Mean_Thickness_1920x1080_30cm", "std"),

        Mean_Thickness_1280x960_30cm_Across_Maps=("Group_Mean_Thickness_1280x960_30cm", "mean"),
        SD_Thickness_1280x960_30cm_Across_Maps=("Group_Mean_Thickness_1280x960_30cm", "std")

    )
    .reset_index()
)

master_summary.to_csv("MASTER_merged_crosshair_summary.csv", index=False)

master_df.to_csv("MASTER_merged_crosshair_summary_per_map.csv", index=False)

print("Done. Created map files and MASTER summary.")
