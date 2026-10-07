# Crosshair-Colour-Analysis
This repository contains scripts for collecting and analysing colour data from **Counter-Strike 2 (CS2)** gameplay and evaluating how different crosshair colours perform against the colours present in gameplay environments.

The analysis is organised into four stages:

### 1. Video Clustering Analysis — `1_Video_Clustering_Analysis`
This stage analyses prepared CS2 gameplay videos on a **per-map basis**. Colour data are collected from pixels within a selected region of each video frame and converted to CIELAB colour space. Clustering analysis is then performed to identify the most prominent colours present within each video.

The resulting cluster colours are compared against a predetermined list of crosshair colours. This crosshair colour list was derived from professional CS2 player data; the relevant data are provided in the `data` folder.

The analysis produces several CSV files and figures. Two outputs are used in subsequent stages:

* **`perceptual_results`** — contains the colour-difference and perceptual-contrast calculations between each crosshair colour and the identified cluster colours.
* **`summary_results`** — contains the clustering model-selection results and the results from the final selected cluster solution.

This stage is performed separately for each map.

### 2. Creating Crosshair Summary Files — `2_Creating_Crosshair_Summary_Files`
This stage processes the `perceptual_results` CSV files produced during Stage 1.

For each crosshair colour, the cluster-level colour-difference and perceptual-contrast results are aggregated using the cluster weights. Summary metrics are then calculated, including weighted perceptual measures, pass rates, composite scores, and estimated crosshair line thickness under different display and viewing-distance conditions.

The resulting files are named:

`*_crosshair_perceptual_summary_with_composites.csv`

### 3. Merging Similar Crosshair Colours — `3_Merging_Similar_Crosshairs`
This stage identifies crosshair colours that are sufficiently similar to be represented as a single colour group.

Although two RGB colours may differ by one or more values in their red, green, or blue components, this does not necessarily mean that they are perceptually distinguishable. CIEDE2000 colour difference in CIELAB space is therefore used to group perceptually similar crosshair colours.

The resulting clusters provide a simplified representation of the original crosshair colour set. Each group is represented by a medoid RGB colour, which corresponds to an actual colour from the original crosshair colour list.

The resulting file is:

`colour_clusters_threshold_8.csv`

### 4. Map Merged Crosshair Summary — `4_Map_Merged_Crosshair_Summary`
This stage combines the outputs from Stages 2 and 3.

The `*_crosshair_perceptual_summary_with_composites.csv` files are matched to the merged crosshair colour groups from `colour_clusters_threshold_8.csv`. The perceptual results are then aggregated within each merged colour group to produce simplified results for each map.

The script also combines the map-level results to produce an overall summary across all maps.

The main output files are:

* **`*_merged_crosshair_summary.csv`** — simplified crosshair colour results for each individual map.
* **`MASTER_merged_crosshair_summary.csv`** — overall summary of the merged crosshair colour groups across all maps.
* **`MASTER_merged_crosshair_summary_per_map.csv`** — merged colour group results retained separately for each map.

## Analysis Workflow
The overall workflow can therefore be summarised as:

**CS2 gameplay videos** -> **Colour extraction and clustering** -> **Crosshair colour × environment perceptual analysis** -> **Crosshair-level summary metrics and composite scores** -> **Merging perceptually similar crosshair colours** -> **Map-level and across-map summary results**
