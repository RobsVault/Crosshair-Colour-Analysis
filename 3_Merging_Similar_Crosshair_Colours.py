import csv
import numpy as np
from colormath.color_diff import delta_e_cie2000
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

# Professional crosshair colours
pro_colors = [
    (255, 255, 255), (255, 255, 254), (255, 255, 251), (255, 255, 250), (255, 255, 200),
    (255, 255, 170), (255, 255, 165), (255, 255, 155), (255, 255, 2), (255, 255, 0),
    (255, 250, 255), (255, 247, 255), (255, 100, 255), (255, 56, 128), (255, 50, 129),
    (255, 12, 255), (255, 11, 17), (255, 1, 122), (255, 0, 255), (255, 0, 250),
    (255, 0, 160), (255, 0, 149), (255, 0, 141), (255, 0, 136), (255, 0, 127),
    (255, 0, 70), (255, 0, 42), (255, 0, 2), (255, 0, 0), (254, 255, 255),
    (250, 255, 255), (250, 250, 250), (250, 250, 50), (250, 0, 0), (232, 0, 0),
    (207, 255, 255), (200, 255, 255), (197, 255, 255), (196, 255, 0), (187, 177, 255),
    (181, 0, 255), (175, 220, 255), (175, 98, 255), (167, 255, 16), (164, 106, 255),
    (159, 255, 202), (155, 255, 255), (155, 255, 0), (148, 0, 211), (135, 255, 0),
    (128, 255, 255), (112, 165, 255), (100, 255, 192), (100, 255, 120), (100, 0, 0),
    (88, 206, 255), (85, 255, 165), (80, 200, 120), (77, 33, 177), (52, 155, 255),
    (50, 255, 255), (50, 255, 200), (50, 255, 169), (50, 255, 168), (50, 255, 155),
    (50, 255, 154), (50, 255, 98), (50, 255, 50), (50, 255, 0), (50, 250, 255),
    (50, 250, 250), (50, 250, 185), (50, 250, 175), (50, 250, 170), (50, 250, 160),
    (50, 250, 154), (50, 250, 50), (50, 250, 0), (50, 200, 150), (42, 255, 255),
    (27, 255, 144), (27, 195, 144), (25, 255, 171), (25, 255, 165), (23, 50, 50),
    (18, 255, 255), (16, 255, 165), (15, 255, 255), (10, 255, 145), (5, 255, 255),
    (5, 255, 241), (4, 250, 0), (1, 255, 255), (1, 255, 169), (1, 255, 155),
    (1, 1, 1), (0, 255, 255), (0, 255, 254), (0, 255, 252), (0, 255, 250),
    (0, 255, 244), (0, 255, 241), (0, 255, 207), (0, 255, 200), (0, 255, 199),
    (0, 255, 193), (0, 255, 192), (0, 255, 190), (0, 255, 184), (0, 255, 182),
    (0, 255, 180), (0, 255, 178), (0, 255, 177), (0, 255, 175), (0, 255, 173),
    (0, 255, 170), (0, 255, 169), (0, 255, 168), (0, 255, 167), (0, 255, 166),
    (0, 255, 165), (0, 255, 164), (0, 255, 162), (0, 255, 160), (0, 255, 159),
    (0, 255, 157), (0, 255, 155), (0, 255, 154), (0, 255, 153), (0, 255, 152),
    (0, 255, 150), (0, 255, 145), (0, 255, 144), (0, 255, 141), (0, 255, 140),
    (0, 255, 138), (0, 255, 136), (0, 255, 135), (0, 255, 133), (0, 255, 131),
    (0, 255, 130), (0, 255, 127), (0, 255, 125), (0, 255, 123), (0, 255, 121),
    (0, 255, 120), (0, 255, 118), (0, 255, 103), (0, 255, 65), (0, 255, 53),
    (0, 255, 4), (0, 255, 1), (0, 255, 0), (0, 252, 150), (0, 250, 255),
    (0, 250, 160), (0, 250, 144), (0, 250, 0), (0, 249, 151), (0, 248, 31),
    (0, 245, 233), (0, 238, 144), (0, 231, 199), (0, 218, 26), (0, 205, 0),
    (0, 201, 200), (0, 200, 255), (0, 175, 0), (0, 172, 170), (0, 161, 0),
    (0, 154, 255), (0, 144, 238), (0, 100, 255), (0, 99, 165), (0, 5, 255),
    (0, 4, 135), (0, 0, 255), (0, 0, 176), (0, 0, 0),
]

additional_colors = [
    (128, 128, 128), (192, 192, 192), (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128), (255, 165, 0), (75, 0, 130),
    (255, 0, 150), (34, 139, 34), (180, 255, 255), (255, 200, 0), (220, 190, 255),
    (250, 0, 100),
]

all_colors = np.array(pro_colors + additional_colors, dtype=np.uint8)

# Convert to LAB
# lab = color.rgb2lab(all_colors[np.newaxis, :, :] / 255.0)[0]
def rgb_to_lab(rgb):
    srgb = sRGBColor(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    lab = convert_color(srgb, LabColor)
    return np.array([lab.lab_l, lab.lab_a, lab.lab_b])

lab = np.array([rgb_to_lab(c) for c in all_colors])

# Compute ΔE00 matrix
n = len(lab)
deltaE_matrix = np.zeros((n, n))
for i in range(n):
    lab1 = LabColor(*lab[i])
    for j in range(i + 1, n):
        lab2 = LabColor(*lab[j])
        deltaE_matrix[i, j] = deltaE_matrix[j, i] = delta_e_cie2000(lab1, lab2)

# Agglomerative clustering
condensed_dist = squareform(deltaE_matrix)
Z = linkage(condensed_dist, method='average')
threshold = 8  # ΔE equal to or below which colours are considered similar and assigned to the same cluster
labels = fcluster(Z, t=threshold, criterion='distance')

# Organize clusters
clusters = {}
for idx, label in enumerate(labels):
    clusters.setdefault(label, []).append(
        tuple(int(x) for x in all_colors[idx])
    )

# Calculate the LAB centroid for each cluster.
# The centroid represents the mean perceptual position of the colours in the cluster,
# This may not correspond to an original crosshair colour.
lab_clusters = {}
for label, members in clusters.items():
    member_indices = [i for i, l in enumerate(labels) if l == label]
    lab_avg = lab[member_indices].mean(axis=0)
    lab_clusters[label] = lab_avg

def lab_to_rgb(lab):
    lab_obj = LabColor(*lab)
    srgb = convert_color(lab_obj, sRGBColor)
    r = int(round(srgb.clamped_rgb_r * 255))
    g = int(round(srgb.clamped_rgb_g * 255))
    b = int(round(srgb.clamped_rgb_b * 255))
    return (r, g, b)

rgb_centroids = {}

for label, lab_avg in lab_clusters.items():
    rgb_centroids[label] = lab_to_rgb(lab_avg)

# Identify the medoid for each cluster.
# Unlike the centroid, the medoid is an actual colour from the original dataset.
# It is the colour whose LAB value is closest to the cluster centroid according to CIEDE2000.
medoid_rgb = {}

for label, members in clusters.items():

    member_indices = [i for i, l in enumerate(labels) if l == label]

    lab_avg = lab_clusters[label]

    distances = []

    for idx in member_indices:
        d = delta_e_cie2000(
            LabColor(*lab_avg),
            LabColor(*lab[idx])
        )
        distances.append(d)

    closest_idx = member_indices[np.argmin(distances)]

    medoid_rgb[label] = tuple(
        int(x) for x in all_colors[closest_idx]
    )

# Calculate the perceptual spread of each cluster relative to its medoid.
# These statistics describe how closely the merged colours are grouped around the representative colour.
cluster_medoid_stats = {}

for label, members in clusters.items():

    member_indices = [i for i, l in enumerate(labels) if l == label]

    # Find the LAB value of the medoid
    medoid_idx = np.where(
        np.all(all_colors == medoid_rgb[label], axis=1)
    )[0][0]

    medoid_lab = lab[medoid_idx]

    member_deltaEs = []

    # Distance from every colour in a merged group to the medoid
    for idx in member_indices:
        if idx == medoid_idx:
            continue

        d = delta_e_cie2000(
            LabColor(*lab[idx]),
            LabColor(*medoid_lab)
        )
        member_deltaEs.append(d)

    if member_deltaEs:
        cluster_medoid_stats[label] = (
            np.mean(member_deltaEs),
            np.std(member_deltaEs),
            np.max(member_deltaEs)
        )
    else:
        cluster_medoid_stats[label] = (0.0, 0.0, 0.0)

# Calculate the overall spread within each cluster using pairwise CIEDE2000 differences between all unique colour pairs.
#
# These statistics complement the medoid-based statistics above:
# medoid based values describe the spread around a representative colour,
# while pairwise values describe the overall perceptual spread of the cluster.
cluster_pairwise_stats = {}

for label, members in clusters.items():

    member_indices = [i for i, l in enumerate(labels) if l == label]

    pairwise_deltaEs = []

    # Compare every unique pair of colours in the cluster
    for i in range(len(member_indices)):
        for j in range(i + 1, len(member_indices)):
            pairwise_deltaEs.append(
                deltaE_matrix[member_indices[i], member_indices[j]]
            )

    if pairwise_deltaEs:
        cluster_pairwise_stats[label] = (
            np.mean(pairwise_deltaEs),
            np.std(pairwise_deltaEs),
            np.max(pairwise_deltaEs)
        )
    else:
        cluster_pairwise_stats[label] = (
            0.0,
            0.0,
            0.0
        )

# Write to CSV
filename = f"colour_clusters_{threshold}.csv"
with open(filename, "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Cluster",
        "Cluster Size",
        "RGB Colors",
        "LAB Avg (L*,a*,b*)",
        "Centroid RGB",
        "Medoid RGB",
        "Centroid-to-Medoid DeltaE00",
        "Mean DeltaE00 to Medoid",
        "Std Dev DeltaE00 to Medoid",
        "Max DeltaE00 to Medoid",
        "Mean Pairwise DeltaE00",
        "Std Dev Pairwise DeltaE00",
        "Max Pairwise DeltaE00"
    ])

    for label, members in clusters.items():
        cluster_size = len(members)
        lab_avg = lab_clusters[label]
        lab_avg_str = ", ".join(f"{x:.2f}" for x in lab_avg)
        rgb_list_str = "; ".join(str(c) for c in members)
        centroid_rgb = rgb_centroids[label]
        representative_rgb = medoid_rgb[label]

        # Calculate ΔE between centroid and medoid
        centroid_lab = lab_clusters[label]

        medoid_idx = np.where(
            np.all(all_colors == medoid_rgb[label], axis=1)
        )[0][0]

        medoid_lab = lab[medoid_idx]

        centroid_medoid_deltaE = delta_e_cie2000(
            LabColor(*centroid_lab),
            LabColor(*medoid_lab)
        )

        mean_deltaE, std_deltaE, max_deltaE = cluster_medoid_stats[label]
        mean_pairwise, std_pairwise, max_pairwise = cluster_pairwise_stats[label]

        writer.writerow([
            label,
            cluster_size,
            rgb_list_str,
            lab_avg_str,
            centroid_rgb,
            representative_rgb,
            round(centroid_medoid_deltaE, 2),
            round(mean_deltaE, 2),
            round(std_deltaE, 2),
            round(max_deltaE, 2),
            round(mean_pairwise, 2),
            round(std_pairwise, 2),
            round(max_pairwise, 2)
        ])

print("Clusters saved to color_clusters.csv")

