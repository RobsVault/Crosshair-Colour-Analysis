import cv2
import numpy as np
if not hasattr(np, "asscalar"):
    np.asscalar = lambda x: x.item()
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from sklearn.mixture import GaussianMixture
from pathlib import Path
from colormath.color_objects import LabColor
from skimage.color import rgb2lab, lab2rgb
from colormath.color_diff import delta_e_cie2000
from itertools import combinations

# Give me logging details of the processes occurring through the code
def log(msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")

# Region class - specifies the variables for this class
# I am mimicking the area utilised by Jeong et al., 2024
# (https://www.researchgate.net/publication/377643936_Investigating_the_gaze_control_ability_of_VALORANT_players_using_a_Python_based_tool)
class CircularROI:
    def __init__(self, center_x, center_y, radius):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

# Colour conversion: RGB → Lab
# We use LAB because the Euclidean distances roughly reflect human perceptual differences
# We need in LAB colour space for perceptual difference calculation
def rgb_to_lab(rgb_tuple):
    rgb = np.array(rgb_tuple, dtype=np.float32)[np.newaxis, np.newaxis, :] / 255.0
    lab = rgb2lab(rgb)[0, 0]
    return lab

# Extract pixels from the area of interest at each frame like Jeong et al., 2024
def extract_circular_pixels(frame, roi):

    h, w = frame.shape[:2]

    yy, xx = np.ogrid[:h, :w]

    mask = ((xx - roi.center_x) ** 2 +
            (yy - roi.center_y) ** 2) <= roi.radius ** 2

    pixels = frame[mask]

    if pixels.size == 0:
        raise ValueError("Circular ROI returned no pixels.")

    return pixels.reshape(-1, 3)

# Remove the red artifact pixels that we were finding when clustering our data
def remove_red_artifact(lab_pixels):
    target = np.array([53, 80, 67])
    dist = np.linalg.norm(lab_pixels - target, axis=1)
    return lab_pixels[dist > 2.0]

# Open the videos in the folder
# Skip every 6 frames
# Convert cv2 from BGR to RGB
# Collect all the pixels from this frame
# Convert all pixels to LAB colour space and put them into a list and create stacked dataframe
def collect_lab_pixels_from_folder(folder_path, roi,
                                   frame_skip=6,
                                   sample_cap=50000000):

    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = [f for f in folder_path.iterdir()
                   if f.suffix.lower() in video_extensions]

    np.random.shuffle(video_files)

    reservoir = []
    total_seen = 0
    rng = np.random.default_rng()

    for video_path in video_files:

        log(f"START processing video: {video_path.name}")
        video_pixel_count = 0

        print(f"Collecting from {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Visually check the region is correct being sampled
            cv2.circle(frame_rgb,
                       (roi.center_x, roi.center_y),
                       roi.radius,
                       (255, 0, 0),
                       2)

            if frame_idx == 0:
                cv2.imwrite("roi_debug_frame.png",
                            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            )

            region_pixels = extract_circular_pixels(frame_rgb, roi)

            # Convert ONLY those ROI pixels to Lab
            region_pixels = region_pixels.astype(np.float32) / 255.0
            lab_pixels = rgb2lab(region_pixels.reshape(-1, 1, 3)).reshape(-1, 3)

            # Remove artifact pixels
            lab_pixels = remove_red_artifact(lab_pixels)

            video_pixel_count += len(lab_pixels)

            for lab_vector in lab_pixels:

                total_seen += 1

                if len(reservoir) < sample_cap:
                    reservoir.append(lab_vector)
                else:
                    j = rng.integers(0, total_seen)
                    if j < sample_cap:
                        reservoir[j] = lab_vector

            frame_idx += 1

        cap.release()
        log(f"FINISHED {video_path.name} | Pixels extracted: {video_pixel_count:,}")

    # shuffle data around after sampling is complete
    np.random.shuffle(reservoir)
    reservoir = np.array(reservoir, dtype=np.float32)

    log(f"Reservoir sampling complete | Total sampled pixels: {len(reservoir):,}")
    return reservoir

# Calculate BIC (BIC=−2ln(L)+kln(n)) - L = Likelihood, k = # of parameters, n = # of samples
# lower BIC is better as number of clusters increases that is multiplied by the Ln(number of samples)
# We take a gradient to see if improvement slows dramatically at any point, helps with determining optimal number of clusters
# BIC and its gradient are visually inspected; final k is manually selected.
def gmm_model_selection(pixels, max_k, output_folder):

    bic_values = []
    bhat_distance_means = []
    bhat_distance_stds = []
    bhat_distance_mins = []
    bhat_distance_maxs = []
    bhat_coefficient_means = []
    bhat_coefficient_stds = []
    bhat_coefficient_mins = []
    bhat_coefficient_maxs = []

    k_values = list(range(2, max_k + 1))

    for k in k_values:

        model = GaussianMixture(
            n_components=k,
            covariance_type='full',
            n_init = 1,
            verbose = 1,
            random_state = 1
        )

        model.fit(pixels)
        bic = model.bic(pixels)
        (bhat_distance_mean, bhat_distance_std, bhat_distance_min, bhat_distance_max,
         bhat_coefficient_mean, bhat_coefficient_std, bhat_coefficient_min, bhat_coefficient_max, _) = compute_bhattacharyya_stats(model)

        bhat_distance_means.append(bhat_distance_mean)
        bhat_distance_stds.append(bhat_distance_std)
        bhat_distance_mins.append(bhat_distance_min)
        bhat_distance_maxs.append(bhat_distance_max)
        bhat_coefficient_means.append(bhat_coefficient_mean)
        bhat_coefficient_stds.append(bhat_coefficient_std)
        bhat_coefficient_mins.append(bhat_coefficient_min)
        bhat_coefficient_maxs.append(bhat_coefficient_max)
        bic_values.append(bic)

    bic_values = np.array(bic_values)
    bic_gradient = np.gradient(bic_values)
    bhat_distance_means = np.array(bhat_distance_means)
    bhat_distance_stds = np.array(bhat_distance_stds)
    bhat_distance_mins = np.array(bhat_distance_mins)
    bhat_distance_maxs = np.array(bhat_distance_maxs)
    bhat_coefficient_means = np.array(bhat_coefficient_means)
    bhat_coefficient_stds = np.array(bhat_coefficient_stds)
    bhat_coefficient_mins = np.array(bhat_coefficient_mins)
    bhat_coefficient_maxs = np.array(bhat_coefficient_maxs)

    # Plots (Show + Save)
    # BIC
    plt.figure()
    plt.plot(k_values, bic_values, marker="o")
    plt.xticks(range(min(k_values), max(k_values) + 1, 1))
    plt.title(f"GMM BIC")
    plt.xlabel("Number of Clusters")
    plt.ylabel("BIC")
    plt.savefig(output_folder / f"GMM_BIC.png")
    plt.show()

    # BIC Gradient
    plt.figure()
    plt.plot(k_values, bic_gradient, marker="o")
    plt.xticks(range(min(k_values), max(k_values) + 1, 1))
    plt.title(f"GMM BIC Gradient")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Gradient")
    plt.savefig(output_folder / f"GMM_gradient.png")
    plt.show()

    # Final k selected manually after visual inspection of BIC and BIC gradient
    final_k = int(input("Enter preferred k for GMM via visual inspection (Elbow Method): "))

    print(f"Final selected k for GMM: {final_k}")

    return (final_k, k_values, bic_values, bic_gradient,
            bhat_distance_means, bhat_distance_stds, bhat_distance_mins, bhat_distance_maxs,
            bhat_coefficient_means, bhat_coefficient_stds, bhat_coefficient_mins, bhat_coefficient_maxs)

# Perceptual contrast calculations (APCA)
def srgb_to_luminance(rgb):
    rgb = np.array(rgb) / 255.0

    Y = (
        (rgb[0] ** 2.4) * 0.2126 +
        (rgb[1] ** 2.4) * 0.7152 +
        (rgb[2] ** 2.4) * 0.0722
    )
    return Y

def compute_apca_lc(text_rgb, bg_rgb):

    Y_text_raw = srgb_to_luminance(text_rgb)
    Y_bg_raw = srgb_to_luminance(bg_rgb)

    Y_text = Y_text_raw
    Y_bg = Y_bg_raw

    if Y_text < 0.022:
        Y_text = ((0.022 - Y_text) ** 1.414) + Y_text
    if Y_bg < 0.022:
        Y_bg = ((0.022 - Y_bg) ** 1.414) + Y_bg

    # Determine polarity
    if Y_text < Y_bg:
        # dark crosshair/text on light map colour/background
        contrast = (Y_bg ** 0.56) - (Y_text ** 0.57)
    else:
        # light crosshair/text on dark map colour/background
        contrast = (Y_bg ** 0.65) - (Y_text ** 0.62)

    contrast *= 1.14

    if abs(contrast) < 0.1:
        return 0.0

    if contrast > 0:
        return (contrast - 0.027) * 100
    else:
        return (contrast + 0.027) * 100

# Delta E CIE2000 calculation
def compute_deltaE(cluster_lab, cluster_rgb, weights,
                   candidate_rgb,
                   candidate_lab,
                   model_name
                   ):

    records = []

    for i, lab_vals in enumerate(cluster_lab):

        lab_cluster = LabColor(*lab_vals, illuminant='d65')
        cluster_rgb_tuple = tuple(map(int, cluster_rgb[i]))

        # Clean LAB tuple for CSV
        clean_lab_tuple = tuple(float(x) for x in lab_vals)

        for cand_rgb, cand_lab in zip(candidate_rgb, candidate_lab):

            lab_cand_obj = LabColor(*cand_lab, illuminant='d65')
            delta = delta_e_cie2000(lab_cluster, lab_cand_obj)

            # Lightness contrast (APCA)
            Lc = compute_apca_lc(cand_rgb, cluster_rgb_tuple)
            pass_apca = abs(Lc) >= 15

            records.append({
                "Model": model_name,
                "Cluster LAB": clean_lab_tuple,
                "Cluster RGB": cluster_rgb_tuple,
                "Cluster Weight": weights[i],
                "Crosshair RGB": cand_rgb,
                "DeltaE": float(delta),
                "APCA_Lc": Lc,
                "APCA_Pass": pass_apca
            })

    return records

# This equation is defined here in Gaussian case: https://en.wikipedia.org/wiki/Bhattacharyya_distance
def bhattacharyya_distance(mu1, cov1, mu2, cov2):

    mu1 = np.array(mu1)
    mu2 = np.array(mu2)

    cov_mean = (cov1 + cov2) / 2

    diff = mu1 - mu2
    inv_cov = np.linalg.pinv(cov_mean)

    term1 = 0.125 * diff.T @ inv_cov @ diff
    term2 = 0.5 * np.log(
        np.linalg.det(cov_mean) /
        np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2) + 1e-8)
    )

    return float(term1 + term2)

def compute_bhattacharyya_stats(gmm):
    means = gmm.means_
    covs = gmm.covariances_

    gmm_centers_rgb = []

    for c in means:
        lab_arr = np.array(c, dtype=np.float32)[np.newaxis, np.newaxis, :]
        rgb = lab2rgb(lab_arr)[0, 0]
        gmm_centers_rgb.append([
            int(np.clip(rgb[0] * 255, 0, 255)),
            int(np.clip(rgb[1] * 255, 0, 255)),
            int(np.clip(rgb[2] * 255, 0, 255))
        ])

    distances = []
    coefficients = []
    pair_records = []

    for i, j in combinations(range(gmm.n_components), 2):
        distance = bhattacharyya_distance(
            means[i], covs[i],
            means[j], covs[j]
        )

        coefficient = np.exp(-distance)

        distances.append(distance)
        coefficients.append(coefficient)

        pair_records.append({
            "Cluster_A": i,
            "Cluster_B": j,
            "Cluster_A_RGB": tuple(gmm_centers_rgb[i]),
            "Cluster_B_RGB": tuple(gmm_centers_rgb[j]),
            "Bhattacharyya_Distance": distance,
            "Bhattacharyya_Coefficient": coefficient
        })

    return (
        np.mean(distances),
        np.std(distances),
        np.min(distances),
        np.max(distances),
        np.mean(coefficients),
        np.std(coefficients),
        np.min(coefficients),
        np.max(coefficients),
        pair_records
    )

def plot_lab_clusters(sample, labels, centers_lab, output_folder, title):

    # Subsample for speed (important with millions of points)
    max_points = 50000
    if len(sample) > max_points:
        idx = np.random.choice(len(sample), max_points, replace=False)
        sample_plot = sample[idx]
        labels_plot = labels[idx]
    else:
        sample_plot = sample
        labels_plot = labels

    L = sample_plot[:, 0]
    a = sample_plot[:, 1]
    b = sample_plot[:, 2]

    # Convert cluster centres to RGB colours for visualisation
    cluster_colors = []
    for c in centers_lab:
        lab_arr = np.array(c, dtype=np.float32)[np.newaxis, np.newaxis, :]
        rgb = lab2rgb(lab_arr)[0, 0]

        cluster_colors.append(rgb)

    cluster_colors = np.array(cluster_colors)

    # Assign each pixel its cluster colour
    point_colors = cluster_colors[labels_plot]

    # 3D LAB scatter no gridlines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False

    ax.scatter(
        L, a, b,
        c=point_colors,
        s=2,
        alpha=0.2,
        zorder=1
    )

    # Cluster centres as small dots with black outline
    ax.scatter(
        centers_lab[:, 0],
        centers_lab[:, 1],
        centers_lab[:, 2],
        c=cluster_colors,
        s=60,
        edgecolors='black',  # outline
        linewidths=1.0,
        depthshade=False,  # keeps them visible
        zorder=10  # bring to front
    )

    ax.set_xlabel("L - Lightness")
    ax.set_ylabel("A - Green(-ve) / Red(+ve)")
    ax.set_zlabel("B - Blue(-ve) / Yellow(+ve)")
    ax.grid(False)
    ax.set_title(title)

    plt.savefig(output_folder / f"{title}_LAB_3D_NO_GRID.png")
    plt.show()

    # 3D LAB scatter with gridlines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False

    ax.scatter(
        L, a, b,
        c=point_colors,
        s=2,
        alpha=0.2,
        zorder=1
    )

    # Cluster centres as small dots with black outline
    ax.scatter(
        centers_lab[:, 0],
        centers_lab[:, 1],
        centers_lab[:, 2],
        c=cluster_colors,
        s=60,
        edgecolors='black',  # outline
        linewidths=1.0,
        depthshade=False,  # keeps them visible
        zorder=10  # bring to front
    )

    ax.set_xlabel("L - Lightness")
    ax.set_ylabel("A - Green(-ve) / Red(+ve)")
    ax.set_zlabel("B - Blue(-ve) / Yellow(+ve)")
    ax.set_title(title)

    plt.savefig(output_folder / f"{title}_LAB_3D.png")
    plt.show()

    # 2D projections (A vs. B) - No L in this plot
    plt.figure()

    plt.scatter(
        a, b,
        c=point_colors,
        s=2,
        alpha=0.6
    )

    plt.scatter(
        centers_lab[:, 1],
        centers_lab[:, 2],
        c=cluster_colors,
        s=60,
        edgecolors='black',
        linewidths=1.0,
        zorder=10
    )

    plt.xlabel("A - Green(-ve) / Red(+ve)")
    plt.ylabel("B - Blue(-ve) / Yellow(+ve)")
    plt.title(title + " (a vs b)")

    plt.savefig(output_folder / f"{title}_ab.png")
    plt.show()

def run_crosshair_clustering_on_folder(
        input_folder,
        roi,
        comparison_colors,
        comparison_colors_lab,
        max_k=20,
        sample_size=50000000):

    output_folder = Path("Video_Lab_Results")
    output_folder.mkdir(exist_ok=True)

    reservoir_file = output_folder / "lab_reservoir.npy"

    if reservoir_file.exists():
        log("Loading existing LAB reservoir...")
        LAB_reservoir = np.load(reservoir_file).astype(np.float32)

        # Check saved reservoir file for red artifacts
        print("\nChecking for red artifact in loaded reservoir...")

        target = np.array([53, 80, 67])
        dist = np.linalg.norm(LAB_reservoir - target, axis=1)

        matches = np.sum(dist < 1.0)

        print(f"Matching pixels: {matches}")
        print(f"Percentage: {matches / len(LAB_reservoir):.6f}")

        # Remove the red artifacts from the saved reservoir file
        LAB_reservoir = LAB_reservoir[dist > 2.0]

        print("\nAfter filtering:")
        dist_after = np.linalg.norm(LAB_reservoir - target, axis=1)
        print(f"Remaining matches: {np.sum(dist_after < 1.0)}")

    else:
        log("Collecting LAB pixels from video...")
        LAB_reservoir = collect_lab_pixels_from_folder(input_folder, roi)

        # Save reservoir here
        np.save(reservoir_file, LAB_reservoir)
        log("LAB reservoir saved.")

    perceptual_records = []
    summary_records = []
    all_bhattacharyya_records = []

    log(f"\n=== Starting Analysis ===")

    idx = np.random.choice(
        len(LAB_reservoir),
        size=min(sample_size, len(LAB_reservoir)),
        replace=False
    )

    sample = LAB_reservoir[idx]

    # Can use a smaller sample for cluster number determination if memory cannot handle large reservoir
    selection_size = min(1000000, len(sample))

    selection_idx = np.random.choice(
        len(sample),
        size=selection_size,
        replace=False
    )

    selection_sample = sample[selection_idx]

    log(f"Model selection sample size: {len(selection_sample):,}")
    log(f"Sample size: {len(sample):,}")

    # GMM
    (gmm_k, k_vals, bic_vals, bic_grad, bhat_distance_means, bhat_distance_stds, bhat_distance_mins, bhat_distance_maxs,
     bhat_coefficient_means, bhat_coefficient_stds, bhat_coefficient_mins, bhat_coefficient_maxs) = gmm_model_selection(
        selection_sample, max_k, output_folder)

    gmm = GaussianMixture(
        n_components = gmm_k,
        covariance_type ='full',
        n_init = 5,
        verbose = 1,
        random_state = 1
    )

    gmm.fit(sample)

    gmm_labels = gmm.predict(sample)

    plot_lab_clusters(
        sample,
        gmm_labels,
        gmm.means_,
        output_folder,
        f"GMM"
    )

    gmm_centers_lab = gmm.means_
    gmm_centers_rgb = []

    for c in gmm_centers_lab:
        lab_arr = np.array(c, dtype=np.float32)[np.newaxis, np.newaxis, :]
        rgb = lab2rgb(lab_arr)[0, 0]

        gmm_centers_rgb.append([
            int(np.clip(rgb[0] * 255, 0, 255)),
            int(np.clip(rgb[1] * 255, 0, 255)),
            int(np.clip(rgb[2] * 255, 0, 255))
        ])

    weights = gmm.weights_

    perceptual_records += compute_deltaE(
        gmm_centers_lab,
        gmm_centers_rgb,
        weights,
        comparison_colors,
        comparison_colors_lab,
        "GMM"
    )

    for i, k in enumerate(k_vals):
        summary_records.append({
            "Model": "GMM",
            "k": k,
            "BIC": bic_vals[i],
            "BIC_Gradient": bic_grad[i],
            "Bhat_Distance_Mean": bhat_distance_means[i],
            "Bhat_Distance_STD": bhat_distance_stds[i],
            "Bhat_Distance_Min": bhat_distance_mins[i],
            "Bhat_Distance_Max": bhat_distance_maxs[i],
            "Bhat_Coefficient_Mean": bhat_coefficient_means[i],
            "Bhat_Coefficient_STD": bhat_coefficient_stds[i],
            "Bhat_Coefficient_Min": bhat_coefficient_mins[i],
            "Bhat_Coefficient_Max": bhat_coefficient_maxs[i],
            "Selected_k": gmm_k
        })

    (final_bhat_distance_mean, final_bhat_distance_std, final_bhat_distance_min, final_bhat_distance_max,
     final_bhat_coefficient_mean, final_bhat_coefficient_std, final_bhat_coefficient_min, final_bhat_coefficient_max,
     bhat_pairs) = compute_bhattacharyya_stats(gmm)

    summary_records.append({
        "Model": "GMM_FINAL",
        "k": gmm_k,
        "BIC": gmm.bic(sample),
        "BIC_Gradient": None,
        "Bhat_Distance_Mean": final_bhat_distance_mean,
        "Bhat_Distance_STD": final_bhat_distance_std,
        "Bhat_Distance_Min": final_bhat_distance_min,
        "Bhat_Distance_Max": final_bhat_distance_max,
        "Bhat_Coefficient_Mean": final_bhat_coefficient_mean,
        "Bhat_Coefficient_STD": final_bhat_coefficient_std,
        "Bhat_Coefficient_Min": final_bhat_coefficient_min,
        "Bhat_Coefficient_Max": final_bhat_coefficient_max,
        "Selected_k": gmm_k
    })

    for pair in bhat_pairs:
        all_bhattacharyya_records.append({
            "Model": "GMM_FINAL",
            "k": gmm_k,
            "Cluster_A": pair["Cluster_A"],
            "Cluster_A_RGB": pair["Cluster_A_RGB"],
            "Cluster_B": pair["Cluster_B"],
            "Cluster_B_RGB": pair["Cluster_B_RGB"],
            "Bhattacharyya_Distance": pair["Bhattacharyya_Distance"],
            "Bhattacharyya_Coefficient": pair["Bhattacharyya_Coefficient"]
        })

    log(f"==== Ending Analysis ====")

    pd.DataFrame(perceptual_records).to_csv(
        output_folder / "perceptual_results.csv",
        index=False
    )

    pd.DataFrame(summary_records).to_csv(
        output_folder / "summary_results.csv",
        index=False
    )

    pd.DataFrame(all_bhattacharyya_records).to_csv(
        output_folder / "bhattacharyya_results.csv",
        index=False
    )

    print("\nAll processing complete.")

if __name__ == "__main__":

    input_folder = Path(__file__).resolve().parent.parent / "Video Files"

    roi = CircularROI(
        center_x=1920 // 2,
        center_y=1080 // 2,
        radius=125
    )

    # Candidate colours for perceptual comparison - various commonly used colours and variety of non-common colours
    # Additional colours could be added to this list or adapted on future pro crosshair datasets
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
        (128, 128, 128),  # Grey
        (192, 192, 192),  # Silver
        (128, 0, 0),  # Maroon
        (128, 128, 0),  # Olive
        (0, 128, 0),  # Green
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (0, 0, 128),  # Navy Blue
        (255, 165, 0),  # Orange
        (75, 0, 130),  # Indigo
        (255, 0, 150),  # Pink
        (34, 139, 34),  # Forest Green
        (180, 255, 255),  # Light Blue
        (255, 200, 0),  # Golden Yellow
        (220, 190, 255),  # Lavender
        (250, 0, 100),  # Hot Pink
    ]

    comparison_colors = pro_colors + additional_colors
    comparison_colors_lab = [rgb_to_lab(c) for c in comparison_colors]

    run_crosshair_clustering_on_folder(
        input_folder=input_folder,
        roi=roi,
        comparison_colors=comparison_colors,
        comparison_colors_lab=comparison_colors_lab,
        max_k=20
    )
