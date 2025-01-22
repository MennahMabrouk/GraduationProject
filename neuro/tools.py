import sys
import os
from nilearn.input_data import NiftiMasker
import pandas as pd
from nilearn.maskers import NiftiSpheresMasker
from nilearn.reporting import get_clusters_table
from nilearn import image, masking
from nilearn.datasets import fetch_spm_auditory
from nilearn.plotting import plot_stat_map, show
from nilearn.glm.first_level import FirstLevelModel
import matplotlib.pyplot as plt

# Redirect stderr to suppress macOS IMK logs
sys.stderr = open(os.devnull, 'w')

# Function to load fMRI data
def load_fmri_data():
    try:
        subject_data = fetch_spm_auditory()
        fmri_img = subject_data.func[0]
        mean_img = image.mean_img(fmri_img, copy_header=True)
        mask = masking.compute_epi_mask(mean_img)
        fmri_img = image.clean_img(fmri_img, standardize=False)
        fmri_img = image.smooth_img(fmri_img, 5.0)
        events = pd.read_csv(subject_data.events, sep="\t")
        return fmri_img, mean_img, mask, events
    except Exception as e:
        print(f"Error loading data: {e}")
        raise  # Re-raise the exception to stop execution if data loading fails

# Button 1: Perform GLM and plot Z-map
def perform_glm_and_plot_z_map():
    try:
        fmri_img, mean_img, mask, events = load_fmri_data()
        fmri_glm = FirstLevelModel(
            t_r=7,
            drift_model="cosine",
            signal_scaling=False,
            mask_img=mask,
            minimize_memory=False,
        )
        fmri_glm = fmri_glm.fit(fmri_img, events)
        z_map = fmri_glm.compute_contrast("listening")
        threshold = 3.1
        plot_stat_map(
            z_map,
            bg_img=mean_img,
            threshold=threshold,
            title=f"listening > rest (t-test; |Z|>{threshold})",
        )
        show()
    except Exception as e:
        print(f"Error in perform_glm_and_plot_z_map: {e}")

# Button 2: Extract and plot timeseries for clusters
def extract_and_plot_timeseries(threshold=3.1):
    try:
        fmri_img, mean_img, mask, events = load_fmri_data()
        fmri_glm = FirstLevelModel(
            t_r=7,
            drift_model="cosine",
            signal_scaling=False,
            mask_img=mask,
            minimize_memory=False,
        )
        fmri_glm = fmri_glm.fit(fmri_img, events)
        z_map = fmri_glm.compute_contrast("listening")

        table = get_clusters_table(z_map, stat_threshold=threshold, cluster_threshold=20)
        table.set_index("Cluster ID", drop=True)
        coords = table.loc[range(1, 7), ["X", "Y", "Z"]].to_numpy()
        masker = NiftiSpheresMasker(coords)
        real_timeseries = masker.fit_transform(fmri_img)
        predicted_timeseries = masker.fit_transform(fmri_glm.predicted[0])

        colors = ["blue", "navy", "purple", "magenta", "olive", "teal"]
        fig1, axs1 = plt.subplots(2, 6)
        for i in range(6):
            axs1[0, i].set_title(f"Cluster peak {coords[i]}\n")
            axs1[0, i].plot(real_timeseries[:, i], c=colors[i], lw=2)
            axs1[0, i].plot(predicted_timeseries[:, i], c="r", ls="--", lw=2)
            axs1[0, i].set_xlabel("Time")
            axs1[0, i].set_ylabel("Signal intensity", labelpad=0)
            roi_img = plot_stat_map(
                z_map,
                cut_coords=[coords[i][2]],
                threshold=3.1,
                figure=fig1,
                axes=axs1[1, i],
                display_mode="z",
                colorbar=False,
                bg_img=mean_img,
            )
            roi_img.add_markers([coords[i]], colors[i], 300)
        fig1.set_size_inches(24, 14)
        show()
    except Exception as e:
        print(f"Error in extract_and_plot_timeseries: {e}")

# Button 3: Plot residuals histogram
def plot_residuals_histogram():
    try:
        fmri_img, mean_img, mask, events = load_fmri_data()
        fmri_glm = FirstLevelModel(
            t_r=7,
            drift_model="cosine",
            signal_scaling=False,
            mask_img=mask,
            minimize_memory=False,
        )
        fmri_glm = fmri_glm.fit(fmri_img, events)
        z_map = fmri_glm.compute_contrast("listening")

        table = get_clusters_table(z_map, stat_threshold=3.1, cluster_threshold=20)
        table.set_index("Cluster ID", drop=True)
        coords = table.loc[range(1, 7), ["X", "Y", "Z"]].to_numpy()
        masker = NiftiSpheresMasker(coords)
        resid = masker.fit_transform(fmri_glm.residuals[0])

        fig2, axs2 = plt.subplots(2, 3, constrained_layout=True)
        axs2 = axs2.flatten()
        colors = ["blue", "navy", "purple", "magenta", "olive", "teal"]
        for i in range(6):
            axs2[i].set_title(f"Cluster peak {coords[i]}\n")
            axs2[i].hist(resid[:, i], color=colors[i])
            print(f"Mean residuals: {resid[:, i].mean()}")
        fig2.set_size_inches(12, 7)
        show()
    except Exception as e:
        print(f"Error in plot_residuals_histogram: {e}")

# Button 5: Plot R-squared map
def plot_r_squared_map():
    try:
        fmri_img, mean_img, mask, events = load_fmri_data()
        fmri_glm = FirstLevelModel(
            t_r=7,
            drift_model="cosine",
            signal_scaling=False,
            mask_img=mask,
            minimize_memory=False,
        )
        fmri_glm = fmri_glm.fit(fmri_img, events)

        plot_stat_map(
            fmri_glm.r_square[0],
            bg_img=mean_img,
            threshold=0.1,
            display_mode="z",
            cut_coords=7,
            cmap="inferno",
            title="R-squared",
            vmin=0,
            symmetric_cbar=False,
        )
        show()
    except Exception as e:
        print(f"Error in plot_r_squared_map: {e}")

# Button 6: Perform F-test and plot F-map
def perform_ftest_and_plot_f_map(threshold=3.1):
    try:
        fmri_img, mean_img, mask, events = load_fmri_data()
        fmri_glm = FirstLevelModel(
            t_r=7,
            drift_model="cosine",
            signal_scaling=False,
            mask_img=mask,
            minimize_memory=False,
        )
        fmri_glm = fmri_glm.fit(fmri_img, events)

        z_map_ftest = fmri_glm.compute_contrast(
            "listening", stat_type="F", output_type="z_score"
        )
        plot_stat_map(
            z_map_ftest,
            bg_img=mean_img,
            threshold=threshold,
            display_mode="z",
            cut_coords=7,
            cmap="inferno",
            title=f"listening > rest (F-test; Z>{threshold})",
            symmetric_cbar=False,
            vmin=0,
        )
        show()
    except Exception as e:
        print(f"Error in perform_ftest_and_plot_f_map: {e}")