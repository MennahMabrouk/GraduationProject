from tools import perform_glm_and_plot_z_map, extract_and_plot_timeseries, plot_residuals_histogram, plot_r_squared_map, perform_ftest_and_plot_f_map

# Test each function
fmri_glm, z_map = perform_glm_and_plot_z_map()
masker, coords = extract_and_plot_timeseries(fmri_glm, z_map)
plot_residuals_histogram(masker, fmri_glm, coords)
plot_r_squared_map(fmri_glm)
perform_ftest_and_plot_f_map(fmri_glm)