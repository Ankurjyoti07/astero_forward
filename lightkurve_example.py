"""
PROGRAM NAME:
    lightkurve_example.py

DESCRIPTION:
    python wrapper for lightkurve

INPUT:
    none

LAST MODIFIED:
    6 June 2024
        - update to lightkurve v2.4.x
        - update to python 3.12.x

AUTHORS:
    Dominic Bowman (Newcastle University, UK)

"""

import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
#import peakutils

"""
# NOTES:

defaults:
threshold=8
regressors=3
image_size=25
outlier sigma=60000
"""

###############################################################################
# DOWNLOAD DATA

# define TIC number for target
TESSTIC = '102144103'

# search (remotely) for available TESS sectors in FFI data
tesscuts = lk.search_tesscut('TIC '+TESSTIC)
print(tesscuts)

# download and plot sqaure cut-out of TESS FFI (remotely)
current_sector = 1  # index for array of sectors that will be downloaded
image_size = 20
tpf = tesscuts[current_sector].download(cutout_size=image_size)
#save_sector = tesscuts.table['observation'][current_sector]  # removed: observation->mission
save_sector = tesscuts.table['mission'][current_sector]

# plot TESS-cut of FFI
fig, ax = plt.subplots(nrows=1, ncols=1)
tpf.plot(ax=ax)
ax.set_title('Target pixel file')
plt.show()


###############################################################################
# APERTURE PHOTOMETRY

# Define aperture mask using threshold method uses a sigma-above-background value of,
# e.g. 8, assuming the star is located in the center (should be)
aperture_method = 'threshold'
True_array = []
False_array = []
pick_threshold = 8
mask_aperture = tpf.create_threshold_mask(threshold=pick_threshold, reference_pixel='center')

# Define "sky" background mask (assuming threshold = 0.01)
mask_background = ~tpf.create_threshold_mask(threshold=0.01, reference_pixel=None)

# Plot annoated TPF data with apertures
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
tpf.plot(ax=ax1, aperture_mask=mask_aperture, mask_color='w')
tpf.plot(ax=ax2, aperture_mask=mask_background, mask_color='w')
ax1.set_title('Aperture mask')
ax2.set_title('Background mask')
fig.subplots_adjust()

# Save figure as diagnostic plot
sector_stripped = "_".join(save_sector[4:].split())
plt.savefig('./tess'+TESSTIC.zfill(12)+'_mask_'+sector_stripped+'.png', dpi = 300)

# Extract light curve from TPF
lc_raw = tpf.to_lightcurve(aperture_mask=mask_aperture)
lc_raw = lc_raw[lc_raw.flux_err > 0]

# Plot extracted "raw" light curve
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
lc_raw.plot(ax=ax)
ax.set_title('Raw light curve')
plt.show()


###############################################################################
# DETRENDING BY PCA

# Define Regressors to perform PCA and remove systematics
regressors = tpf.flux[:][:,mask_background]

# Define number of principal components
npcs = 20  # number to inspect

# Design regressor matrix
dm = lk.DesignMatrix(regressors, name='regressors').pca(npcs).append_constant()

# Plot first npcs components to inspect
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
#ax.plot(tpf.time, dm.values[:,:-1] + np.arange(npcs)*0.2, '.', color='k', ms=2)
mask_nan = [True if not np.isnan(i) else False for i in lc_raw.flux]
ax.plot(tpf[mask_nan].time.value, dm.values[:,:-1] + np.arange(npcs)*0.2, '.', color='k', ms=2)
ax.axes.get_yaxis().set_visible(False)
ax.set_title('The first principal component is at the bottom')
plt.savefig('./tess'+TESSTIC.zfill(12)+'_pca_regressors_'+sector_stripped+'.png', dpi = 300)
plt.show()

# Define number of principal components to use
npcs = 1  # usually a few are needed, but this depends on the star

# Design regressor matrix
dm = lk.DesignMatrix(regressors, name='regressors').pca(npcs).append_constant()

# Apply the detrending
rc = lk.RegressionCorrector(lc_raw)

# Get the detrended light curve
lc = rc.correct(dm)  # comment out if npcs=0 is needed

# plot a simple diagnostic plot
rc.diagnose()
plt.savefig('./tess'+TESSTIC.zfill(12)+'_raw_light_curve_'+sector_stripped+'.png', dpi = 300)
plt.show()


#lc=lc_raw  # uncomment if regressors npcs=0 is needed

###############################################################################
# SIGMA CLIPPING

# Apply sigma-clipping
lc_clean, mask_outliers = lc.remove_outliers(sigma=500000,return_mask=True)

# plot diagnostic light curve figures (before and after)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10,8))
ax1.plot(lc.time[mask_outliers], lc.flux[mask_outliers], marker='.', ls='None', color='red', label='Outliers')
lc.plot(ax=ax1, marker='.', ls='None')
ax1.legend(loc='best')
ax1.set_title('Detrended light curve')
lc_clean.plot(ax=ax2, marker='.', ls='None')
ax2.set_title('Detrended light curve, outliers removed (scatter plot)')
lc_clean.plot(ax=ax3)
ax3.set_title('Detrended light curve, outliers removed (line plot)')
fig.subplots_adjust(hspace=0.5)
plt.savefig('./tess'+TESSTIC.zfill(12)+'_detrended_light_curve_'+sector_stripped+'.png', dpi = 300)
plt.show()



###############################################################################
# EXPORT TO ASCII FILE

master_flux = []
master_time = []
raw_master_flux = []
raw_master_time = []

# zero-pad TIC for file name
saveTIC = TESSTIC.zfill(12)

# detrended light curve data
# Convert from flux to mag (detrended light curve)
save_flux = lc_clean.flux
save_time = lc_clean.time
save_mag = -2.5*np.log10(save_flux.to_value())
save_mag = save_mag - np.mean(save_mag)

master_time = np.append(master_time, save_time.to_value(format='btjd'))
master_flux = np.append(master_flux, save_mag)

# Check sorting
master_index_sort = np.argsort(master_time, axis = 0)
master_time = master_time[master_index_sort]
master_flux = master_flux[master_index_sort]

# remove NaNs
master_time = master_time[~np.isnan(master_flux)]
master_flux = master_flux[~np.isnan(master_flux)]

# remove the median
master_flux = master_flux - np.median(master_flux)

# Save to file
save_light_curve = np.array([master_time, master_flux])
master_file = open('./tess'+saveTIC+"_"+sector_stripped+"_LC_data.txt", 'wb')
np.savetxt(master_file, save_light_curve.T, fmt='%.10f', delimiter=',   ')
master_file.close()


# raw light curve data
# Convert from flux to mag (raw light curve)
raw_flux = lc_raw.flux
raw_time = lc_raw.time
raw_mag = -2.5*np.log10(raw_flux.to_value())

raw_master_time = np.append(raw_master_time, raw_time.to_value(format='btjd'))
raw_master_flux = np.append(raw_master_flux, raw_mag)

# Check sorting
raw_master_index_sort = np.argsort(raw_master_time, axis = 0)
raw_master_time = raw_master_time[raw_master_index_sort]
raw_master_flux = raw_master_flux[raw_master_index_sort]

# remove NaNs
raw_master_time = raw_master_time[~np.isnan(raw_master_flux)]
raw_master_flux = raw_master_flux[~np.isnan(raw_master_flux)]

# remove the median
raw_master_flux = raw_master_flux - np.median(raw_master_flux)

# save to file
raw_light_curve = np.array([raw_master_time, raw_master_flux])
raw_file = open('./tess'+saveTIC+"_"+sector_stripped+"_LC_data_raw.txt", 'wb')
np.savetxt(raw_file, raw_light_curve.T, fmt='%.10f', delimiter=',   ')
raw_file.close()
