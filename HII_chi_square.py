import numpy as np
import matplotlib.pyplot as plt
import emcee
import py21cmfast as p21c
from scipy.fft import fftshift, fft2
from tqdm import tqdm  # Import tqdm for the progress bar
import corner  # Import corner for corner plot visualization
import getdist.plots as plots
import scipy.signal as psg
from astropy.stats import mad_std
from getdist import MCSamples, plots
from getdist.mcsamples import MCSamples
from getdist.plots import get_subplot_plotter
from statsmodels.stats.moment_helpers import cov2corr
from multiprocessing import Pool, cpu_count  # Import for parallelization
import healpy as hp
from scipy.interpolate import interp1d
from scipy.optimize import brentq 

####### GENERATE BRIGHTNESS TEMPERATURE MAP #########

# Function to simulate a 21cmFAST map for a given cosmology and redshift range
def generate_21cm_map(cosmo_params, z_low, z_high, box_size=100, box_dim=64, seed=None, HII_eff_factor=None):

    # Define cosmological parameters
    cosmo_params_obj = p21c.CosmoParams(
        OMm=cosmo_params['Omega_c'] + cosmo_params['Omega_b'],
        OMb=cosmo_params['Omega_b']
    )
    
    # Define user parameters
    user_params_obj = p21c.UserParams(
        HII_DIM=box_dim,
        BOX_LEN=box_size
    )
    
    # Define astrophysical parameters (is passed here)
    astro_params_obj = p21c.AstroParams(
        HII_EFF_FACTOR=HII_eff_factor if HII_eff_factor is not None else 5 # Default to 1e8 if M_min is not provided
    )
    
    # Run the lightcone generation with the specified random seed
    lightcone = p21c.run_lightcone(
        redshift=z_low,
        max_redshift=z_high,
        cosmo_params=cosmo_params_obj,
        user_params=user_params_obj,
        astro_params=astro_params_obj,  # Include astro params with M_min
        random_seed=seed,  # Explicitly set the seed
        USE_INTERPOLATION_TABLES=True  # Suppress warnings
    )

    # Extract the 21cm brightness temperature lightcone
    brightness_temp = lightcone.brightness_temp  # Shape: (number_of_redshifts, box_dim, box_dim)
    
    # Return the middle slice (customize as needed)
    return brightness_temp[:, :, box_dim // 2]


################ MOCK DATA GENERATION ###############

# True cosmology parameters (these would be the "true" values you want to simulate with)
# Define true cosmology parameters
true_cosmo_params = {
    'Omega_c': 0.26,  # Cold dark matter density
    'Omega_b': 0.048  # Baryonic matter density
}

true_HII_eff_factor = 5

# Generate the map with M_min set to a specific value
true_map_data = generate_21cm_map(true_cosmo_params, z_low=7.0, z_high=10.0, seed=1, HII_eff_factor=15)

# Save the generated 21cm map data as a .npy file
np.save('21cm_map_mock_data_HII_eff_factor.npy', true_map_data)
#true_map_data = np.load('21cm_map_mock_data_HII_eff_factor.npy', allow_pickle=True)

############### N MAPS WITH SAME COSMOLOGY BUT DIFFERENT RANDOM SEEDS ################

from concurrent.futures import ProcessPoolExecutor

def generate_map_with_seed(cosmo_params, z_low, z_high, box_size, box_dim, seed, HII_eff_factor):
    """
    Helper function to generate a single map given a seed, which can be used for parallel processing.
    
    Args:
        cosmo_params (dict): Cosmological parameters.
        z_low (float): Low redshift.
        z_high (float): High redshift.
        box_size (float): Size of the simulation box (in Mpc).
        box_dim (int): Dimensions of the simulation box (pixels).
        seed (int): Random seed for map generation.
        HII_eff_factor (float or None): Max radius of bubbles to be generated.
        
    Returns:
        np.array: Generated 21cm map.
    """
    try:
        # Generate the map with the current seed and HII_eff_factor
        return generate_21cm_map(cosmo_params, z_low, z_high, box_size, box_dim, seed=seed, HII_eff_factor=HII_eff_factor)
    except ValueError as e:
        print(f"Error while generating map with seed {seed}: {e}")
        return None  # Return None in case of error


def generate_multiple_maps(cosmo_params, z_low, z_high, box_size=100, box_dim=64, n_maps=10000, start_seed=1, HII_eff_factor=None, n_cpus=40):
    """
    Generate multiple maps with different seeds and varying M_min values in parallel using `n_cpus` CPUs.
    
    Args:
        cosmo_params (dict): Cosmological parameters.
        z_low (float): Low redshift.
        z_high (float): High redshift.
        box_size (float): Size of the simulation box (in Mpc).
        box_dim (int): Dimensions of the simulation box (pixels).
        n_maps (int): Number of maps to generate.
        start_seed (int): Starting random seed.
        HII_eff_factor (float or None): Max radius of bubbles to be generated.
        n_cpus (int): Number of CPU cores to use for parallelization (default is 40).
        
    Returns:
        np.array: Array of generated 21cm maps.
    """
    # Initialize a list to store the maps
    maps = []

    # Create a ProcessPoolExecutor with the specified number of CPUs (n_cpus)
    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        # Generate a list of futures for each map generation task
        futures = [executor.submit(generate_map_with_seed, cosmo_params, z_low, z_high, box_size, box_dim, start_seed + i, HII_eff_factor) for i in range(n_maps)]
        
        # Wait for all futures to complete and collect results
        for future in futures:
            result = future.result()  # This will block until the result is available
            if result is not None:
                maps.append(result)

    return np.array(maps)  # Convert to a NumPy array for easier manipulation


# Example usage: Generate 10 maps starting from seed 1 with varying HII_eff_factor, using 40 CPUs
maps = generate_multiple_maps(true_cosmo_params, z_low=7.0, z_high=10.0, n_maps=10000, start_seed=1, HII_eff_factor=None, n_cpus=45)
# Save all the maps in a single file
np.save('21cm_maps_N_seeds_fixed_HII_eff_factor.npy', maps)
#maps = np.load('21cm_maps_N_seeds_fixed_HII_eff_factor.npy', allow_pickle=True)



########### SET OF SIMMUALTION WITH VARIATION OF A PARAMETER ##############
HII_eff_factor_min = 1
HII_eff_factor_max = 10



# Number of maps and range for M_min
n_maps = 100
HII_eff_factor_values = np.linspace(HII_eff_factor_min, HII_eff_factor_max, n_maps)
seed = 1  # Fixed seed

def generate_map_for_R_bubble(HII_eff_factor, true_cosmo_params, z_low, z_high, box_size, box_dim, seed):
    """
    Helper function to generate a single map given an HII_eff_factor value, which can be used for parallel processing.
    
    Args:
        HII_eff_factor (float): Maximum bubble radius for the map.
        true_cosmo_params (dict): Cosmological parameters.
        z_low (float): Low redshift.
        z_high (float): High redshift.
        box_size (float): Size of the simulation box (in Mpc).
        box_dim (int): Dimensions of the simulation box (pixels).
        seed (int): Random seed for map generation.
        
    Returns:
        np.array: Generated 21cm map.
    """
    try:
        # Generate the map with the specified HII_eff_factor
        return generate_21cm_map(
            true_cosmo_params,
            z_low=z_low,
            z_high=z_high,
            box_size=box_size,
            box_dim=box_dim,
            seed=seed,
            HII_eff_factor=HII_eff_factor
        )
    except ValueError as e:
        print(f"Error while generating map with HII_eff_factor={HII_eff_factor:.2e}: {e}")
        return None  # Return None in case of error


def generate_maps_parallel(HII_eff_factor_values, true_cosmo_params, z_low=7.0, z_high=10.0, box_size=100, box_dim=64, seed=1, n_cpus=40):
    """
    Generate multiple maps for each value of HII_eff_factor in parallel.
    
    Args:
        HII_eff_factor_values (list of float): Values of HII_eff_factor to use for map generation.
        true_cosmo_params (dict): Cosmological parameters.
        z_low (float): Low redshift.
        z_high (float): High redshift.
        box_size (float): Size of the simulation box (in Mpc).
        box_dim (int): Dimensions of the simulation box (pixels).
        seed (int): Random seed for all maps.
        n_cpus (int): Number of CPU cores to use for parallelization (default is 40).
        
    Returns:
        np.array: Array of generated 21cm maps.
    """
    generated_maps = []

    # Use ProcessPoolExecutor for parallel map generation
    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        # Submit tasks for each HII_eff_factor value
        futures = [executor.submit(generate_map_for_R_bubble, HII_eff_factor, true_cosmo_params, z_low, z_high, box_size, box_dim, seed) for HII_eff_factor in HII_eff_factor_values]
        
        # Collect results as they complete
        for future in futures:
            result = future.result()  # This will block until the result is available
            if result is not None:
                generated_maps.append(result)

    return np.array(generated_maps)  # Convert to a NumPy array for convenience


# Convert to NumPy array for convenience
generated_maps = generate_maps_parallel(
    HII_eff_factor_values,
    true_cosmo_params=true_cosmo_params,
    z_low=7.0,
    z_high=10.0,
    box_size=100,
    box_dim=64,
    seed=seed,
    n_cpus=40
)

# Save the generated maps and the corresponding M_min values in a single file
np.savez('generated_21cm_maps_with_fixed_seed_HII_eff_factor.npz', maps=generated_maps, HII_eff_factor_values=HII_eff_factor_values)

print("Data saved to 'generated_21cm_maps_with_fixed_seed_HII_eff_factor.npz'.")
#generated_maps, HII_eff_factor_values = np.load('generated_21cm_maps_with_fixed_seed_HII_eff_factor.npz', allow_pickle=True)

################# EXTRACTION OF THE L1-NORM #################

def b3spline_fast(step_hole):
    c1 = 1./16
    c2 = 1./4
    c3 = 3./8
    length = 4*step_hole+1
    kernel1d = np.zeros((1,length))
    kernel1d[0,0] = c1
    kernel1d[0,-1] = c1
    kernel1d[0,step_hole] = c2
    kernel1d[0,-1-step_hole] = c2
    kernel1d[0,2*step_hole] = c3
    kernel2d = np.dot(kernel1d.T,kernel1d)
    return kernel2d


def star2d(im,scale,fast = True,gen2=True,normalization=False):
    (nx,ny) = np.shape(im)
    nz = scale
    
    # Normalized transfromation
    head = 'star2d_gen2' if gen2 else 'star2d_gen1'
    trans = 1 if gen2 else 2
    wt = np.zeros((nz,nx,ny))
    step_hole = 1
    im_in = np.copy(im)
    
    for i in np.arange(nz-1):
        if fast:
            kernel2d = b3spline_fast(step_hole)
            im_out = psg.convolve2d(im_in, kernel2d, boundary='symm',mode='same')
        else:
            im_out = b3splineTrans(im_in,step_hole)
            
        if gen2:
            if fast:
                im_aux = psg.convolve2d(im_out, kernel2d, boundary='symm',mode='same')
            else:
                im_aux = b3splineTrans(im_out,step_hole)
            wt[i,:,:] = im_in - im_aux
        else:        
            wt[i,:,:] = im_in - im_out
            
        if normalization:
            wt[i,:,:] /= wavtl.trTab[i]
        im_in = np.copy(im_out)
        step_hole *= 2
        
    wt[nz-1,:,:] = np.copy(im_out)

    
    return wt


def noise_coeff(image, nscales=5):
    noise_sigma = np.random.randn(image.shape[0], image.shape[0])
    noise_wavelet = star2d(noise_sigma, nscales, fast = True, gen2=False, normalization=False)
    coeff_j = np.array([np.std(scale) for scale in noise_wavelet])
    return coeff_j

def get_l1norm(image, noise, nscales=5, nbins=50):

    # add noise to noiseless image
    image_noisy = image #+ noise
    image_starlet = star2d(image_noisy, nscales, fast = True, gen2=False, normalization=False)
    noise_estimate = mad_std(image_noisy)
    #coeff_j = noise_coeff(image, nscales)
    l1_coll = []
    bins_coll = []
    for image_j in image_starlet:

        sigma_j = noise_estimate

        snr = image_j#/ sigma_j
        thresholds_snr = np.linspace(np.min(snr), np.max(snr), nbins + 1)
        bins_snr = 0.5 * (thresholds_snr[:-1] + thresholds_snr[1:])
        digitized = np.digitize(snr, thresholds_snr)
        bin_l1_norm = [np.sum(np.abs(snr[digitized == i])) for i in range(1, len(thresholds_snr))]
        l1_coll.append(bin_l1_norm)
        
        #bins_coll.append(bins_snr)
        #counts, bins = np.histogram(snr, bins=thresholds_snr)
        #l1_coll.append(counts)
        #bins = 0.5 * (bins_snr[1:] + bins_snr[:-1])
        bins_coll.append(bins_snr)
    return np.array(bins_coll), np.array(l1_coll)


#____________________ANGULAR POWER SPECTRUM CALCULATION__________________#

# Function to generate an angular power spectrum from a 21cmFAST map
def calculate_angular_power_spectrum(map_2d, nside=256):
    """
    Calculate the angular power spectrum (C_l) from a 2D map.
    
    Parameters:
    - map_2d: 2D numpy array (brightness temperature map).
    - nside: HEALPix nside parameter (determines resolution).
    
    Returns:
    - ell: Multipole moments (angular scales).
    - Cl: Angular power spectrum (C_l values).
    """
    # Normalize the map to have zero mean and unit variance
    map_2d_normalized = (map_2d - np.mean(map_2d)) / np.std(map_2d)
    
    # Determine angular coordinates for the map
    nx, ny = map_2d.shape
    theta = np.linspace(0, np.pi, nx)  # 0 to π for polar angle
    phi = np.linspace(0, 2 * np.pi, ny)  # 0 to 2π for azimuthal angle
    theta, phi = np.meshgrid(theta, phi, indexing='ij')
    
    # Flatten the theta and phi arrays for HEALPix mapping
    theta_flat = theta.flatten()
    phi_flat = phi.flatten()
    
    # Map 2D grid to HEALPix spherical grid
    npix = hp.nside2npix(nside)
    healpix_map = np.zeros(npix)
    indices = hp.ang2pix(nside, theta_flat, phi_flat)
    healpix_map[indices] = map_2d_normalized.flatten()
    
    # Calculate the angular power spectrum
    Cl = hp.anafast(healpix_map)
    ell = np.arange(len(Cl))
    
    return ell, Cl



#-----------------FOR THE MOCK DATA-----------------------#
mock_noise = noise_coeff(true_map_data)  # Generate noise
bins, mock_l1_norm = get_l1norm(true_map_data, mock_noise)
np.save('mock_l1_norm_HII_eff_factor.npy', mock_l1_norm)

ell, mock_Cell = calculate_angular_power_spectrum(true_map_data)
np.save('mock_Cell_HII_eff_factor.npy', mock_Cell)

# Create a figure and axis
plt.figure(figsize=(10, 6))

nscales=5
# Plot each scale's L1 norm as a function of the bins
for scale_idx in range(nscales):
    plt.plot(bins[scale_idx], mock_l1_norm[scale_idx], label=f"Scale {scale_idx + 1}")

# Customize the plot
plt.title("L1 Norm vs SNR Bins (for each scale)")
plt.xlabel("SNR Bins")
plt.ylabel("L1 Norm")
plt.legend(loc="upper right")
plt.grid(True)
# Save the plot as a PNG file
plt.savefig('mock_l1_norm_plot_HII_eff_factor.pdf')



# Create a figure and axis
plt.figure(figsize=(10, 6))
plt.plot(ell, mock_Cell)
# Customize the plot
plt.title("mock Cell")
plt.xlabel("Cell")
plt.ylabel("ell")
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
# Save the plot as a PNG file
plt.savefig('mock_Cell_plot_HII_eff_factor.pdf')

#________________FOR THE IDENTICAL SIMS_____________________#
###_______FOR L1 NORM__________###
# Placeholder for L1 norm results across different seeds
l1_norms_all_seeds = []
all_l1_norms = []
# Loop through each map generated from different seeds
for i, map_data in enumerate(maps):
    print(f"Processing map {i + 1} for l1 norm...")
    # Generate noise for the current map
    noise = noise_coeff(map_data)
    
    # Compute the L1 norm for the current map and its corresponding noise
    bins, l1_norm = get_l1norm(map_data, noise)
    
    # Store the L1 norm for the current map
    l1_norms_all_seeds.append(l1_norm.flatten())
    # Append the L1 norm to the list
    all_l1_norms.append(l1_norm)

# Convert L1 norms to a NumPy array
l1_norms_all_seeds = np.array(l1_norms_all_seeds)  # Shape: (n_seeds, n_bins_per_scale)

# Estimate the covariance matrix
data_covariance_l1_norm = np.cov(l1_norms_all_seeds, rowvar=False)

# Convert the list of L1 norms to a NumPy array
all_l1_norms = np.array(all_l1_norms)

# Save the array to a file
np.save('all_l1_norms_HII_eff_factor.npy', all_l1_norms)
print("All L1 norms saved successfully to 'all_l1_norms_HII_eff_factor.npy'.")

# Save the covariance matrix and bins for later use
np.save('data_covariance_HII_eff_factor.npy', data_covariance_l1_norm)
np.save('l1_norms_all_seeds_HII_eff_factor.npy', l1_norms_all_seeds) 

print("Data covariance matrix saved successfully.")
print(f"Shape of mock_l1_norm: {mock_l1_norm.shape}")
print(f"Shape of l1_norms_all_seeds: {np.array(l1_norms_all_seeds).shape}")

###_______FOR CELL__________###


# Placeholder for L1 norm results across different seeds
Cell_all_seeds = []
all_Cell = []
# Loop through each map generated from different seeds
for i, map_data in enumerate(maps):
    print(f"Processing map {i + 1} for Cell...")
    
    # Compute the L1 norm for the current map and its corresponding noise
    ell, Cell = calculate_angular_power_spectrum(map_data)
    
    # Store the L1 norm for the current map
    Cell_all_seeds.append(Cell.flatten())
    # Append the L1 norm to the list
    all_Cell.append(Cell)

# Convert L1 norms to a NumPy array
Cell_all_seeds = np.array(Cell_all_seeds)  # Shape: (n_seeds, n_bins_per_scale)

# Estimate the covariance matrix
data_covariance_Cell = np.cov(Cell_all_seeds, rowvar=False)

# Convert the list of L1 norms to a NumPy array
all_Cell = np.array(all_Cell)


# Save the array to a file
np.save('all_Cells_HII_eff_factor.npy', all_Cell)
print("All Cell saved successfully to 'all_Cell_HII_eff_factor.npy'.")

# Save the covariance matrix and bins for later use
np.save('data_covariance_Cell_HII_eff_factor.npy', data_covariance_Cell)
np.save('Cell_all_seeds_HII_eff_factor.npy', Cell_all_seeds) 

print("Data covariance matrix for Cell saved successfully.")
print(f"Shape of mock_Cell: {mock_Cell.shape}")
print(f"Shape of Cell_all_seeds: {np.array(Cell_all_seeds).shape}")

#________________FOR THE N SIMS_____________________#


###_______FOR L1 NORM__________###
# Placeholder for L1 norm results across simulations with varying M_min
l1_norms_all_HII_eff_factor = []

# Loop through the simulations generated with varying M_min
for i, (map_data, HII_eff_factor) in enumerate(zip(generated_maps, HII_eff_factor_values)):
    print(f"Processing map {i + 1} with HII_eff_factor = {HII_eff_factor}...")

    
    # Generate noise for the current map
    noise = noise_coeff(map_data)
    
    # Compute the L1 norm for the current map and its corresponding noise
    bins, l1_norm = get_l1norm(map_data, noise)
    
    # Append the L1 norm and bins for the current simulation
    l1_norms_all_HII_eff_factor.append({
        "HII_eff_factor": HII_eff_factor,
        "bins": bins,
        "l1_norm": l1_norm
    })

# Save the results
np.save('l1_norms_varying_HII_eff_factor.npy', l1_norms_all_HII_eff_factor)

print("L1 norms for varying HII_eff_factor saved successfully.")

###_______FOR CELL__________###

# Placeholder for L1 norm results across simulations with varying M_min
Cell_all_HII_eff_factor = []

# Loop through the simulations generated with varying M_min
for i, (map_data, HII_eff_factor) in enumerate(zip(generated_maps, HII_eff_factor_values)):
    print(f"Processing map {i + 1} with HII_eff_factor = {HII_eff_factor}...")

    
    # Compute the L1 norm for the current map and its corresponding noise
    ell, Cell = calculate_angular_power_spectrum(map_data)
    
    # Append the L1 norm and bins for the current simulation
    Cell_all_HII_eff_factor.append({
        "HII_eff_factor": HII_eff_factor,
        "ell": ell,
        "Cell": Cell
    })

# Save the results
np.save('Cell_varying_HII_eff_factor.npy', Cell_all_HII_eff_factor)

print("Cell for varying HII_eff_factor saved successfully.")

#_____________COMPUTE THE CHI SQUARED_________________#

def calculate_noise(cov_matrix):
    # Ensure input is a numpy array
    cov_matrix = np.array(cov_matrix)
    
    # Extract variances (diagonal elements)
    variances = np.diag(cov_matrix)
    
    # Calculate standard deviations (noise)
    noise = np.sqrt(variances)
    
    return noise

sigma_l1_norm = calculate_noise(data_covariance_l1_norm)
sigma_Cell = calculate_noise(data_covariance_Cell)

def compute_chi_square(observed, expected, uncertainties):
    # Ensure inputs are numpy arrays for element-wise operations
    O = np.array(observed)
    E = np.array(expected)
    sigma = np.array(uncertainties)

    # Calculate chi-square
    chi_square = np.sum((O - E)**2 / sigma**2)
    return chi_square

chi_square_all_l1_norm = []
chi_square_all_Cell = []

for l1_data in l1_norms_all_HII_eff_factor:
    expected = l1_data["l1_norm"]  # Extract the L1 norm from the dictionary
    chi_square = compute_chi_square(mock_l1_norm.flatten(), expected.flatten(), sigma_l1_norm)
    chi_square_all_l1_norm.append(chi_square)

for Cell_data in Cell_all_HII_eff_factor:
    expected = Cell_data["Cell"]  # Extract the Cell from the dictionary
    chi_square = compute_chi_square(mock_Cell.flatten(), expected.flatten(), sigma_Cell)
    chi_square_all_Cell.append(chi_square)


plt.figure(figsize=(10, 6))
plt.plot(HII_eff_factor_values, chi_square_all_l1_norm)
plt.title('Chi square as a function of the parameter for l1 norm')
plt.xlabel('Chi square')
plt.ylabel('R bubble max')
plt.grid(True)
plt.savefig('chi_square_l1_norm_HII_eff_factor.pdf')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(HII_eff_factor_values, chi_square_all_Cell)
plt.title('Chi square as a function of the parameter for C ell')
plt.xlabel('Chi square')
plt.ylabel('R bubble max')
plt.grid(True)
plt.savefig('chi_square_C_ell_HII_eff_factor.pdf')
plt.show()

#_____________________INTERPOLATE VBETWEEN THE VALUES OF CHI SQUARE______________________#
# Generate a finer range of HII_eff_factor values for interpolation
HII_eff_factor_fine = np.linspace(HII_eff_factor_min, HII_eff_factor_max, 500)


###_______FOR L1_NORM__________###

# Interpolate chi-square values
interpolation_function_l1_norm = interp1d(
    HII_eff_factor_values, 
    chi_square_all_l1_norm, 
    kind='cubic',  # Use 'linear' or 'cubic' based on desired smoothness
    fill_value="extrapolate"  # Allows extrapolation if needed
)


# Compute interpolated chi-square values
chi_square_interpolated_l1_norm = interpolation_function_l1_norm(HII_eff_factor_fine)

# Plot the interpolated chi-square values
plt.figure(figsize=(10, 6))
plt.plot(HII_eff_factor_values, chi_square_all_l1_norm, 'o', label="Computed \(\chi^2\)")
plt.plot(HII_eff_factor_fine, chi_square_interpolated_l1_norm, '-', label="Interpolated \(\chi^2\)")
plt.title('Chi-square as a function of HII_eff_factor for l1_norm (Interpolated)')
plt.xlabel('HII_eff_factor')
plt.ylabel('Chi-square')
plt.legend()
plt.grid(True)
plt.savefig('chi_square_interpolated_l1_norm_HII_eff_factor.pdf')
plt.show()


# Find the minimum chi-square and corresponding HII_eff_factor
min_chi_square_l1_norm = np.min(chi_square_interpolated_l1_norm)
optimal_HII_eff_factor_l1_norm = HII_eff_factor_fine[np.argmin(chi_square_interpolated_l1_norm)]

print(f"Minimum chi-square for l1 norm: {min_chi_square_l1_norm:.4f}")
print(f"Optimal HII_eff_factor for l1 norm: {optimal_HII_eff_factor_l1_norm:.4f}")

# Prepare data for saving
interpolated_data_l1_norm = np.column_stack((HII_eff_factor_fine, chi_square_interpolated_l1_norm))


# Save as a binary NumPy file
np.save('interpolated_chi_square_l1_norm_HII_eff_factor.npy', interpolated_data_l1_norm)

print("Interpolated chi-square values saved successfully for l1_norm.")




###_______FOR CELL__________###



# Interpolate chi-square values
interpolation_function_Cell = interp1d(
    HII_eff_factor_values, 
    chi_square_all_Cell, 
    kind='cubic',  # Use 'linear' or 'cubic' based on desired smoothness
    fill_value="extrapolate"  # Allows extrapolation if needed
)


# Compute interpolated chi-square values
chi_square_interpolated_Cell = interpolation_function_Cell(HII_eff_factor_fine)

# Plot the interpolated chi-square values
plt.figure(figsize=(10, 6))
plt.plot(HII_eff_factor_values, chi_square_all_Cell, 'o', label="Computed \(\chi^2\)")
plt.plot(HII_eff_factor_fine, chi_square_interpolated_Cell, '-', label="Interpolated \(\chi^2\)")
plt.title('Chi-square as a function of HII_eff_factor for Cell (Interpolated)')
plt.xlabel('HII_eff_factor')
plt.ylabel('Chi-square')
plt.legend()
plt.grid(True)
plt.savefig('chi_square_interpolated_Cell_HII_eff_factor.pdf')
plt.show()


# Find the minimum chi-square and corresponding HII_eff_factor
min_chi_square_Cell = np.min(chi_square_interpolated_Cell)
optimal_HII_eff_factor_Cell = HII_eff_factor_fine[np.argmin(chi_square_interpolated_Cell)]

print(f"Minimum chi-square for Cell: {min_chi_square_Cell:.4f}")
print(f"Optimal HII_eff_factor for Cell: {optimal_HII_eff_factor_Cell:.4f}")

# Prepare data for saving
interpolated_data_Cell = np.column_stack((HII_eff_factor_fine, chi_square_interpolated_Cell))


# Save as a binary NumPy file
np.save('interpolated_chi_square_Cell_HII_eff_factor.npy', interpolated_data_Cell)

print("Interpolated chi-square values saved successfully for Cell.")

#_____________________________68.5% confidence interval on HII_eff_factor____________________________#
###_____________FOR L1 NORM_____________###

# Step 1: Find minimum chi-square
chi2_min_l1_norm = np.min(chi_square_interpolated_l1_norm)

# Step 2: Calculate Delta chi-square
delta_chi2_l1_norm = chi_square_interpolated_l1_norm - chi2_min_l1_norm

# Step 3: Identify HII_eff_factor bounds for Delta chi-square = 1
# Define a function to find the roots
def delta_chi2_eq(R, delta_chi2, HII_eff_factor_fine):
    interp_func = interp1d(HII_eff_factor_fine, delta_chi2, kind='linear', bounds_error=False, fill_value="extrapolate")
    return interp_func(R) - 1

# Find the lower and upper bounds using a root-finding algorithm
HII_eff_factor_lower = brentq(delta_chi2_eq, HII_eff_factor_fine[0], HII_eff_factor_fine[np.argmin(delta_chi2_l1_norm)], args=(delta_chi2_l1_norm, HII_eff_factor_fine))
HII_eff_factor_upper = brentq(delta_chi2_eq, HII_eff_factor_fine[np.argmin(delta_chi2_l1_norm)], HII_eff_factor_fine[-1], args=(delta_chi2_l1_norm, HII_eff_factor_fine))

# Combine the results into the confidence interval
HII_eff_factor_conf_interval_l1_norm = [HII_eff_factor_lower, HII_eff_factor_upper]

print(f"68.5% confidence interval on HII_eff_factor for l1_norm: {HII_eff_factor_conf_interval_l1_norm}")

# Save the confidence interval in a .npy file
np.save('confidence_interval_HII_eff_factor_l1_norm.npy', HII_eff_factor_conf_interval_l1_norm)

# Step 4: Plotting
plt.figure(figsize=(10, 6))

# Plot chi-square as a function of HII_eff_factor
plt.plot(HII_eff_factor_fine, chi_square_interpolated_l1_norm, label=r'$\chi^2$', color='blue')

# Highlight the confidence interval
for r in HII_eff_factor_conf_interval_l1_norm:
    plt.axvline(r, color='red', linestyle='--', label=f'68.5% CI: R = {r:.2f}')

# Annotate minimum chi-square
plt.axhline(chi2_min_l1_norm, color='green', linestyle=':', label=r'$\chi^2_{\text{min}}$')

# Customize plot
plt.title(r'$\chi^2$ as a Function of $R_{\text{bubble max}}$ for l1 norm', fontsize=14)
plt.xlabel(r'$R_{\text{bubble max}}$', fontsize=12)
plt.ylabel(r'$\chi^2$', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)

# Save and show the plot
plt.savefig('chi_square_with_confidence_interval_l1_norm_HII_eff_factor.pdf')
plt.show()

###_____________FOR CELL_____________###

# Step 1: Find minimum chi-square
chi2_min_Cell = np.min(chi_square_interpolated_Cell)

# Step 2: Calculate Delta chi-square
delta_chi2_Cell = chi_square_interpolated_Cell - chi2_min_Cell

# Step 3: Identify HII_eff_factor bounds for Delta chi-square = 1
# Define a function to find the roots
def delta_chi2_eq(R, delta_chi2, HII_eff_factor_fine):
    interp_func = interp1d(HII_eff_factor_fine, delta_chi2, kind='linear', bounds_error=False, fill_value="extrapolate")
    return interp_func(R) - 1

# Find the lower and upper bounds using a root-finding algorithm
HII_eff_factor_lower_Cell = brentq(delta_chi2_eq, HII_eff_factor_fine[0], HII_eff_factor_fine[np.argmin(delta_chi2_Cell)], args=(delta_chi2_Cell, HII_eff_factor_fine))
HII_eff_factor_upper_Cell = brentq(delta_chi2_eq, HII_eff_factor_fine[np.argmin(delta_chi2_Cell)], HII_eff_factor_fine[-1], args=(delta_chi2_Cell, HII_eff_factor_fine))

# Combine the results into the confidence interval
HII_eff_factor_conf_interval_Cell = [HII_eff_factor_lower_Cell, HII_eff_factor_upper_Cell]



# Print results
print(f"68.5% confidence interval on HII_eff_factor for C_ell: {HII_eff_factor_conf_interval_Cell}")

# Save the confidence interval in a .npy file
np.save('confidence_interval_HII_eff_factor_Cell.npy', HII_eff_factor_conf_interval_Cell)

# Step 4: Plotting
plt.figure(figsize=(10, 6))

# Plot chi-square as a function of HII_eff_factor
plt.plot(HII_eff_factor_fine, chi_square_interpolated_Cell, label=r'$\chi^2$', color='blue')

# Highlight the confidence interval
for r in HII_eff_factor_conf_interval_Cell:
    plt.axvline(r, color='red', linestyle='--', label=f'68.5% CI: R = {r:.2f}')

# Annotate minimum chi-square
plt.axhline(chi2_min_Cell, color='green', linestyle=':', label=r'$\chi^2_{\text{min}}$')

# Customize plot
plt.title(r'$\chi^2$ as a Function of $R_{\text{bubble max}}$ for C ell', fontsize=14)
plt.xlabel(r'$R_{\text{bubble max}}$', fontsize=12)
plt.ylabel(r'$\chi^2$', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)

# Save and show the plot
plt.savefig('chi_square_with_confidence_interval_Cell_HII_eff_factor.pdf')
plt.show()

#____________________________CHECK THE BEST FIT VS MOCK DATA L1_NORM____________________#

best_fit_HII_eff_factor_l1_norm = HII_eff_factor_values[np.argmin(chi_square_all_l1_norm)]  # Best-fit HII_eff_factor

# Find the index of the best fit HII_eff_factor
best_fit_index_l1_norm = np.argmin(chi_square_all_l1_norm)

# Extract the L1 norm for the best-fit HII_eff_factor
best_fit_l1_norm = l1_norms_all_HII_eff_factor[best_fit_index_l1_norm]["l1_norm"]

# Extract corresponding bins as well
best_fit_bins = l1_norms_all_HII_eff_factor[best_fit_index_l1_norm]["bins"]

# Optionally, print to check
print(f"Best fit HII_eff_factor: {best_fit_HII_eff_factor_l1_norm}")
print(f"Best fit L1 norm shape: {best_fit_l1_norm.shape}")


# Plot the best fit vs the data L1 norm
plt.figure(figsize=(10, 6))

# Plot the actual data L1 norm (from mock data)
plt.plot(best_fit_bins, mock_l1_norm, label="Data L1 Norm", color='blue', linestyle='-', marker='o', markersize=4)

# Plot the best fit L1 norm
plt.plot(best_fit_bins, best_fit_l1_norm, label="Best Fit L1 Norm", color='red', linestyle='--', marker='x', markersize=6)

# Customize the plot
plt.title("Best Fit vs Data L1 Norm", fontsize=14)
plt.xlabel("SNR Bins", fontsize=12)
plt.ylabel("L1 Norm", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True)
# Optionally save the plot to a file
plt.savefig("best_fit_vs_data_L1_norm_HII_eff_factor.pdf")
# Show plot
plt.show()
#____________________________CHECK THE BEST FIT VS MOCK DATA CELL____________________#

best_fit_HII_eff_factor_Cell = HII_eff_factor_values[np.argmin(chi_square_all_Cell)]  # Best-fit HII_eff_factor

# Find the index of the best fit HII_eff_factor
best_fit_index_Cell = np.argmin(chi_square_all_Cell)

# Extract the L1 norm for the best-fit HII_eff_factor
best_fit_Cell = Cell_all_HII_eff_factor[best_fit_index_Cell]["Cell"]

# Extract corresponding bins as well
best_fit_ell = Cell_all_HII_eff_factor[best_fit_index_Cell]["ell"]

# Optionally, print to check
print(f"Best fit HII_eff_factor for Cell: {best_fit_HII_eff_factor_Cell}")
print(f"Best fit Cell shape: {best_fit_Cell.shape}")


# Plot the best fit vs the data L1 norm
plt.figure(figsize=(10, 6))

# Plot the actual data L1 norm (from mock data)
plt.plot(best_fit_ell, mock_Cell, label="Data L1 Norm", color='blue', linestyle='-', marker='o', markersize=4)

# Plot the best fit L1 norm
plt.plot(best_fit_ell, best_fit_Cell, label="Best Fit Cell", color='red', linestyle='--', marker='x', markersize=6)

# Customize the plot
plt.title("Best Fit vs Data Cell", fontsize=14)
plt.xlabel("ell", fontsize=12)
plt.ylabel("Cell", fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="upper right")
plt.grid(True)
# Optionally save the plot to a file
plt.savefig("best_fit_vs_data_Cell_HII_eff_factor.pdf")
# Show plot
plt.show()
