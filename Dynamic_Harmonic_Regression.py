
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import curve_fit
from matplotlib import font_manager

######################################################### Harmonic regression coefficients extraction of S2 indices
region = 'Tarbes'
year = 2021
name='S2_important'
in_directory=  f'F:/Project/{region}{year}/' 
dataset = np.load(f'{in_directory}/{name}.npz', allow_pickle=True)
array_names = dataset.files
print(array_names)

# Use a dictionary to hold the extracted arrays
data = {key: dataset[key] for key in array_names}

S2 = data["S2_ind"]
y=data['y']
dates = data["dates_S2"]

### Function to calculate t which will be used in Harmonic regression 
def calculate_t_data(dates, start_date, end_date):
    t_data = []
    start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
    period_length = (end_date - start_date).days
    
    for date_str in dates:
        date_str = str(date_str) 
        date = datetime.datetime.strptime(date_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")
        day_of_period = (date - start_date).days
        t = day_of_period / period_length
        t_data.append(t)
        
    return t_data

# Defin the start and end date of time series. Ensure the date format is correct for the dates in your data
start_date = str(dates[0])
start_date = datetime.datetime.strptime(start_date.split('.')[0], "%Y-%m-%dT%H:%M:%S")
end_date = str(dates[-1])
end_date = datetime.datetime.strptime(end_date.split('.')[0], "%Y-%m-%dT%H:%M:%S")

print(start_date)
print(end_date)

t_data = calculate_t_data(dates, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
print(t_data)

# Function to define the harmonic model
def harmonic_model(t, c, a1, b1, a2, b2, omega=0.8):
    return c + a1 * np.cos(2 * np.pi * omega * t) + b1 * np.sin(2 * np.pi * omega * t) +                a2 * np.cos(4 * np.pi * omega * t) + b2 * np.sin(4 * np.pi * omega * t)

initial_guess = [1, 1, 1, 1, 1]  # Initial guess for parameters c, a1, b1, a2, b2
num_samples, num_dates, num_features = S2.shape

# Array to store coefficients
fitted_coeffs = np.zeros((num_samples, num_features, 5))  # Assuming 5 coefficients (c, a1, b1, a2, b2)

# Fitting the model and storing coefficients
#params_S2 will hold the optimal parameters found by curve_fit.
#_ will hold the covariance matrix of the parameters, but since _ is used, it indicates that the covariance matrix is not needed in the subsequent code and can be discarded.

for i in range(num_samples):
    for j in range(num_features):
        params_S2, _ = curve_fit(harmonic_model, t_data, S2[i,:,j], p0=initial_guess)        
        c_S2, a1_S2, b1_S2, a2_S2, b2_S2 = params_S2
        fitted_coeffs[i, j] = [c_S2, a1_S2, b1_S2, a2_S2, b2_S2]

############################################################ Evaluation       
# Function to calculate R-squared
def calculate_r_squared(observed, predicted):
    ss_res = np.sum((observed - predicted) ** 2, axis=0)
    ss_tot = np.sum((observed - np.mean(observed, axis=0)) ** 2, axis=0)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Calculate R-squared for each sample and feature
num_features =4
r_squared_values = np.zeros((num_samples, num_features))

for i in range(num_samples):
    for j in range(num_features):
        predicted = harmonic_model(np.array(t_data), *fitted_coeffs[i, j])
        r_squared_values[i, j] = calculate_r_squared(S2[i, :, j], predicted)

# Print mean R-squared value for 4 classes: TRN, MISm SOJ and other classes 
mean_r_squared_All = {}
mean_r_squared_TRN = {}
mean_r_squared_MIS = {}
mean_r_squared_SOJ = {}

for i in range(num_features):  
    mean_r_squared_All[i] = np.mean(r_squared_values[:, i])
    print(f"Mean R-squared_All_{i}: {mean_r_squared_All[i]}")

    mean_r_squared_TRN[i] = np.mean(r_squared_values[:, i][y == 'TRN'])
    print(f"Mean R-squared_TRN_{i}: {mean_r_squared_TRN[i]}")

    mean_r_squared_MIS[i] = np.mean(r_squared_values[:, i][y == 'MIS'])
    print(f"Mean R-squared_MIS_{i}: {mean_r_squared_MIS[i]}")

    mean_r_squared_SOJ[i] = np.mean(r_squared_values[:, i][y == 'SOJ'])
    print(f"Mean R-squared_SOJ_{i}: {mean_r_squared_SOJ[i]}")

###### Add the fitted coeffcients as a new arry to the input npz file

data["S2_ind_Harmonic"] = fitted_coeffs
new_dataset_path = f'{in_directory}/{name}.npz'
np.savez(new_dataset_path,  **data        
        )   

