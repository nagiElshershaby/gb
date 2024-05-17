import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cftime import num2date, Datetime360Day

def plot_time_custom_function(file_path, variable_name, start_time, end_time, custom_function=None):
    dataset = nc.Dataset(file_path)

    try:
        xlon = dataset.variables['xlon'][:]
        xlat = dataset.variables['xlat'][:]
        time_values = dataset.variables['time'][:]

        var_name = None
        for varname in dataset.variables.keys():
            if variable_name in varname.lower():
                var_name = varname
                break
        if var_name is None:
            raise ValueError(f"Variable '{variable_name}' not found in the NetCDF file.")

        data = dataset.variables[var_name][:].squeeze()

        # Convert time values to datetime objects
        time_dates = num2date(time_values, units=dataset.variables['time'].units, calendar='360_day')
        start_datetime = start_time
        end_datetime = end_time

        start_index = np.argmax(time_dates >= start_datetime)
        end_index = np.argmax(time_dates >= end_datetime)

        if custom_function is None:
            # Default to mean if custom function is not provided
            aggregated_data = np.mean(data[start_index:end_index], axis=0)
        else:
            # Handle NaN and Inf values in the data
            data_slice = data[start_index:end_index]
            data_slice = np.ma.masked_invalid(data_slice)  # Mask NaN and Inf values
            aggregated_data = custom_function(data_slice, axis=0)

        # Check if aggregated_data is 1D and convert it to 2D if needed
        if len(aggregated_data.shape) == 1:
            aggregated_data = np.expand_dims(aggregated_data, axis=0)

        # Handle NaN and Inf values in the aggregated data for plotting
        aggregated_data = np.ma.masked_invalid(aggregated_data)


        # Plotting
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Define custom colormap from dark blue to light blue
        colors = [(0.5, 0, 0), (1, 0.3, 0.3), (1, 0.6, 0.6), (1, 0.8, 0.8), (1, 0.9, 0.9), (1, 1, 1)]
        cmap = mcolors.LinearSegmentedColormap.from_list('dark_blue_to_light_blue', colors)

        # Plot aggregated data
        levels = np.linspace(aggregated_data.min(), aggregated_data.max(), num=len(colors) + 1)
        contour = plt.contourf(xlon, xlat, aggregated_data, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())

        # Add coastlines
        ax.coastlines()

        # Add colorbar at the bottom
        cbar = plt.colorbar(contour, orientation='horizontal', ticks=levels, format="%.2f")
        cbar.set_label(var_name)

        # Add title and labels
        # plt.title(f"{variable_name} from time step {start_time} to {end_time}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Show plot
        plt.show()

    finally:
        # Close the NetCDF file
        dataset.close()

# Example usage:
# file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
# variable_name = 'one_day_precipitation_amount_per_time_period'
# plot_time_custom_function(file_path, variable_name)

# Example custom function: Count number of times a value is greater than zero
def count_positive(data, **kwargs):
    return np.sum(data > 0, **kwargs)

from datetime import datetime, timedelta
from cftime import Datetime360Day
# Example usage:
file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
variable_name = 'one_day_precipitation_amount_per_time_period'
ntopical_nights_file_path = 'netcdf_files/ntopical_nights.mint.lt.20_RCP85.nc'
ntopical_nights_variable_name = 'night'
start_datetime = Datetime360Day(2024, 1, 1)
end_datetime = Datetime360Day(2024, 9, 1)
plot_time_custom_function(file_path, variable_name, start_time=start_datetime, end_time=end_datetime)
