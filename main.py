# import netCDF4 as nc
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
#
# # Open the NetCDF file
# file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
# dataset = nc.Dataset(file_path)

# try:
#     # Extract latitude and longitude data
#     xlon = dataset.variables['xlon'][:]
#     xlat = dataset.variables['xlat'][:]
#
#     # Extract tropical nights index data
#     highest_one_day_precipitation_amount_per_time_period = dataset.variables['highest_one_day_precipitation_amount_per_time_period'][:].squeeze()
#
#     # Plotting
#     fig = plt.figure(figsize=(10, 8))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#
#     # Plot tropical nights index
#     # plt.contourf(xlon, xlat, highest_one_day_precipitation_amount_per_time_period , cmap='viridis', transform=ccrs.PlateCarree())
#
#     highest_one_day_precipitation_amount_per_time_period_2d = highest_one_day_precipitation_amount_per_time_period[0, :, :]
#
#     # Plot tropical nights index
#     plt.contourf(xlon, xlat, highest_one_day_precipitation_amount_per_time_period_2d, cmap='viridis', transform=ccrs.PlateCarree())
#     # Add coastlines
#     ax.coastlines()
#
#     # Add colorbar
#     cbar = plt.colorbar()
#     cbar.set_label('highest_one_day_precipitation_amount_per_time_period')
#
#     # Add title and labels
#     plt.title('highest_one_day_precipitation_amount_per_time_period')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#
#     # Show plot
#     plt.show()
#
# finally:
#     # Close the NetCDF file
#     dataset.close()

#-----------------------------------------------------------------

# file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
# dataset = nc.Dataset(file_path)
#
# try:
#     # Accessing dimensions
#     print("Dimensions:")
#     for dimname, dim in dataset.dimensions.items():
#         print(f"  Dimension name: {dimname}")
#         print(f"  Size: {len(dim)}")
#
#     # Accessing variables
#     print("\nVariables:")
#     for varname, var in dataset.variables.items():
#         print(f"  Variable name: {varname}")
#         print(f"  Dimensions: {var.dimensions}")
#         print(f"  Shape: {var.shape}")
#         print(f"  Attributes: {var.ncattrs()}")
#
#         # Accessing variable data
#         if len(var.shape) > 0:  # Avoid accessing data for scalar variables
#             print(f"  Data: {var[:]}")  # Print the data if variable is not a scalar
#
#     # Accessing global attributes
#     print("\nGlobal Attributes:")
#     for attrname in dataset.ncattrs():
#         print(f"  Attribute name: {attrname}")
#         print(f"  Attribute value: {getattr(dataset, attrname)}")
#
# finally:
#     # Close the NetCDF file
#     dataset.close()

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

def plot_netCDF_data(file_path):
    # Open the NetCDF file
    dataset = nc.Dataset(file_path)

    try:
        # Extract latitude and longitude data
        xlon = dataset.variables['xlon'][:]
        xlat = dataset.variables['xlat'][:]

        # Extract precipitation data
        variable_name = None
        for varname in dataset.variables.keys():
            if 'pr' in varname.lower():  # Assuming precipitation data has 'pr' in its name
                variable_name = varname
                break
        if variable_name is None:
            raise ValueError("Precipitation variable not found in the NetCDF file.")

        precipitation_data = dataset.variables[variable_name][:].squeeze()
        precipitation_data_2d = precipitation_data[0, :, :]

        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Plot precipitation data
        plt.contourf(xlon, xlat, precipitation_data_2d, cmap='viridis', transform=ccrs.PlateCarree())

        # Add coastlines
        ax.coastlines()

        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label(variable_name)

        # Add title and labels
        plt.title(variable_name)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Show plot
        plt.show()

    finally:
        # Close the NetCDF file
        dataset.close()

def plot_netCDF_data_with_variable_name(file_path, variable_name):
    # Open the NetCDF file
    dataset = nc.Dataset(file_path)

    try:
        # Extract latitude and longitude data
        xlon = dataset.variables['xlon'][:]
        xlat = dataset.variables['xlat'][:]

        # Extract data for the specified variable
        # if variable_name not in dataset.variables:
        #     raise ValueError(f"Variable '{variable_name}' not found in the NetCDF file.")

        var_name = None
        print(dataset.variables.keys())
        for varname in dataset.variables.keys():
            if variable_name in varname.lower():  # Assuming precipitation data has 'pr' in its name
                var_name = varname
                break
        if var_name is None:
            raise ValueError(f"Variable '{var_name}' not found in the NetCDF file.")

        # data = dataset.variables[var_name][:].squeeze()
        # data_2d = data[1, :, :]  # Assuming the time dimension is the first dimension

        data = dataset.variables[var_name][:].squeeze()  # Extract the variable data
        print(data.shape)
        print(len(data.shape))
        # Check the shape of the variable data
        if len(data.shape) == 3:
            # Assuming the time dimension is the first dimension
            data_2d = data[0, :, :]
        # elif len(data.shape) == 2:
        #     # If the data is already 2D, no need to slice it further
        #     data_2d = data
        else:
            data_2d = data
        # else:
        #     raise ValueError("Unsupported data shape. Expected 2D or 3D data.")

        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Plot data
        plt.contourf(xlon, xlat, data_2d, cmap='viridis', transform=ccrs.PlateCarree())

        # Add coastlines
        ax.coastlines()

        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label(variable_name)

        # Add title and labels
        plt.title(variable_name)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Show plot
        plt.show()

    finally:
        # Close the NetCDF file
        dataset.close()

def plot_time_aggregated_data(file_path, variable_name, start_time=0, end_time=None, aggregation_func=np.mean):
    # Open the NetCDF file
    dataset = nc.Dataset(file_path)

    try:
        # Extract latitude and longitude data
        xlon = dataset.variables['xlon'][:]
        xlat = dataset.variables['xlat'][:]

        # Extract data for the specified variable
        var_name = None
        for varname in dataset.variables.keys():
            if variable_name in varname.lower():  # Case-insensitive check for variable name match
                var_name = varname
                break
        if var_name is None:
            raise ValueError(f"Variable '{variable_name}' not found in the NetCDF file.")

        data = dataset.variables[var_name][:].squeeze()  # Extract the variable data and squeeze

        # Check if data is 2D or 3D
        if len(data.shape) == 3:  # 3D array (time, latitude, longitude)
            # Check if end_time is specified, if not, set it to the end of the time dimension
            if end_time is None:
                end_time = len(data)

            # Calculate the aggregated data across the specified time range
            aggregated_data = aggregation_func(data[start_time:end_time], axis=0)
        elif len(data.shape) == 2:  # 2D array (latitude, longitude)
            aggregated_data = data  # No time dimension, so no aggregation needed
        else:
            raise ValueError("Unsupported data shape. Expected 2D or 3D data.")

        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Plot aggregated data
        plt.contourf(xlon, xlat, aggregated_data, cmap='viridis', transform=ccrs.PlateCarree())

        # Add coastlines
        ax.coastlines()

        # Add colorbar at the bottom
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_label(var_name)

        # Add title and labels
        plt.title(f"{variable_name} from time step {start_time} to {end_time}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Show plot
        plt.show()

    finally:
        # Close the NetCDF file
        dataset.close()


def plot_time_custom_function(file_path, variable_name, start_time=0, end_time=30, custom_function=None):
    # Open the NetCDF file
    dataset = nc.Dataset(file_path)

    try:
        # Extract latitude and longitude data
        xlon = dataset.variables['xlon'][:]
        xlat = dataset.variables['xlat'][:]

        # Extract data for the specified variable
        var_name = None
        for varname in dataset.variables.keys():
            if variable_name in varname.lower():  # Case-insensitive check for variable name match
                var_name = varname
                break
        if var_name is None:
            raise ValueError(f"Variable '{variable_name}' not found in the NetCDF file.")

        data = dataset.variables[var_name][:].squeeze()  # Extract the variable data and squeeze

        # Check if data is 2D or 3D
        print(len(data.shape))
        if len(data.shape) == 3:  # 3D array (time, latitude, longitude)
            # Check if end_time is specified, if not, set it to the end of the time dimension
            if end_time is None:
                end_time = len(data)

            # Calculate the aggregated data across the specified time range
            aggregated_data = custom_function(data[start_time:end_time], axis=0)
        elif len(data.shape) == 2:  # 2D array (latitude, longitude)
            aggregated_data = data  # Apply custom function directly
        else:
            raise ValueError("Unsupported data shape. Expected 2D or 3D data.")

        # Plotting
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # # Define custom colors for the segments
        # colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        #
        # # Create a custom colormap
        # cmap = mcolors.ListedColormap(colors)

        # Define custom colormap from dark blue to light blue
        colors = [(1, 1, 1), (0.8, 0.9, 1), (0.6, 0.8, 1), (0.3, 0.6, 1), (0, 0.3, 1), (0, 0, 0.5)] # blues
        # colors = [(0.5, 0, 0), (1, 0, 0), (1, 0.3, 0.3), (1, 0.6, 0.6), (1, 0.8, 0.8), (1, 1, 1)] # reds
        # colors.reverse()
        cmap = mcolors.LinearSegmentedColormap.from_list('dark_blue_to_light_blue', colors)

        # Plot aggregated data
        levels = np.linspace(aggregated_data.min(), aggregated_data.max(), num=len(colors) + 1)  # Create levels
        contour = plt.contourf(xlon, xlat, aggregated_data, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())

        # Add coastlines
        ax.coastlines()

        # Add colorbar at the bottom
        cbar = plt.colorbar(contour, orientation='horizontal', ticks=levels, format="%.2f")
        cbar.set_label(var_name)

        # Add title and labels
        plt.title(f"{variable_name} from time step {start_time} to {end_time}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Show plot
        plt.show()

    finally:
        # Close the NetCDF file
        dataset.close()

# Example custom function: Count number of times a value is greater than zero
def count_positive(data, **kwargs):
    return np.sum(data > 0, **kwargs)


# Example usage:
file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
variable_name = 'one_day_precipitation_amount_per_time_period'
#ntopical_nights_file_path = 'netcdf_files/ntopical_nights.mint.lt.20_RCP85.nc'
ntopical_nights_file_path = 'netcdf_files/ndays_maxt.gt.25_RCP85.nc'
ntopical_nights_variable_name = 'day'
# plot_netCDF_data(file_path)
# plot_netCDF_data_with_variable_name(file_path, variable_name)
# plot_netCDF_data_with_variable_name(ntopical_nights_file_path, ntopical_nights_variable_name)
# plot_time_aggregated_data(file_path, variable_name)
# plot_time_aggregated_data(ntopical_nights_file_path, ntopical_nights_variable_name)
# plot_time_custom_function(file_path, variable_name, custom_function=count_positive, start_time= 100)
plot_time_custom_function(ntopical_nights_file_path, ntopical_nights_variable_name, custom_function=count_positive)
