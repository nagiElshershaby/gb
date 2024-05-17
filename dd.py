import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
from shapely.geometry import Polygon


def create_color_levels(aggregated_data, istemperature=False, ishigh=False):
    """
    Create color levels for plotting based on temperature and non-temperature variables.

    Args:
        aggregated_data (numpy.ndarray): Aggregated data for plotting.
        istemperature (bool, optional): Flag indicating if the variable is temperature (default: False).
        ishigh (bool, optional): Flag indicating if the temperature is high (default: False).

    Returns:
        levels (numpy.ndarray): Array of levels for the color map.
        cmap (matplotlib.colors.Colormap): Colormap for the plot.
    """
    # Define colormaps for temperature and non-temperature variables
    colorsblue = [(1, 1, 1), (0.8, 0.9, 1), (0.6, 0.8, 1), (0.3, 0.6, 1), (0, 0.3, 1), (0, 0, 0.5)]
    colorsred = [(0.5, 0, 0), (1, 0.3, 0.3), (1, 0.6, 0.6), (1, 0.8, 0.8), (1, 0.9, 0.9), (1, 1, 1)]

    # Define the colormap based on istemperature and ishigh flags
    if istemperature:
        cmap = plt.cm.colors.ListedColormap(colorsblue) if not ishigh else plt.cm.colors.ListedColormap(colorsred)

        # Plot aggregated data
    levels = np.linspace(aggregated_data.min(), aggregated_data.max(), num=len(cmap.colors) + 1)
    return levels, cmap


def create_polygon(lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4):
    """
    Creates a polygon based on the provided coordinates.

    Args:
        lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4 (float): Coordinates of the four corners of the polygon.

    Returns:
        Polygon: A shapely Polygon object representing the polygon.
    """
    coords = [(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4), (lon1, lat1)]
    polygon = Polygon(coords)
    return polygon


def plot_time_custom_function(file_path, variable_name, start_time=0, end_time=None, custom_function=None,
                              istemperature=False, ishigh=False, lon1=None, lat1=None, lon2=None, lat2=None, lon3=None,
                              lat3=None, lon4=None, lat4=None):
    """
    Plots data from a NetCDF file with a square on the map.

    Args:
        file_path (str): Path to the NetCDF file.
        variable_name (str): Name of the variable to plot.
        start_time (int, optional): Start time for data extraction (default: 0).
        end_time (int, optional): End time for data extraction (default: None).
        custom_function (callable, optional): Custom function to apply to the data.
        istemperature (bool, optional): Flag indicating if the variable is temperature (default: False).
        ishigh (bool, optional): Flag indicating if the temperature is high (default: False).
        lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4 (float, optional):
            Coordinates of the four corners of the square to plot.

    Raises:
        ValueError: If the variable is not found in the NetCDF file or if invalid coordinates are provided.
    """

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
            aggregated_data = custom_function(data[start_time:end_time], axis=0)
        elif len(data.shape) == 2:  # 2D array (latitude, longitude)
            aggregated_data = data  # Apply custom function directly
        else:
            raise ValueError("Unsupported data shape. Expected 2D or 3D data.")

        # Plotting
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        # Define colormaps for temperature and non-temperature variables
        levels, cmap = create_color_levels(aggregated_data, istemperature, ishigh)

        # Plot aggregated data
        levels = np.linspace(aggregated_data.min(), aggregated_data.max(), num=len(cmap.colors) + 1)  # Create levels
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

        # Check if square coordinates are provided
        if any([lon1 is None, lat1 is None, lon2 is None, lat2 is None, lon3 is None, lat3 is None, lon4 is None,
                lat4 is None]):
            raise ValueError(
                "Please provide all four coordinates (lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4) for the square.")

        # Create the polygon object using the provided coordinates
        polygon = create_polygon(lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4)

        # Add the polygon object to the plot
        ax.add_geometries([polygon], ccrs.PlateCarree(), facecolor='red', alpha=0.3)

        # Adjust limits slightly larger than the square for better visualization
        x_min = min(lon1, lon2, lon3, lon4) - 1
        x_max = max(lon1, lon2, lon3, lon4) + 1
        y_min = min(lat1, lat2, lat3, lat4) - 1
        y_max = max(lat1, lat2, lat3, lat4) + 1
        ax.set_extent([x_min, x_max, y_min, y_max], ccrs.PlateCarree())

        plt.show()

    finally:
        # Close the NetCDF file
        dataset.close()


def count_positive(data, **kwargs):
    return np.sum(data > 0, **kwargs)


file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
variable_name = 'one_day_precipitation_amount_per_time_period'
ntopical_nights_file_path = 'netcdf_files/ntopical_nights.mint.lt.20_RCP85.nc'
ntopical_nights_variable_name = 'night'
plot_time_custom_function(ntopical_nights_file_path,
                          ntopical_nights_variable_name,
                          custom_function=count_positive,
                          ishigh=True,
                          istemperature=True,
                          lon1=24, lat1=22,
                          lon2=36, lat2=22,
                          lon3=36, lat3=32,
                          lon4=24, lat4=32)