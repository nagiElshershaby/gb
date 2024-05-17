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
            aggregated_data = custom_function(data[start_time:end_time])
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

def count_frost_days(data, threshold=0, **kwargs):
    """
    Count the number of days where the daily minimum temperature falls below freezing.

    Args:
        data (numpy.ndarray): Array containing daily minimum temperature values.
        threshold (float, optional): Threshold temperature for defining frost days (default: 0).

    Returns:
        int: Number of frost days.
    """
    return np.sum(data < threshold, **kwargs)



def count_summer_days(data, threshold=25, **kwargs):
    """
    Count the number of days where the daily maximum temperature exceeds a threshold.

    Args:
        data (numpy.ndarray): Array containing daily maximum temperature values.
        threshold (float, optional): Threshold temperature for defining summer days (default: 25).

    Returns:
        int: Number of summer days.
    """
    return np.sum(data > threshold, **kwargs)


def count_icing_days(data, threshold=0, **kwargs):
    """
    Count the number of days where the daily maximum temperature falls below freezing.

    Args:
        data (numpy.ndarray): Array containing daily maximum temperature values.
        threshold (float, optional): Threshold temperature for defining icing days (default: 0).

    Returns:
        int: Number of icing days.
    """
    return np.sum(data < threshold, **kwargs)

def count_tropical_nights(data, threshold=20, **kwargs):
    """
    Count the number of days where the daily minimum temperature exceeds a threshold.

    Args:
        data (numpy.ndarray): Array containing daily minimum temperature values.
        threshold (float, optional): Threshold temperature for defining tropical nights (default: 20).

    Returns:
        int: Number of tropical nights.
    """
    return np.sum(data > threshold, **kwargs)


def count_growing_season_length(data, threshold_high=5, threshold_low=5, consecutive_days=6):
    """
    Count the growing season length based on daily mean temperature.

    Args:
        data (numpy.ndarray): Array containing daily mean temperature values.
        threshold_high (float, optional): Threshold temperature for defining high temperature days (default: 5).
        threshold_low (float, optional): Threshold temperature for defining low temperature days (default: 5).
        consecutive_days (int, optional): Number of consecutive days to consider for the span (default: 6).

    Returns:
        int: Growing season length.
    """
    above_threshold = data > threshold_high
    below_threshold = data < threshold_low

    # Find the first occurrence of at least consecutive_days days with TG > threshold_high
    start_idx = 0
    for i in range(len(data) - consecutive_days + 1):
        if np.all(above_threshold[i:i + consecutive_days]):
            start_idx = i
            break

    # Find the first occurrence after July 1st (or January 1st in SH) of at least consecutive_days days with TG < threshold_low
    end_idx = len(data)
    for i in range(start_idx, len(data) - consecutive_days + 1):
        if i >= 181:  # July 1st (or January 1st in SH)
            if np.all(below_threshold[i:i + consecutive_days]):
                end_idx = i
                break

    # Count the number of days between the two spans
    growing_season_length = end_idx - start_idx

    return growing_season_length

def calculate_max_daily_max_temperature(data):
    """
    Calculate the maximum value of daily maximum temperature within a specific period (e.g., month).

    Args:
        data (numpy.ndarray): Array containing daily maximum temperature values.

    Returns:
        float: Maximum value of daily maximum temperature.
    """
    return np.max(data)

def calculate_highest_daily_min_temperature(data):
    """
    Calculate the highest daily minimum temperature recorded within a specific period.

    Args:
        data (numpy.ndarray): Array containing daily minimum temperature values.

    Returns:
        float: Highest daily minimum temperature recorded within the specified period.
    """
    return np.max(data)


def calculate_min_daily_max_temperature(data):
    """
    Calculate the minimum value of daily maximum temperature within a specific period.

    Args:
        data (numpy.ndarray): Array containing daily maximum temperature values.

    Returns:
        float: Minimum value of daily maximum temperature recorded within the specified period.
    """
    return np.min(max(data))
def calculate_min_daily_min_temperature(data):
    """
    Calculate the minimum value of daily minimum temperature within a specific period.

    Args:
        data (numpy.ndarray): Array containing daily minimum temperature values.

    Returns:
        float: Minimum value of daily minimum temperature recorded within the specified period.
    """
    return np.min(data)

def percentage_days_below_percentile(data, percentile=10):
    """
    Calculate the percentage of days when the daily minimum temperature is below the given percentile.

    Args:
        data (numpy.ndarray): Array containing daily minimum temperature values.
        percentile (int, optional): Percentile value (default: 10).

    Returns:
        float: Percentage of days when the daily minimum temperature is below the given percentile.
    """
    # Calculate the percentile value
    percentile_value = np.percentile(data, percentile)

    # Count the number of days below the percentile
    days_below_percentile = np.sum(data < percentile_value)

    # Calculate the percentage
    total_days = len(data)
    percentage = (days_below_percentile / total_days) * 100

    return percentage

def percentage_days_above_percentile(data, percentile=90):
    """
    Calculate the percentage of days when the daily minimum temperature is above the given percentile.

    Args:
        data (numpy.ndarray): Array containing daily minimum temperature values.
        percentile (int, optional): Percentile value (default: 90).

    Returns:
        float: Percentage of days when the daily minimum temperature is above the given percentile.
    """
    # Calculate the percentile value
    percentile_value = np.percentile(data, percentile)

    # Count the number of days above the percentile
    days_above_percentile = np.sum(data > percentile_value)

    # Calculate the percentage
    total_days = len(data)
    percentage = (days_above_percentile / total_days) * 100

    return percentage


def calculate_WSDI(data, percentile=90, consecutive_days=6):
    """
    Calculate the Warm Spell Duration Index (WSDI).

    Args:
        data (numpy.ndarray): Array containing daily maximum temperature values.
        percentile (int, optional): Percentile value (default: 90).
        consecutive_days (int, optional): Number of consecutive days (default: 6).

    Returns:
        int: Annual count of days with at least 6 consecutive days when TX > 90th percentile.
    """
    # Calculate the 90th percentile value
    percentile_value = np.percentile(data, percentile)

    # Find the indices where TX > 90th percentile
    exceed_indices = np.where(data > percentile_value)[0]

    # Initialize counter for consecutive days
    consecutive_count = 0

    # Initialize counter for WSDI
    WSDI_count = 0

    # Iterate through the indices
    for i in range(1, len(exceed_indices)):
        # Check if the current index is consecutive with the previous one
        if exceed_indices[i] == exceed_indices[i - 1] + 1:
            consecutive_count += 1
        else:
            consecutive_count = 0

        # Check if there are at least 6 consecutive days
        if consecutive_count >= consecutive_days - 1:
            WSDI_count += 1

    return WSDI_count


def calculate_CSDI(data, percentile=10, consecutive_days=6):
    """
    Calculate the Cold Spell Duration Index (CSDI).

    Args:
        data (numpy.ndarray): Array containing daily minimum temperature values.
        percentile (int, optional): Percentile value (default: 10).
        consecutive_days (int, optional): Number of consecutive days (default: 6).

    Returns:
        int: Annual count of days with at least 6 consecutive days when TN < 10th percentile.
    """
    # Calculate the 10th percentile value
    percentile_value = np.percentile(data, percentile)

    # Find the indices where TN < 10th percentile
    below_indices = np.where(data < percentile_value)[0]

    # Initialize counter for consecutive days
    consecutive_count = 0

    # Initialize counter for CSDI
    CSDI_count = 0

    # Iterate through the indices
    for i in range(1, len(below_indices)):
        # Check if the current index is consecutive with the previous one
        if below_indices[i] == below_indices[i - 1] + 1:
            consecutive_count += 1
        else:
            consecutive_count = 0

        # Check if there are at least 6 consecutive days
        if consecutive_count >= consecutive_days - 1:
            CSDI_count += 1

    return CSDI_count


def calculate_DTR(daily_max_temps, daily_min_temps):
    """
    Calculate the Daily Temperature Range (DTR).

    Args:
        daily_max_temps (numpy.ndarray): Array containing daily maximum temperature values.
        daily_min_temps (numpy.ndarray): Array containing daily minimum temperature values.

    Returns:
        float: Daily Temperature Range.
    """
    # Ensure the input arrays have the same shape
    assert daily_max_temps.shape == daily_min_temps.shape, "Input arrays must have the same shape"

    # Calculate the difference between daily maximum and minimum temperatures
    temp_range = daily_max_temps - daily_min_temps

    # Calculate the average temperature range
    avg_temp_range = np.mean(temp_range)

    return avg_temp_range


def calculate_ETR(daily_max_temps, daily_min_temps):
    """
    Calculate the Extreme Temperature Range (ETR) for each month.

    Args:
        daily_max_temps (numpy.ndarray): Array containing daily maximum temperature values for each month.
        daily_min_temps (numpy.ndarray): Array containing daily minimum temperature values for each month.

    Returns:
        numpy.ndarray: Array containing the Extreme Temperature Range (ETR) for each month.
    """
    # Ensure the input arrays have the same shape
    assert daily_max_temps.shape == daily_min_temps.shape, "Input arrays must have the same shape"

    # Calculate the extreme temperature range for each month
    ETR = daily_max_temps - daily_min_temps

    return ETR


def calculate_CDD(data, base_temperature):
    """
    Calculate the Cooling Degree Days (CDD) for a specific base temperature.

    Args:
        data (numpy.ndarray): Array containing daily mean temperature values.
        base_temperature (float): Base temperature for CDD calculation.

    Returns:
        float: Cooling Degree Days (CDD) for the specified base temperature.
    """
    # Calculate the differences between daily mean temperature and base temperature
    differences = data - base_temperature

    # Set negative differences to zero (since CDD only accounts for temperatures above base temperature)
    differences[differences < 0] = 0

    # Sum the positive differences to get the CDD
    CDD = np.sum(differences)

    return CDD


import numpy as np


def calculate_HDD(data, base_temperature):
    """
    Calculate the Heating Degree Days (HDD) for a specific base temperature.

    Args:
        data (numpy.ndarray): Array containing daily mean temperature values.
        base_temperature (float): Base temperature for HDD calculation.

    Returns:
        float: Heating Degree Days (HDD) for the specified base temperature.
    """
    # Calculate the differences between base temperature and daily mean temperature
    differences = base_temperature - data

    # Set negative differences to zero (since HDD only accounts for temperatures below base temperature)
    differences[differences < 0] = 0

    # Sum the positive differences to get the HDD
    HDD = np.sum(differences)

    return HDD


# Example usage:
# HDD = calculate_HDD(data, base_temperature)


file_path = 'netcdf_files/extreme_temp_range_RCP45.nc'
variable_name = 'temp'
ntopical_nights_file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
ntopical_nights_variable_name = 'pr'
plot_time_custom_function(file_path,
                          variable_name,
                          custom_function=count_positive,
                          ishigh=True,
                          istemperature=True,
                          lon1=24, lat1=22,
                          lon2=36, lat2=22,
                          lon3=36, lat3=32,
                          lon4=24, lat4=32)