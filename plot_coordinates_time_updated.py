import cftime
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
from shapely.geometry import Polygon
from datetime import datetime


def create_color_levels(aggregated_data, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    levels = np.linspace(aggregated_data.min(), aggregated_data.max(), num=cmap.N + 1)
    return levels, cmap


def create_polygon(lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4):
    coords = [(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4), (lon1, lat1)]
    polygon = Polygon(coords)
    return polygon


def filter_data_by_season_and_year(data, time_variable, season, start_year, end_year):
    months = {
        'january': [1],
        'february': [2],
        'march': [3],
        'april': [4],
        'may': [5],
        'june': [6],
        'july': [7],
        'august': [8],
        'september': [9],
        'october': [10],
        'november': [11],
        'december': [12],
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'fall': [9, 10, 11],
        'annual': list(range(1, 13))
    }

    selected_months = months.get(season.lower(), list(range(1, 13)))
    filtered_data = []
    filtered_dates = []

    for i, date in enumerate(
            cftime.num2date(time_variable[:], units=time_variable.units, calendar=time_variable.calendar)):
        if date.year in range(start_year, end_year + 1) and date.month in selected_months:
            # print(date)
            filtered_data.append(data[i])
            filtered_dates.append(date)
    filtered_data = np.array(filtered_data)
    # print(filtered_dates)
    return filtered_data, filtered_dates


def plot_time_custom_function_with_dates(file_path, variable_name, start_date=None, end_date=None,
                                         start_year=None, end_year=None, season='annual',
                                         index_name='TXx', data_type='temp', lon1=None, lat1=None,
                                         lon3=None, lat3=None):
    dataset = nc.Dataset(file_path)
    try:
        xlon = dataset.variables['xlon'][:]
        xlat = dataset.variables['xlat'][:]
        var_name = None
        for varname in dataset.variables.keys():
            if variable_name in varname.lower():
                var_name = varname
                break
        if var_name is None:
            raise ValueError(f"Variable '{variable_name}' not found in the NetCDF file.")
        data = dataset.variables[var_name][:].squeeze()
        time_variable = dataset.variables['time']

        if len(data.shape) == 3:
            if start_date and end_date:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                start_idx = nc.date2index(start_date_obj, time_variable, select='nearest')
                end_idx = nc.date2index(end_date_obj, time_variable, select='nearest')
                aggregated_data = index_function_by_name(index_name, data[start_idx:end_idx + 1], axis=0)
            elif start_year and end_year and season:
                filtered_data, filtered_dates = filter_data_by_season_and_year(data, time_variable, season, start_year,
                                                                               end_year)
                aggregated_data = index_function_by_name(index_name, filtered_data, axis=0)
            else:
                aggregated_data = index_function_by_name(index_name, data, axis=0)
        elif len(data.shape) == 2:
            aggregated_data = data
        else:
            raise ValueError("Unsupported data shape. Expected 2D or 3D data.")

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cmap_name = get_color_mapping_according_to_type(data_type)
        levels, cmap = create_color_levels(aggregated_data, cmap_name)
        # print(f"Levels: {levels}")
        # Ensure levels is sorted
        if levels is not None:
            levels = sorted(levels)
            # Contour levels must be increasing
            if levels[0] > levels[-1]:
                levels = levels[::-1]
            elif levels[0] == levels[-1]:
                levels = np.linspace(levels[0], levels[0] + 1, num=10)
            print(f"Sorted levels: {levels}")
        else:
            levels = np.linspace(np.min(aggregated_data), np.max(aggregated_data), num=10)
            print(f"Default levels: {levels}")

        contour = plt.contourf(xlon, xlat, aggregated_data, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
        ax.coastlines()
        cbar = plt.colorbar(contour, orientation='horizontal')
        num_ticks = 10
        ticks = np.linspace(aggregated_data.min(), aggregated_data.max(), num=num_ticks)
        cbar.set_ticks(ticks)
        cbar.set_label(var_name)
        if start_date and end_date:
            plt.title(f"{variable_name} from {start_date} to {end_date}")
        elif start_year and end_year and season:
            plt.title(f"{variable_name} from {start_year} to {end_year} ({season.capitalize()})")
        else:
            plt.title(f"{variable_name}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        if any(coord is None for coord in [lon1, lat1, lon3, lat3]):
            extent = [xlon.min(), xlon.max(), xlat.min(), xlat.max()]
        else:
            polygon = create_polygon(lon1, lat1, lon3, lat1, lon3, lat3, lon1, lat3)
            ax.add_geometries([polygon], ccrs.PlateCarree(), facecolor='none', alpha=0.3)
            x_min = min(lon1, lon3) - 1
            x_max = max(lon1, lon3) + 1
            y_min = min(lat1, lat3) - 1
            y_max = max(lat1, lat3) + 1
            extent = [x_min, x_max, y_min, y_max]

        ax.set_extent(extent, crs=ccrs.PlateCarree())

        plt.show()
    finally:
        dataset.close()


def count_positive(data, **kwargs):
    return np.sum(data > 0, **kwargs)


def get_color_mapping_according_to_type(t):
    if t in ['temp', 'temperature']:
        return 'plasma'
    elif t in ['PR', 'pr']:
        return 'viridis'
    elif t in ['HW', 'hw']:
        return 'magma'
    elif t in ['CW', 'cw']:
        return 'inferno'
    else:
        return 'plasma'


def count_frost_days(data, **kwargs):
    return np.sum(data < 0, **kwargs)


def count_summer_days(data, **kwargs):
    return np.sum(data > 25, **kwargs)


def count_icing_days(data, **kwargs):
    return np.sum(data < 0, **kwargs)


def count_tropical_nights(data, **kwargs):
    return np.sum(data > 20, **kwargs)


def max_daily_max_temp(data, **kwargs):
    return np.max(data, **kwargs)


def max_daily_min_temp(data, **kwargs):
    return np.max(data, **kwargs)


def min_daily_max_temp(data, **kwargs):
    return np.min(data, **kwargs)


def min_daily_min_temp(data, **kwargs):
    return np.min(data, **kwargs)


# Add more index functions as needed

def index_function_by_name(index_name, data, **kwargs):
    if index_name == 'FD':
        return count_frost_days(data, **kwargs)
    elif index_name == 'SU':
        return count_summer_days(data, **kwargs)
    elif index_name == 'ID':
        return count_icing_days(data, **kwargs)
    elif index_name == 'TR':
        return count_tropical_nights(data, **kwargs)
    elif index_name == 'TXx':
        return max_daily_max_temp(data, **kwargs)
    elif index_name == 'TNx':
        return max_daily_min_temp(data, **kwargs)
    elif index_name == 'TXn':
        return min_daily_max_temp(data, **kwargs)
    elif index_name == 'TNn':
        return min_daily_min_temp(data, **kwargs)
    # Add more elif clauses for other indices
    else:
        raise ValueError(f"Unknown index name: {index_name}")


# Example usage
file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
variable_name = 'highest_one_day_precipitation_amount_per_time_period'
plot_time_custom_function_with_dates(file_path, variable_name, start_year=2019, end_year=2031, season='annual',
                                     index_name='FD', data_type='temp', lon1=24, lat1=22, lat3=32)
