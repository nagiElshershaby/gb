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


def plot_time_custom_function_with_dates(file_path, variable_name, start_date=None, end_date=None,
                                         index_name='TXx', dataType='temp', lon1=None, lat1=None,
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

        if len(data.shape) == 3:
            if start_date is None and end_date is None:
                aggregated_data = index_function_by_name(index_name, data, axis=0)
            elif start_date is None:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                end_idx = nc.date2index(end_date_obj, dataset.variables['time'], select='nearest')
                aggregated_data = index_function_by_name(index_name, data[:end_idx + 1], axis=0)
            elif end_date is None:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                start_idx = nc.date2index(start_date_obj, dataset.variables['time'], select='nearest')
                aggregated_data = index_function_by_name(index_name, data[start_idx:], axis=0)
            else:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                start_idx = nc.date2index(start_date_obj, dataset.variables['time'], select='nearest')
                end_idx = nc.date2index(end_date_obj, dataset.variables['time'], select='nearest')
                aggregated_data = index_function_by_name(index_name, data[start_idx:end_idx + 1], axis=0)
        elif len(data.shape) == 2:
            aggregated_data = data
        else:
            raise ValueError("Unsupported data shape. Expected 2D or 3D data.")

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cmap_name = get_color_mapping_according_to_type(dataType)
        levels, cmap = create_color_levels(aggregated_data, cmap_name)
        contour = plt.contourf(xlon, xlat, aggregated_data, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
        ax.coastlines()
        cbar = plt.colorbar(contour, orientation='horizontal')
        num_ticks = 10
        ticks = np.linspace(aggregated_data.min(), aggregated_data.max(), num=num_ticks)
        cbar.set_ticks(ticks)
        cbar.set_label(var_name)
        if start_date is None and end_date is None:
            plt.title(f"{variable_name}")
        elif start_date is None:
            plt.title(f"{variable_name} until {end_date}")
        elif end_date is None:
            plt.title(f"{variable_name} from {start_date}")
        else:
            plt.title(f"{variable_name} from {start_date} to {end_date}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Calculate extent dynamically if no coordinates are specified
        if any(coord is None for coord in [lon1, lat1, lon3, lat3]):
            extent = [xlon.min(), xlon.max(), xlat.min(), xlat.max()]
        else:
            polygon = create_polygon(lon1, lat1, lon3, lat1, lon3, lat3, lon1, lat3)
            ax.add_geometries([polygon], ccrs.PlateCarree(), facecolor='none', alpha=0.3)
            # Adjust limits slightly larger than the square for better visualization
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
    if t == 'temp' or 'temperature':
        return 'plasma'
    elif t == 'PR' or 'pr':
        return 'viridis'
    elif t == 'HW' or 'hw':
        return 'magma'
    elif t == 'CW' or 'cw':
        return 'inferno'
    else:
        return 'plasma'


def index_function_by_name(index_name, data, **kwargs):
    if index_name == 'TXx':
        return np.sum(data > 0, **kwargs)


# Example usage
file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
variable_name = 'highest_one_day_precipitation_amount_per_time_period'
plot_time_custom_function_with_dates(file_path, variable_name, start_date=None, end_date='2030-01-01',
                                     index_name='TXx', dataType='hw', lon1=24, lat1=22,
                                     lon3=36, lat3=32)
