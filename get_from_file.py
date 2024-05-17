import netCDF4 as nc
from datetime import datetime, timedelta

# Open the NetCDF file
import netCDF4 as nc
import cftime

# Open the NetCDF file
file_path = 'netcdf_files/ndays_maxt.gt.25_RCP85.nc'
dataset = nc.Dataset(file_path)

try:
    # Get the 'time' variable
    time_variable = dataset.variables['time']

    # Extract the minimum and maximum values
    min_date = time_variable[:].min()
    max_date = time_variable[:].max()

    # Define the reference date
    reference_date = cftime.num2date(0, units=time_variable.units, calendar=time_variable.calendar)

    # Convert numerical dates to datetime objects using cftime
    min_date_datetime = cftime.num2date(min_date, units=time_variable.units, calendar=time_variable.calendar)
    max_date_datetime = cftime.num2date(max_date, units=time_variable.units, calendar=time_variable.calendar)

    # Format dates as 'YYYY-MM-DD'
    min_date_formatted = min_date_datetime.strftime("%Y-%m-%d")
    max_date_formatted = max_date_datetime.strftime("%Y-%m-%d")

    print(f"Minimum Date: {min_date_formatted}")
    print(f"Maximum Date: {max_date_formatted}")

finally:
    # Close the NetCDF file
    dataset.close()
