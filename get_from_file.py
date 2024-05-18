# import netCDF4 as nc
# from datetime import datetime, timedelta

# Open the NetCDF file
import netCDF4 as nc
import cftime

# Open the NetCDF file
file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
dataset = nc.Dataset(file_path)
#
# try:
#     # Get the 'time' variable
#     time_variable = dataset.variables['time']
#
#     # Extract the minimum and maximum values
#     min_date = time_variable[:].min()
#     max_date = time_variable[:].max()
#
#     # Define the reference date
#     reference_date = cftime.num2date(0, units=time_variable.units, calendar=time_variable.calendar)
#
#     # Convert numerical dates to datetime objects using cftime
#     min_date_datetime = cftime.num2date(min_date, units=time_variable.units, calendar=time_variable.calendar)
#     max_date_datetime = cftime.num2date(max_date, units=time_variable.units, calendar=time_variable.calendar)
#
#     # Format dates as 'YYYY-MM-DD'
#     min_date_formatted = min_date_datetime.strftime("%Y-%m-%d")
#     max_date_formatted = max_date_datetime.strftime("%Y-%m-%d")
#
#     print(f"Minimum Date: {min_date_formatted}")
#     print(f"Maximum Date: {max_date_formatted}")
#
# finally:
#     # Close the NetCDF file
#     dataset.close()



try:
    # Get the 'time' variable
    time_variable = dataset.variables['time']

    # Extract the minimum and maximum values
    min_date = time_variable[:].min()
    max_date = time_variable[:].max()
    # Convert numerical dates to datetime objects using cftime
    dates = cftime.num2date(time_variable[:], units=time_variable.units, calendar=time_variable.calendar)
    # Convert numerical dates to datetime objects using cftime
    min_date_datetime = cftime.num2date(min_date, units=time_variable.units, calendar=time_variable.calendar)
    max_date_datetime = cftime.num2date(max_date, units=time_variable.units, calendar=time_variable.calendar)
    print(min_date_datetime)
    print(max_date_datetime)
    # Format dates as 'YYYY-MM-DD'
    min_date_formatted = min_date_datetime.strftime("%Y-%m-%d")
    max_date_formatted = max_date_datetime.strftime("%Y-%m-%d")
    min_year_formatted = min_date_datetime.strftime("%Y")
    max_year_formatted = max_date_datetime.strftime("%Y")

    print(f"Minimum year: {min_year_formatted}")
    print(f"Maximum year: {max_year_formatted}")
    print(f"Minimum Date: {min_date_formatted}")
    print(f"Maximum Date: {max_date_formatted}")

    # Print all dates to verify the range
    # c = 0
    # for date in dates:
    #     c += 1
    #     print(date)
    # print(c)


finally:
    # Close the NetCDF file
    dataset.close()
