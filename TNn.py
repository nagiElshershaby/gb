import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cftime
from io import BytesIO
import base64
from PIL import Image
import io


def prepare_data(file_path):
    dataset = nc.Dataset(file_path)
    time_var = dataset.variables['time']
    time_units = time_var.units
    time_calendar = time_var.calendar if 'calendar' in time_var.ncattrs() else 'standard'
    dates = nc.num2date(time_var[:], units=time_units, calendar=time_calendar)

    # Manually convert cftime objects to pandas datetime objects while handling the 360-day calendar
    def cftime_to_datetime(cftime_obj):
        if isinstance(cftime_obj, cftime.Datetime360Day):
            year = cftime_obj.year
            month = cftime_obj.month
            day = min(cftime_obj.day, 28)  # Adjust days greater than 28 to avoid invalid dates
            return pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
        else:
            return pd.Timestamp(cftime_obj.strftime('%Y-%m-%d'))

    dates = pd.Series(dates).apply(cftime_to_datetime)

    # Ensure dates are recognized as datetime-like by pandas
    dates = pd.to_datetime(dates)

    lon = dataset.variables['xlon'][:]
    lat = dataset.variables['xlat'][:]
    target_variable = dataset.variables['highest_one_day_precipitation_amount_per_time_period'][:]

    dataset.close()

    return dates, time_var, time_units, lon, lat, target_variable


def get_fd(file_path):
    # Extract the daily minimum temperature variable
    dates, time_var, time_units, lon, lat, daily_min_temp = prepare_data(file_path)
    # Calculate the FD index
    fd = np.sum(daily_min_temp < 0, axis=0)

    # Extract latitude and longitude data
    # Plot the FD index on a map
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add coastlines and borders
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    levels = np.arange(0, np.max(fd) + 2, 1)
    cmap = plt.cm.get_cmap('Blues', len(levels))
    # Plot the FD index using color-filled contours
    cp = ax.contourf(lon, lat, fd, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
    # Add a colorbar
    cbar = plt.colorbar(cp, orientation='vertical', pad=0.05, aspect=30)
    cbar.set_label('Number of Frost Days')
    # Add title and labels
    plt.title('Number of Frost Days')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Show the plot
    plt.show()


def get_su(file_path):
    dates, time_var, time_units, lon, lat, daily_min_temp = prepare_data(file_path)
    sd = np.sum(daily_min_temp > 25, axis=0)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    levels = np.arange(0, np.max(sd) + 2, 1)
    cmap = plt.cm.get_cmap('coolwarm', len(levels))
    # Plot the FD index using color-filled contours
    cp = ax.contourf(lon, lat, sd, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
    # Add a colorbar
    cbar = plt.colorbar(cp, orientation='vertical', pad=0.05, aspect=30)
    cbar.set_label('Number of Summer Days')
    # Add title and labels
    plt.title('Number of Summer Days')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plot_base64 = generate_plot(plt)
    return plot_base64


def get_id(file_path):
    dates,time_var, time_units, lon, lat, daily_max_temp = prepare_data(file_path)
    # Calculate the number of icing days
    icing_days = np.sum(daily_max_temp < 0, axis=0)
    plt.figure(figsize=(10, 5))
    plt.contourf(lon, lat, icing_days, cmap='cool')
    plt.colorbar(label='Number of Icing Days')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Number of Icing Days')
    plot_base64 = generate_plot(plt)
    return plot_base64


def get_tr(file_path):
    dates, time_var, time_units, lon, lat, daily_min_temp = prepare_data(file_path)

    # Calculate the TR index
    tr = np.sum(daily_min_temp > 20, axis=0)

    # Plot the TR index
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    levels = np.arange(0, np.max(tr) + 2, 1)
    cmap = plt.cm.get_cmap('coolwarm', len(levels))
    # Plot the TR index using color-filled contours
    cp = ax.contourf(lon, lat, tr, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cp, orientation='vertical', pad=0.05, aspect=30)
    cbar.set_label('Number of Tropical Nights')
    plt.title('Number of Tropical Nights')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plot_base64 = generate_plot(plt)
    return plot_base64


def calculate_gsl(daily_mean_temp, dates, hemisphere='NH'):
    # Define the threshold temperature
    tg_threshold = 5.0
    gsl = np.full((daily_mean_temp.shape[1], daily_mean_temp.shape[2]), np.nan)

    for y in range(daily_mean_temp.shape[1]):
        for x in range(daily_mean_temp.shape[2]):
            temp_series = daily_mean_temp[:, y, x]

            # Find the first span of at least 6 consecutive days with TG > 5°C
            above_5 = (temp_series > tg_threshold).astype(int)
            first_6_day_span_start = np.where(np.convolve(above_5, np.ones(6, dtype=int), 'valid') == 6)[0]

            if len(first_6_day_span_start) == 0:
                continue  # No growing season found

            start_index = first_6_day_span_start[0]

            # Determine the hemisphere
            if hemisphere == 'NH':
                after_july_1st = dates >= pd.Timestamp(f'{dates[start_index].year}-07-01')
            else:
                after_jan_1st = dates >= pd.Timestamp(f'{dates[start_index].year}-01-01')

            after_july_1st_or_jan_1st = after_july_1st if hemisphere == 'NH' else after_jan_1st

            # Find the first span after July 1st (or January 1st in SH) of at least 6 consecutive days with TG < 5°C
            below_5_after_july_1st = (temp_series[after_july_1st_or_jan_1st] < tg_threshold).astype(int)
            first_6_day_span_end = np.where(np.convolve(below_5_after_july_1st, np.ones(6, dtype=int), 'valid') == 6)[0]

            if len(first_6_day_span_end) == 0:
                continue  # No ending found

            end_index = first_6_day_span_end[0] + np.where(after_july_1st_or_jan_1st)[0][0]

            gsl[y, x] = end_index - start_index

    return gsl


def get_gsl(file_path, hemisphere='NH'):
    dates, time_var, time_units, lon, lat, daily_mean_temp = prepare_data(file_path)

    # Calculate the GSL index
    gsl = calculate_gsl(daily_mean_temp, dates, hemisphere)

    # Plot the GSL index
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    levels = np.arange(0, np.nanmax(gsl) + 10, 10)
    cmap = plt.cm.get_cmap('coolwarm', len(levels))
    cp = ax.contourf(lon, lat, gsl, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cp, orientation='vertical', pad=0.05, aspect=30)
    cbar.set_label('Growing Season Length (days)')
    plt.title('Growing Season Length')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plot_base64 = generate_plot(plt)
    return plot_base64


def get_tnn(file_path):
    dates, time_var,time_units, lon, lat, daily_min_temp = prepare_data(file_path)
    time_calendar = dates.calendar if 'calendar' in dates.ncattrs() else 'standard'
    dates = nc.num2date(dates[:], units=time_units, calendar=time_calendar)

    # Manually convert cftime objects to pandas datetime objects while handling the 360-day calendar
    # def cftime_to_datetime(cftime_obj):
    #     if isinstance(cftime_obj, cftime.Datetime360Day):
    #         year = cftime_obj.year
    #         month = cftime_obj.month
    #         day = min(cftime_obj.day, 28)  # Adjust days greater than 28 to avoid invalid dates
    #         return pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
    #     else:
    #         return pd.Timestamp(cftime_obj.strftime('%Y-%m-%d'))
    #
    # dates = pd.Series(dates).apply(cftime_to_datetime)
    #
    # # Ensure dates are recognized as datetime-like by pandas
    # dates = pd.to_datetime(dates)

    # Extract the daily maximum precipitation variable
    # daily_min_temp = dataset.variables['highest_one_day_precipitation_amount_per_time_period'][:]

    # Get the shape of the data for reshaping
    time_len, y_len, x_len = daily_min_temp.shape

    # Convert the dates to a pandas DataFrame to easily group by month
    dates_df = pd.DataFrame({'date': dates})
    dates_df['month'] = dates_df['date'].dt.to_period('M')

    # Initialize an array to store the minimum temperatures for each month
    tnn = np.full((y_len, x_len), np.nan)

    # Loop over each unique month and calculate the overall minimum precipitation
    for month in dates_df['month'].unique():
        # Get the indices for the current month
        month_indices = dates_df[dates_df['month'] == month].index
        # Calculate the minimum temperature for the current month across the time dimension
        tnn_month = np.min(daily_min_temp[month_indices, :, :], axis=0)
        # Update the overall minimum temperature array
        tnn = np.nanmin(np.dstack((tnn, tnn_month)), axis=2)

    # Extract latitude and longitude data
    xlon = lon
    xlat = lat

    # Plot the overall TNn
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Define the color levels and plot the data
    levels = np.arange(np.nanmin(tnn), np.nanmax(tnn) + 1, 1)
    cp = ax.contourf(xlon, xlat, tnn, levels=levels, cmap='viridis', transform=ccrs.PlateCarree())
    plt.colorbar(cp, orientation='vertical', pad=0.05, aspect=50, label='TNn (K)')

    # Add title and labels
    plt.title('Overall TNn')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def generate_plot(plt):  # Optional data argument
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.clf()  # Clear the plot for future usage
    return plot_base64


file_path = 'netcdf_files/max_pr_monthly_RCP45.nc'
base64_image_string = get_tnn(file_path)
decoded_image = base64.b64decode(base64_image_string)

# Convert the decoded image to a PIL Image object
image = Image.open(io.BytesIO(decoded_image))

# Display the image using Matplotlib
plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.show()
