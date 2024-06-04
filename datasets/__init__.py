import base64
import io

import firebase_admin
from firebase_admin import db, credentials, storage
from firebase_admin.exceptions import FirebaseError

import numpy as np
import uuid
import netCDF4 as nc
import cftime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime

from flask import Flask, request, jsonify
from flask_restful import Api

from shapely.geometry import Polygon

app = Flask(__name__)

api = Api(app)

# Fetch the service account key Json file content
cred = credentials.Certificate('graduation-beef5-firebase-adminsdk-afqzq-dd4a2248ce.json')

# initialize the app with a service account, granting admin privileges

config = {
    "apiKey": "AIzaSyAW0rmbkCChp9Q305VcRtoQvgWlY4e8VG0",
    "authDomain": "graduation-beef5.firebaseapp.com",
    "databaseURL": "https://graduation-beef5-default-rtdb.firebaseio.com",
    "projectId": "graduation-beef5",
    "storageBucket": "graduation-beef5.appspot.com",
    "messagingSenderId": "446252346321",
    "appId": "1:446252346321:web:ce8d1203fc81cee1d20ef0",
    "measurementId": "G-3V65JWN3GJ"
}
firebase_admin.initialize_app(cred, config)


def get_min_max_dates(file_path):
    """
    Extracts the minimum and maximum dates from a NetCDF file.

    Args:
    - file_path (str): The path to the NetCDF file.

    Returns:
    - min_date (str): The minimum date in 'YYYY-MM-DD' format.
    - max_date (str): The maximum date in 'YYYY-MM-DD' format.
    # Example usage
        file_path = 'netcdf_files/ndays_maxt.gt.25_RCP85.nc'
        min_date, max_date = get_min_max_dates(file_path)
        print(f"Minimum Date: {min_date}")
        print(f"Maximum Date: {max_date}")
    """
    dataset = nc.Dataset(file_path)

    try:
        # Get the 'time' variable
        time_variable = dataset.variables['time']

        # Extract the minimum and maximum values
        min_date = time_variable[:].min()
        max_date = time_variable[:].max()

        # Convert numerical dates to datetime objects using cftime
        min_date_datetime = cftime.num2date(min_date, units=time_variable.units, calendar=time_variable.calendar)
        max_date_datetime = cftime.num2date(max_date, units=time_variable.units, calendar=time_variable.calendar)

        # Format dates as 'YYYY-MM-DD'
        min_date_formatted = min_date_datetime.strftime("%Y-%m-%d")
        max_date_formatted = max_date_datetime.strftime("%Y-%m-%d")

        return min_date_formatted, max_date_formatted

    finally:
        # Close the NetCDF file
        dataset.close()


def get_available_indexes(t):
    if t == 'temp':
        return [
            'FD', 'SU', 'ID', 'TR', 'GSL', 'TXx', 'TNx', 'TXn', 'TNn', 'TN10p', 'TX10p',
            'TN90p', 'TX90p', 'WSDI', 'CSDI', 'DTR', 'ETR', 'CDDcoldn', 'GDDgrown',
            'HDDheatn', 'TMge5', 'TMlt5', 'TMge10', 'TMlt10', 'TMm', 'TXm', 'TNm',
            'TXge30', 'TXge35', 'TXgt50p', 'TNlt2', 'TNltm2', 'TNltm20', 'TXbdTNbd'
        ]
    elif t == 'pr':
        return ['Rx1day', 'Rx5day', 'SPI', 'SPEI', 'SDII', 'R10mm', 'R20mm', 'Rnnmm', 'CDD', 'CWD', 'R95p', 'R99p',
                'R95pTOT', 'R99pTOT', 'PRCPTOT']
    elif t == 'lone':
        return []
    else:
        return ['HWN', 'HWF', 'HWD', 'HWM', 'HWA', 'CWN_ECF', 'CWF_ECF', 'CWD_ECF', 'CWM_ECF', 'CWA_ECF']


def create_dataset(path_local, name, type, access, view, description, var_name):
    available_indexes = get_available_indexes(type)
    # Extracts the minimum and maximum dates from a NetCDF file.
    min_date, max_date = get_min_max_dates(path_local)

    dataset_id = str(uuid.uuid4())
    file_path_on_cloud = f'datasets/{access}/{dataset_id}'
    return {
        'id': dataset_id,
        'name': name,
        'file_path_on_cloud': file_path_on_cloud,
        'type': type,
        'description': description,
        'available_indexes': available_indexes,
        'var_name': var_name,
        'min_date': min_date,
        'max_date': max_date,
        'access': access,
        'view': view
    }


def send_dataset_to_firebase(dataset, path_local):
    bucket = storage.bucket()
    path_on_cloud = dataset['file_path_on_cloud']
    blob = bucket.blob(path_on_cloud)
    blob.upload_from_filename(path_local)
    ref = db.reference(f'datasets/{dataset["access"]}/{dataset["id"]}')
    ref.set(dataset)


def upload_dataset(path_local, name, type, access, view, description, var_name):
    dataset = create_dataset(path_local, name, type, access, view, description, var_name)
    send_dataset_to_firebase(dataset, path_local)


@app.route('/upload_dataset', methods=['POST'])
def upload_dataset_endpoint():
    # Get data from the request
    data = request.get_json()

    # Extract parameters from the data
    path_local = data.get('path_local')
    name = data.get('name')
    type = data.get('type')
    access = data.get('access')
    view = data.get('view')
    description = data.get('description')
    var_name = data.get('var_name')

    # Upload the dataset
    upload_dataset(path_local, name, type, access, view, description, var_name)

    return jsonify({"message": "Dataset uploaded successfully"})


@app.route('/update_dataset/<access>/<dataset_id>', methods=['PUT'])
def update_dataset(access, dataset_id):
    try:
        # Parse the request JSON data
        updated_data = request.get_json()

        # Update the dataset in Firebase
        ref = db.reference(f'datasets/{access}/{dataset_id}')
        ref.update(updated_data)

        return jsonify({'message': 'Dataset updated successfully'}), 200
    except FirebaseError as e:
        error_message = f'Error updating dataset: {str(e)}'
        return jsonify({'error': error_message}), 500


@app.route('/delete_dataset/<access>/<dataset_id>', methods=['DELETE'])
def delete_dataset(access, dataset_id):
    try:
        ref = db.reference(f'datasets/{access}/{dataset_id}')
        ref.delete()
        return f'Dataset with ID {dataset_id} deleted successfully from access level {access}.', 200
    except FirebaseError as e:
        error_message = f'Error deleting dataset: {str(e)}'
        return jsonify({'error': error_message}), 500


@app.route('/retrieve_all_datasets/<access>', methods=['GET'])
def retrieve_all_datasets(access):
    try:
        ref = db.reference(f'datasets/{access}')
        datasets_snapshot = ref.get()
        datasets = []
        if datasets_snapshot:
            for dataset_id, dataset_snapshot in datasets_snapshot.items():
                datasets.append(dataset_snapshot)
        return jsonify(datasets), 200
    except FirebaseError as e:
        error_message = f'Error retrieving datasets: {str(e)}'
        return jsonify({'error': error_message}), 500


def create_user(username, email, access):
    return {
        'username': username,
        'email': email,
        'access': access
    }


# Function to encode email address
def encode_email(email):
    return email.replace('.', ',').replace('@', '_at_')


def decode_email(encoded_email):
    return encoded_email.replace(',', '.').replace('_at_', '@')


def add_new_user(username, email, access):
    encoded_email = encode_email(email)
    user = create_user(username, email, access)
    ref = db.reference(f'users/{encoded_email}')
    ref.set(user)


@app.route('/add_new_user', methods=['POST'])
def add_new_user_endpoint():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    access = data.get('access')

    if not username or not email or not access:
        return jsonify({'error': 'Missing required fields (username, email, access)'}), 400

    try:
        add_new_user(username, email, access)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/update_user/<email>', methods=['PUT'])
def update_user_endpoint(email):
    encoded_email = encode_email(email)
    data = request.json
    username = data.get('username')
    access = data.get('access')

    if not username and not access:
        return jsonify({'error': 'Nothing to update. Provide at least one field (username or access)'}), 400

    try:
        ref = db.reference(f'users/{encoded_email}')
        user = ref.get()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        if username:
            user['username'] = username
        if access:
            user['access'] = access

        ref.update(user)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete_user/<email>', methods=['DELETE'])
def delete_user_endpoint(email):
    encoded_email = encode_email(email)
    try:
        ref = db.reference(f'users/{encoded_email}')
        user = ref.get()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        ref.delete()
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_user/<email>', methods=['GET'])
def get_user_endpoint(email):
    encoded_email = encode_email(email)
    try:
        ref = db.reference(f'users/{encoded_email}')
        user = ref.get()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify(user), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_all_users', methods=['GET'])
def get_all_users_endpoint():
    try:
        ref = db.reference(f'users')
        users = ref.get()
        if not users:
            return jsonify({'error': 'No Users found'}), 404

        return jsonify(users), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

        # Handle different dimensions
        if len(data.shape) == 2:
            aggregated_data = data
        elif len(data.shape) >= 3:
            if start_date and end_date:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                start_idx = nc.date2index(start_date_obj, time_variable, select='nearest')
                end_idx = nc.date2index(end_date_obj, time_variable, select='nearest')
                sliced_data = data[start_idx:end_idx + 1]
            elif start_year and end_year and season:
                filtered_data, filtered_dates = filter_data_by_season_and_year(data, time_variable, season, start_year,
                                                                               end_year)
                sliced_data = filtered_data
            else:
                sliced_data = data

            # Aggregate data along the time axis (assuming the time axis is the first dimension)
            if len(sliced_data.shape) == 3:
                aggregated_data = index_function_by_name(index_name, sliced_data, axis=0)
            else:
                aggregated_data = index_function_by_name(index_name, sliced_data, axis=0)
        else:
            raise ValueError("Unsupported data shape. Expected 2D or 3D+ data.")

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cmap_name = get_color_mapping_according_to_type(data_type)
        levels, cmap = create_color_levels(aggregated_data, cmap_name)
        # Ensure levels is sorted
        if levels is not None:
            levels = sorted(levels)
            # Contour levels must be increasing
            if levels[0] > levels[-1]:
                levels = levels[::-1]
            elif levels[0] == levels[-1]:
                levels = np.linspace(levels[0], levels[0] + 1, num=10)
            # print(f"Sorted levels: {levels}")
        else:
            levels = np.linspace(np.min(aggregated_data), np.max(aggregated_data), num=10)
            # print(f"Default levels: {levels}")
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

        # Save the plot to a BytesIO object and encode it as a base64 string
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    finally:
        dataset.close()


@app.route('/plot_local', methods=['POST'])
def plot_endpoint():
    # this function will be called when the endpoint is hit
    request_data = request.get_json()

    file_path = request_data.get('file_path')
    variable_name = request_data.get('var_name')
    start_date = request_data.get('start_date')
    end_date = request_data.get('end_date')
    start_year = request_data.get('start_year')
    end_year = request_data.get('end_year')
    season = request_data.get('season', 'annual')
    index_name = request_data.get('index_name', 'TXx')
    data_type = request_data.get('data_type', 'temp')
    lon1 = request_data.get('lon1')
    lat1 = request_data.get('lat1')
    lon3 = request_data.get('lon3')
    lat3 = request_data.get('lat3')

    try:
        img_str = plot_time_custom_function_with_dates(
            file_path, variable_name, start_date, end_date,
            start_year, end_year, season, index_name, data_type,
            lon1, lat1, lon3, lat3
        )
        return jsonify({"image": img_str}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_firebase', methods=['POST'])
def plot_firebase_endpoint():
    request_data = request.get_json()

    dataset_id = request_data.get('dataset_id')
    access = request_data.get('access')
    start_date = request_data.get('start_date')
    end_date = request_data.get('end_date')
    start_year = request_data.get('start_year')
    end_year = request_data.get('end_year')
    season = request_data.get('season', 'annual')
    index_name = request_data.get('index_name', 'TXx')
    lon1 = request_data.get('lon1')
    lat1 = request_data.get('lat1')
    lon3 = request_data.get('lon3')
    lat3 = request_data.get('lat3')

    ref = db.reference(f'datasets/{access}/{dataset_id}')
    dataset = ref.get()

    path_on_cloud = dataset['file_path_on_cloud']
    data_type = dataset['type']
    variable_name = dataset['var_name']

    file_name = f"{dataset['name']}.nc"
    bucket = storage.bucket()
    blob = bucket.blob(path_on_cloud)
    blob.download_to_filename(file_name)

    try:
        img_str = plot_time_custom_function_with_dates(
            file_name, variable_name, start_date, end_date,
            start_year, end_year, season, index_name, data_type,
            lon1, lat1, lon3, lat3
        )
        return jsonify({"image": img_str}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
