import base64

import firebase_admin
import numpy as np
from firebase_admin import db, credentials, storage

import uuid
import netCDF4 as nc
import cftime
from firebase_admin.exceptions import FirebaseError

from flask import Flask, request, jsonify, send_file
from flask_restful import Api, Resource

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
        'start_time': min_date,
        'end_time': max_date,
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


def index_function_by_name(index_name, data, **kwargs):
    if index_name == 'TXx':
        return np.sum(data > 0, **kwargs)


if __name__ == "__main__":
    app.run(debug=True)
