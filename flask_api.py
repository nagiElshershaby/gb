import io

from flask import Flask, request, jsonify, send_file
from flask_restful import Api, Resource

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

app = Flask(__name__)

api = Api(app)

class IndexesGenerator(Resource):
    def get(self, index):
        return {"index": index}
    def post(self):
        return {"data": "posted"}

api.add_resource(IndexesGenerator, "/getIndex/<string:index>")

@app.route("/get-users/<user_id>")
def get_user(user_id):
    user_data = {
        "user_id": user_id,
        "name": "user_name"
    }
    extra = request.args.get("extra")
    if extra:
        user_data["extra"] = extra
    return jsonify(user_data), 200

@app.route("/create-user", methods= ["POST"])
def create_user():
    if request.method == "POST":
        data = request.get_json()

        return jsonify(data), 201

def count_positive(data, **kwargs):
    return np.sum(data > 0, **kwargs)


























import firebase_admin
from firebase_admin import credentials
# from firebase_admin import db
from firebase_admin import storage

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

# # save data
# ref = db.reference('test/')
# users_ref = ref.child('users')
# # users_ref.set({
# #         'user1':{
# #             'email':'email1@gmail.com',
# #             'password':'password1'
# #         },
# #         'user2':{
# #             'email':'email2@gmail.com',
# #             'password':'password2'
# #         },
# # })
#
# # update data
# hopper_ref = users_ref.child('user1')
# hopper_ref.update({
#     'nickname': 'user1 nickname'
# })
#
# # raad data
# handle = db.reference('test/users/user1')
#
# print(handle.get())
# print(ref.get())










@app.route('/plot_custom_function', methods=['POST'])
def plot_time_custom_function():
    # Parse request parameters
    # file_path = request.form.get('file_path')

    ###############################################################################################################

    bucket = storage.bucket()

    path_on_cloud = 'datasets/admin/10317a6b-e764-42b9-b968-69e87a8d868b'
    path_local = 'netcdf_files/extreme_temp_range_RCP45.nc'

    blob = bucket.blob(path_on_cloud)
    # blob.upload_from_filename(path_local)

    blob.download_to_filename('C:/Users/Prof/PycharmProjects/gb/netcdf_files/test1.nc')

    ###############################################################################################################

    file_path = 'test1.nc'
    variable_name = request.form.get('variable_name')
    start_time = int(request.form.get('start_time', 0))
    end_time = int(request.form.get('end_time', 30))

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
            aggregated_data = count_positive(data[start_time:end_time], axis=0)
        elif len(data.shape) == 2:  # 2D array (latitude, longitude)
            aggregated_data = data  # Apply custom function directly
        else:
            raise ValueError("Unsupported data shape. Expected 2D or 3D data.")

        # Plotting
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Define custom colormap from dark blue to light blue
        colors = [(1, 1, 1), (0.8, 0.9, 1), (0.6, 0.8, 1), (0.3, 0.6, 1), (0, 0.3, 1), (0, 0, 0.5)] # blues
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

        # Save plot to a BytesIO object
        output = io.BytesIO()
        plt.savefig(output, format='png')
        output.seek(0)  # Move the cursor to the beginning of the BytesIO object

        # Return the plot file content
        return send_file(output, mimetype='image/png')

        # # Save plot to a temporary file
        # plot_filename = 'temp_plot.png'
        # plt.savefig(plot_filename)
        #
        # # Close plot to free up memory
        # plt.close(fig)
        #
        # # Return the plot file path
        # return jsonify({'plot_file_path': plot_filename})

    finally:
        # Close the NetCDF file
        dataset.close()

@app.route('/upload/<user_id>/<variable_name>/<start_time>/<end_time>/<custom_function>', methods=['POST'])
def upload_file(user_id, variable_name, start_time, end_time, custom_function):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # plot_time_custom_function(file_path=file, variable_name= variable_name, start_time= int(start_time), end_time=int(end_time))

    # Return the processed data as JSON
    user_data = {
        "user_id": user_id,
        "variable_name": variable_name,
        "start_time": start_time,
        "end_time": end_time,
        "custom_function": custom_function,
        "name": "user_name"
    }
    return jsonify(user_data), 200


if __name__ == "__main__":
    app.run(debug=True)





def plot_time_custom_function(file_path, variable_name, start_time=0, end_time=30, custom_function=count_positive):
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