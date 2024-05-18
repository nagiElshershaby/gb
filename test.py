# import numpy as np
import requests
from PIL import Image
import io

# BASE = "http://127.0.0.1:5000/"
#
#
# def count_positive(data, **kwargs):
#     return np.sum(data > 0, **kwargs)
#
# response = requests.get(BASE + "get-users/<user_id>")
# print(response.json())


# BASE_URL = "http://127.0.0.1:5000/upload"
#
# def upload_file(filename,variable_name,start_time,end_time,custom_function):
#     with open(filename, 'rb') as file:
#         files = {'file': (filename, file)}
#         response = requests.post(BASE_URL + f"/<user_id>/{variable_name}/{start_time}/{end_time}/{custom_function}", files=files)
#         return response.json()
#
# if __name__ == "__main__":
#     filename = "netcdf_files/max_pr_monthly_RCP45.nc"  # Update with the path to your file
#     variable_name = "var name"
#     start_time = "0"
#     end_time = "30"
#     custom_function = "TXx"
#     result = upload_file(filename,variable_name,start_time,end_time,custom_function)
#     print("Response:", result)


#
# def test_plot_custom_function():
#     # Define the URL of the Flask API endpoint
#     url = 'http://127.0.0.1:5000/plot_custom_function'
#
#     # Define request parameters
#     params = {
#             'file_path': 'netcdf_files/extreme_temp_range_RCP45.nc',
#         'variable_name': 'temp',
#         'start_time': 0,
#         'end_time': 30
#     }
#
#     # Send a POST request to the endpoint
#     response = requests.post(url, data=params)
#
#     # Check if the request was successful (status code 200)
#     if response.status_code == 200:
#         # Extract the image content from the response
#         image_data = response.content
#
#         # Create an image object from the image data
#         image = Image.open(io.BytesIO(image_data))
#
#         # Save the image to a file (optional)
#         image.save('plot.png')
#
#         # Display the image (optional)
#         image.show()
#
#     else:
#         print("Error:", response.text)
#
#
# if __name__ == "__main__":
#     test_plot_custom_function()


# import requests
# import json
# import base64
# import matplotlib.pyplot as plt
# import io
#
# from cftime import Datetime360Day
#
#
# def test_plot_endpoint():
#     url = 'http://127.0.0.1:5000/plot_firebase'
#     headers = {'Content-Type': 'application/json'}
#
#     # Define the request payload
#     # payload = {
#     #     "file_path": "C:/Users/Prof/PycharmProjects/gb/netcdf_files/max_pr_monthly_RCP45.nc",
#     #     "variable_name": "highest_one_day_precipitation_amount_per_time_period",
#     #     "start_year": 2019,
#     #     "end_year": 2035,
#     #     "season": "annual",
#     #     "index_name": "TXx",
#     #     "data_type": "hw",
#     #     "lon1": 24,
#     #     "lat1": 22,
#     #     # "lon3": 36,
#     #     "lat3": 32
#     # }
#     payload = {
#         "dataset_id": "10317a6b-e764-42b9-b968-69e87a8d868b",
#         "access": "admin",
#         "start_year": 2019,
#         "end_year": 2035,
#         "season": "annual",
#         "index_name": "TXx",
#         "lon1": 24,
#         "lat1": 22,
#         "lon3": 36,
#         "lat3": 32
#     }
#
#     # Send a POST request to the endpoint
#     response = requests.post(url, headers=headers, data=json.dumps(payload))
#
#     # Print the response status code and content for debugging
#     print("Response status code:", response.status_code)
#     print("Response content:", response.content)
#
#     # Check if the request was successful (status code 200)
#     assert response.status_code == 200
#
#     # Decode the image string from the response JSON
#     img_str = response.json()['image']
#     img_data = base64.b64decode(img_str)
#
#     # Create a BytesIO object to read the image data
#     img_buf = io.BytesIO(img_data)
#
#     # Display the image
#     img = plt.imread(img_buf)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
#
# if __name__ == '__main__':
#     test_plot_endpoint()


import requests
import json
import base64
import matplotlib.pyplot as plt
import io

def test_plot_endpoint():
    url = 'http://127.0.0.1:5000/plot_firebase'
    headers = {'Content-Type': 'application/json'}

    # Define the request payload
    payload = {
        "dataset_id": "27a51e58-5b79-45e7-a31a-d02b55973cc7",
        "access": "admin",
        "start_year": 2019,
        "end_year": 2035,
        "season": "annual",
        # "start_date": "2027-01-01",
        # "end_date": "2035-12-31",
        "index_name": "TXx",
        "lon1": 24,
        "lat1": 22,
        # "lon3": 36,
        "lat3": 32
    }

    # Send a POST request to the endpoint
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Print the response content and status code for debugging
    print("Response Status Code:", response.status_code)
    print("Response Content:", response.content)

    # Check if the request was successful (status code 200)
    assert response.status_code == 200

    # Decode the image string from the response JSON
    img_str = response.json()['image']
    img_data = base64.b64decode(img_str)

    # Create a BytesIO object to read the image data
    img_buf = io.BytesIO(img_data)

    # Display the image
    img = plt.imread(img_buf)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    test_plot_endpoint()
