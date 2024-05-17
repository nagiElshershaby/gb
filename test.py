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



def test_plot_custom_function():
    # Define the URL of the Flask API endpoint
    url = 'http://127.0.0.1:5000/plot_custom_function'

    # Define request parameters
    params = {
            'file_path': 'netcdf_files/extreme_temp_range_RCP45.nc',
        'variable_name': 'temp',
        'start_time': 0,
        'end_time': 30
    }

    # Send a POST request to the endpoint
    response = requests.post(url, data=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract the image content from the response
        image_data = response.content

        # Create an image object from the image data
        image = Image.open(io.BytesIO(image_data))

        # Save the image to a file (optional)
        image.save('plot.png')

        # Display the image (optional)
        image.show()

    else:
        print("Error:", response.text)


if __name__ == "__main__":
    test_plot_custom_function()

