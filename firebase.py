import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
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


bucket = storage.bucket()

path_on_cloud = 'datasets/dataset1/extreme_temp_range_RCP45.nc'
path_local = 'netcdf_files/extreme_temp_range_RCP45.nc'

blob = bucket.blob(path_on_cloud)
blob.upload_from_filename(path_local)

blob.download_to_filename('test.nc')


