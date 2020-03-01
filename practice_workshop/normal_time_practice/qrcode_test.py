import qrcode,pyrebase
from PIL import Image

# initialize our running app
config_running_app = {
    "apiKey": "AIzaSyCjQCwrJHZKAOiNG04ycZnhtIQpGTw5yh4",
    "authDomain": "running-mate-7bb1b.firebaseapp.com",
    "databaseURL": "https://running-mate-7bb1b.firebaseio.com",
    "projectId": "running-mate-7bb1b",
    "storageBucket": "running-mate-7bb1b.appspot.com",
    "messagingSenderId": "610614312598"
}

running_app_firebase = pyrebase.initialize_app(config_running_app)
running_app_db = running_app_firebase.database()
storage = running_app_firebase.storage()

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data('Some data')
qr.make(fit=True)

qr = qrcode.QRCode(
    version = 1,
    error_correction = qrcode.constants.ERROR_CORRECT_H,
    box_size = 10,
    border = 4,
)

# The data that you want to store
data = {
    "gift" : "乳清系列95折"
}

# Add data
qr.add_data(data)
qr.make(fit=True)

# Create an image from the QR Code instance
img = qr.make_image()

# Save it somewhere, change the extension as needed:
img.save("test.png")

