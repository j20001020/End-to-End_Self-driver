from keras.models import load_model
import sys
import numpy as np

import airsim
import time

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
print('Connection established!')

def get_image():
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    
    return image_rgba[76:135,0:255,0:3].astype(float)

# initialize car status
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

image_buf = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1,4))
    
while (True):
    car_state = client.getCarState()
    
    if (car_state.speed < 5):
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0
    
    image_buf[0] = get_image()
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
    model_output = model.predict([image_buf, state_buf])
    car_controls.steering = round(0.5 * float(model_output[0][0]), 2)
    
    print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))
    
    client.setCarControls(car_controls)