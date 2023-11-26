#implement pi camera
import picamera
import multiprocessing as mp
import os
import shutil
class PICAMERA:
    def __init__(self):
        try:
            os.makedirs('data/video')
        except FileExistsError:
            shutil.rmtree('data/video')
            os.makedirs('data/video')
    def get_data_camera(self, name):
        with picamera.PiCamera() as camera:
            camera.resolution = (256,256)
            camera.framerate = 20
            camera.capture('data/video/' + str(name) + '.jpg')
            camera.close()
