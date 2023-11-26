#implement pi camera
import picamera
import multiprocessing as mp
import os
import shutil
class PICAMERA:
    def __init__(self):
        self.camera = picamera.PiCamera()
        self.camera.resolution = (256,256)
        self.camera.framerate = 20
        try:
            os.makedirs('data/video')
        except FileExistsError:
            shutil.rmtree('data/video')
            os.makedirs('data/video')
    def get_data_camera(self, name):
        self.camera.capture('data/video/' + str(name) + '.jpg')
