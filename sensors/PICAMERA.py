#implement pi camera
import picamera
import multiprocessing as mp
import os
import shutil
class PICAMERA:
    def get_data_camera(self, name):
        with picamera.PiCamera() as camera:
            camera.resolution = (512,512)
            camera.capture(f'{self.time}/video/' + str(name) + '.jpg')
            camera.close()
