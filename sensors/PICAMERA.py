#implement pi camera
import picamera
import multiprocessing as mp

class PICAMERA:
    def __init__(self):
        self.camera = picamera.PiCamera()
        self.camera.resolution = (256,256)
        self.camera.framerate = 20
    def get_data_camera(self, name):
        self.camera.start_recording(name + '.h264')
