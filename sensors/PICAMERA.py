#implement pi camera
import picamera
import multiprocessing as mp

class PICAMERA:
    def get_data_camera(self, name):
        with picamera.PiCamera() as camera:
            camera.resolution = (512,512)
            camera.capture(f'/home/pi/Desktop/NIPPA_AEP/{self.date}/video/' + str(name) + '.jpg')
            camera.close()
