#implement pi camera
import picamera

class PICAMERA:
    def get_data_camera(self, name):
        with picamera.PiCamera() as camera:
            camera.resolution = (512,512)
            camera.iso = 1200
            camera.shutter_speed = 1000  
            camera.exposure_mode = 'night' 
            camera.capture(f'/home/pi/Desktop/NIPPA_AEP/{self.date}/video/' + str(name) + '.jpg')
            camera.close()
