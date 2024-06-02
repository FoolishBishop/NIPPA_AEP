# IMPORTANT

If you want to include **new sensors**, create a file with an object class representing the sensor, this class should include these methods and constructors:

```python
class SampleSensor:
    def __init__(self): 
        ##CONFIGURATION VARIABLES
        ...
    def get_data_<sensor_name>(self):
        #gets the data from sensor
        self.queue.put(<data>)
    def display_data_<sensor_name>(self): #optional
        #print functions to display data
```
