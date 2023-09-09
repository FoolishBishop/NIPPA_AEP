# IMPORTANT

If you want to include **new sensors**, create a file with an object class representing the sensor, this class should include these methods and constructors:

```python
def __init__(self):
    ##CONFIGURATION VARIABLES
    self.i2c = board.I2C()   # uses board.SCL and board.SDA
    self.<sensor_name>_queue = mp.Queue()
    ...
def get_data_<sensor_name>(self):
    #gets the data from sensor
    return <data>
def display_data_<sensor_name>(self): #optional
    #print functions to display data
```
