import requests
import numpy as np
from urllib.parse import urlencode
import itertools

def descargar_archivo(url, destino):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(destino, 'wb') as archivo:
            archivo.write(response.content)
        print(f"Archivo descargado correctamente en {destino}")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar el archivo: {e}")

def parametros(ascent_rate: float, burst_altitude: float, descent_rate: float) -> str:
    parametros = {
        'profile': 'standard_profile',
        'pred_type': 'single',
        'launch_datetime': '2023-12-27T09:00:00Z',
        'launch_latitude': '-25.5226',
        'launch_longitude': '302.6944',
        'launch_altitude': '0',
        'ascent_rate': ascent_rate,
        'burst_altitude': burst_altitude,
        'descent_rate': descent_rate,
        'format': 'csv'
    }
    enlace_base = 'https://api.v2.sondehub.org/tawhiri?'
    enlace = enlace_base + urlencode(parametros)
    return enlace

class Generador:
    def __init__(self, ascend_range, descend_range, altitude_range, samples):
        self.ascend_range = ascend_range
        self.descend_range = descend_range
        self.altitude_range = altitude_range
        self.samples = samples

    def forward(self):
        combinations = list(itertools.product(self.ascend_range, self.descend_range, self.altitude_range))
        
        for ascend, descend, altitude in combinations:
            try:
                enlace = parametros(ascend, altitude, descend)
                destino = f"tocos/{ascend}_{descend}_{altitude}.csv"
                descargar_archivo(enlace, destino)
            except:
                print("Error")

if __name__ == "__main__":
    ascend_range = np.arange(2, 5, 1)
    descend_range = np.arange(3, 7, 1)
    altitude_range = np.arange(20000, 30000,1000)
    samples = 30

    mi_generador = Generador(ascend_range, descend_range, altitude_range, samples)
    mi_generador.forward()