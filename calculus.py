import numpy as np
from scipy import integrate

# Definir una secuencia de valores
values = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
delta_x = 1.0  # Intervalo entre los valores

# Calcular las diferencias finitas (aproximación de la derivada)
derivadas_aprox = np.diff(values) / delta_x

# Calcular la integral numérica utilizando la regla del trapecio
integral_aprox = integrate.trapz(values, dx=delta_x)

# Mostrar los resultados
print("Valores:", values)
print("Diferencias finitas (derivadas aproximadas):", derivadas_aprox)
print("Integral numérica:", integral_aprox)


class Calculus():
    def __init__(self):
        super().__init__(Calculus)
        ##
    def integral(self, f, dt):
        return integrate.trapz(f, dx=dt)
    def derivative(self, f, dt):
        return np.diff(f)/dt
    def gradient(F, dt):
        """
        nabla f = F
        nabla f = ma
        #Ecuacion diferencial parcial#
        obtenemos f
        """
        




