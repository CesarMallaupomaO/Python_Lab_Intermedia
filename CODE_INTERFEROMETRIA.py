# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:38:20 2023

@author: César
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


################################DATOS######################################
# Datos de entrada
d_um = np.array([25*2, 50*2, 75*2, 100*2, 125*2, 150*2, 175*2])
N_transiciones = np.array([70, 138, 198, 257, 330, 407, 482])

##########################################################################


# Definición de la función de ajuste 
def funcion_de_ajuste(x, a, b):
    return a * x + b

# Realizar el ajuste de curva
params, covariance = curve_fit(funcion_de_ajuste, N_transiciones, d_um, sigma=np.full(len(d_um),0.5))

# Obtener los parámetros ajustados
a, b = params

# Calcular los valores ajustados
d_um_ajustados = funcion_de_ajuste(N_transiciones, a, b)

# Calcular el error para d_um
error_d_um = d_um - d_um_ajustados

# Graficar los datos y la curva ajustada con barras de error
plt.errorbar(N_transiciones, d_um, yerr=error_d_um, fmt='o', label="Datos con error", color="black")
plt.plot(N_transiciones, d_um_ajustados, label="Ajuste", color="black")
plt.ylabel('d(um)')
plt.xlabel('N(transiciones)')

# Agregar la ecuación ajustada y las incertidumbres 
eq_str = f'Ecuación: d = {a:.2f} * N + {b:.2f}'
uncertainty_str = f'Incertidumbre (d): {np.sqrt(covariance[0, 0]):.2f}'
plt.text(300, 150, eq_str, fontsize=8)
plt.text(300, 140, uncertainty_str, fontsize=8)

plt.legend()
plt.show()

# Imprimir los parámetros del ajuste
print(f'Parámetro "a": {a}')
print(f'Parámetro "b": {b}')


#%%


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Datos de entrada
P_mmHg = np.array([10, 20, 30, 40])
N_s = np.array([4, 7, 8, 11])

incertidumbre_barometro = 0.5
############
N = np.array([1.0 + j*0.633*10**(-6)/(2*0.03) for j in N_s])
############



# Definición de la función de ajuste (puedes cambiarla según el tipo de ajuste que desees)
def funcion_de_ajuste(x, a, b):
    return a * x + b

# Realizar el ajuste de curva
params, covariance = curve_fit(funcion_de_ajuste, P_mmHg, N, sigma=np.full(len(P_mmHg), incertidumbre_barometro))

# Obtener los parámetros ajustados
a, b = params

# Calcular los valores ajustados
N_ajustados = funcion_de_ajuste(P_mmHg, a, b)

# Calcular el error
error = N - N_ajustados

# Graficar los datos y la curva ajustada con barras de error
plt.errorbar(P_mmHg, N, yerr=error, xerr=incertidumbre_barometro, fmt='o', label="Datos con error")
plt.plot(P_mmHg, N_ajustados, label="Ajuste", color='black')
plt.xlabel('P(mmHg)')
plt.ylabel('n(Índice de refracción)')

# Agregar la ecuación ajustada y las incertidumbres en la gráfica
eq_str = f'Ecuación: n = {a:.2f} * P + {b:.2f}'
uncertainty_str = f'Incertidumbre (a): {np.sqrt(covariance[0, 0]):.2f}'



plt.legend()
plt.show()

# Imprimir los parámetros del ajuste
print(f'Parámetro "a": {a}')
ç
print(f'Parámetro "b": {b}')

#%%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Definir la función lineal para el ajuste
def linear_function(x, m, b):
    return m * x + b

# Datos del tercer experimento del índice de refracción del vidrio
N3 = np.array([29.0, 94.0, 205.0, 380.0, 527.0])
z = np.array([5.0, 10.0, 15.0, 25.0, 30.0])

t = 0.5 * 10**(-2)  # Espesor del vidrio
lon = 643 * 10**(-9)  # Longitud de onda hallada anteriormente

X2 = np.zeros(5, float)
for i in range(5):
    X2[i] = 2 * t * (1 - np.cos(z[i] * np.pi / 180)) - N3[i] * lon

Y2 = np.zeros(5, float)
for i in range(5):
    Y2[i] = (2 * t - N3[i] * lon) * (1 - np.cos(z[i] * np.pi / 180))

# Realizar el ajuste de curva
params3, cov_matrix3 = curve_fit(linear_function, X2, Y2)
m3_opt, b3_opt = params3
m3_err, b3_err = np.sqrt(np.diag(cov_matrix3))

# Calcular los valores ajustados
nv_pred = linear_function(X2, m3_opt, b3_opt)
print("El valor de la pendiente es", m3_opt)
print("El valor del término independiente es", b3_opt)

# Graficar los datos y la curva ajustada
plt.plot(X2, Y2, "o", label='Datos', color = "black")
plt.plot(X2, nv_pred, label='Ajuste',color = "black")
equation3 = f'y = {m3_opt:.2f}x + {b3_opt:.2e}'
errors3 = f'$\Delta$ m: {m3_err:.2f}\n$\Delta$ b: {b3_err:.2e}'
plt.text(0.00004, 0.0009, equation3, fontsize=12)  # Posición de la ecuación

plt.xlabel('$f_\phi$')#$2t(1-\cos(\phi))-N\lambda$
plt.ylabel('$g_\phi$')#$(2t-N\lambda)(1-\cos(\phi))$
plt.legend()
plt.show()