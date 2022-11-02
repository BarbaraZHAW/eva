# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem


# Physical properties
# ===================
wall = {'Conductivity': [1.400, 0.040],
        'Density': [2300.0, 16.0],
        'Specific heat': [880, 1210],
        'Width': [0.2, 0.08],
        'Meshes': [4, 2]}

wall = pd.DataFrame(wall, index=['Concrete', 'Insulation'])
wall

air = {'Density': 1.2,
       'Specific heat': 1000}

pd.DataFrame(air, index=['Air'])

h = pd.DataFrame([{'in': 4., 'out': 10.}], index=['h'])
h

S_wall = 3 * 3      # m², wall surface area
V_air = 3 * 3 * 3   # m³, indoor air volume

# conduction
R_cd = wall['Width'] / (wall['Conductivity'] * S_wall)  # K/W

# convection
R_cv = 1 / (h * S_wall)     # K/W
C_wall = wall['Density'] * wall['Specific heat'] * wall['Width'] * S_wall
C_air = air['Density'] * air['Specific heat'] * V_air

# number of temperature nodes and flow branches
no_θ = no_q = sum(wall['Meshes']) + 1
# Conductance matrix
R = np.zeros([no_q])
R[0] = R_cv['out'] + R_cd['Concrete'] / 8
R[1] = R[2] = R[3] = R_cd['Concrete'] / 4
R[4] = R_cd['Concrete'] / 8 + R_cd['Insulation'] / 4
R[5] = R_cd['Insulation'] / 2
R[6] = R_cd['Insulation'] / 4 + R_cv['in']
G = np.diag(np.reciprocal(R))

C = np.zeros(no_θ)
C[0] = C[1] = C[2] = C[3] = C_wall['Concrete'] / 4
C[4] = C[5] = C_wall['Insulation'] / 2
C[6] = C_air
C = np.diag(C)


A = np.eye(no_q, no_θ + 1)
A = -np.diff(A, n=1, axis=1)
pd.DataFrame(A)
b = np.zeros(no_q)
f = np.zeros(no_θ)
b[0] = 1
θ_steady_To = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
np.set_printoptions(precision=3)
print('When To = 1°C, the temperatures in steady-state are:', θ_steady_To, '°C')
print(f'The indoor temperature is: {θ_steady_To[-1]:.3f} °C')

b[0] = 0
f[-1] = 1
θ_steady_Qh = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
print('When Qh = 1W, the temperatures in steady-state are:', θ_steady_Qh, '°C')
print(f'The indoor temperature is: {θ_steady_Qh[-1]:.3f} °C')

# State matrix
As = -np.linalg.inv(C) @ A.T @ G @ A
pd.set_option('display.precision', 1)
pd.DataFrame(As)

Bs = np.linalg.inv(C) @ np.block([A.T @ G, np.eye(no_θ)])
pd.set_option('display.precision', 2)
pd.DataFrame(Bs)

# Select columns for which the input vector is not zero
# 1st for To and last for Qh
Bs = Bs[:, [0, -1]]
pd.DataFrame(Bs, columns=['To', 'Qh'])

# Output matrix
Cs = np.zeros((1, no_θ))
# output: last temperature node
Cs[:, -1] = 1

# Feedthrough (or feedforward) matrix
Ds = np.zeros(Bs.shape[1])

λ = np.linalg.eig(As)[0]    # minimum eigenvalue of matrix A
max_Δt = min(-2 / λ)

np.set_printoptions(precision=1)
print('Time constants: \n', -1 / λ, 's \n')
print('2 x Time constants: \n', -2 / λ, 's \n')
print(f'Max time step Δt = {max_Δt:.2f} s')

Δt = 360
#Δt = 420.5
print(f'Δt = {Δt} s')

filename = './weather_data/FRA_Lyon.074810_IWEC.epw'
start_date = '2000-04-10'
end_date = '2000-05-15'

[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data

weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather[(
    weather.index >= start_date) & (
    weather.index < end_date)]
pd.DataFrame(weather)

days = weather.shape[0] / 24
days
# number of steps
n = int(np.floor(3600 / Δt * 2 * days))
n
# time vector
t = np.arange(0, n * Δt, Δt)
pd.DataFrame(t, columns=['time'])
u = np.block([[np.ones([1, n])],    # To = [1, 1, ... , 1]
              [np.zeros([1, n])]])  # Qh = [0, 0, ... , 0]
pd.DataFrame(u)

# initial values for temperatures obtained by explicit and implicit Euler
θ_exp = np.zeros([no_θ, t.shape[0]])
θ_imp = np.zeros([no_θ, t.shape[0]])

for k in range(t.shape[0] - 1):
    θ_exp[:, k + 1] = (np.eye(no_θ) + Δt * As) @\
        θ_exp[:, k] + Δt * Bs @ u[:, k]
    θ_imp[:, k + 1] = np.linalg.inv(np.eye(no_θ) - Δt * As) @\
        (θ_imp[:, k] + Δt * Bs @ u[:, k])

fig, ax = plt.subplots()
ax.plot(t / 3600, θ_exp[-1, :], t / 3600, θ_imp[-1, :])
ax.set(xlabel='Time [h]', ylabel='Air temperature [°C]', title='Step input: $T_o$')
ax.legend(['Explicit', 'Implicit'])
plt.show()

u = np.block([[np.zeros([1, n])],   # To = [0, 0, ... , 0]
              [np.ones([1, n])]])   # Qh = [1, 1, ... , 1]
pd.DataFrame(u)

θ_exp = np.zeros([no_θ, t.shape[0]])
θ_imp = np.zeros([no_θ, t.shape[0]])

for k in range(t.shape[0] - 1):
    θ_exp[:, k + 1] = (np.eye(no_θ) + Δt * As) @\
        θ_exp[:, k] + Δt * Bs @ u[:, k]
    θ_imp[:, k + 1] = np.linalg.inv(np.eye(no_θ) - Δt * As) @\
        (θ_imp[:, k] + Δt * Bs @ u[:, k])
        
        
fig, ax = plt.subplots()
ax.plot(t / 3600, θ_exp[-1, :], t / 3600, θ_imp[-1, :])
ax.set(xlabel='Time [h]', ylabel='Air temperature [°C]', title='Step input: $Q_h$')
ax.legend(['Explicit', 'Implicit'])
plt.show() 

tw = np.arange(0, 3600 * weather.shape[0], 3600)
pd.DataFrame(tw)

t = np.arange(0, 3600 * weather.shape[0], Δt)

# outdoor temperature at timestep Δt
θ_out = np.interp(t, tw, weather['temp_air'])
pd.DataFrame(θ_out, index=t, columns=['θ °C'])   

u = np.block([[θ_out],
             [np.zeros(θ_out.shape[0])]])
pd.DataFrame(u, index=['To', 'Qh'])  

θ_exp = np.zeros([no_θ, t.shape[0]])
θ_imp = np.zeros([no_θ, t.shape[0]])
for k in range(u.shape[1] - 1):
    θ_exp[:, k + 1] = (np.eye(no_θ) + Δt * As) @\
        θ_exp[:, k] + Δt * Bs @ u[:, k]
    θ_imp[:, k + 1] = np.linalg.inv(np.eye(no_θ) - Δt * As) @\
        (θ_imp[:, k] + Δt * Bs @ u[:, k])
        
fig, ax = plt.subplots()
ax.plot(t / 3600 / 24, θ_exp[-1, :], label='Indoor temperature')
ax.plot(t / 3600 / 24, θ_out, label='Outdoor temperature')
ax.set(xlabel='Time [days]',
       ylabel='Air temperature [°C]',
       title='Explicit Euler')
ax.legend()
plt.show()