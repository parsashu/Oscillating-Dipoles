"""
This code calculates and visualizes the radiation patterns of a "length oscillating dipole" in free space.
Specifically, it explores the intensity (I) of the radiation field as a function of the distance (r) from the dipole and the angle (theta) relative to the dipole axis. 

Key components of the code:

1. **Dipole Constants:**
   - Charge (q), period (T), angular frequency (w), and initial half-length of the dipole (Z0).

2. **Mathematical Tools:**
   - Numerical differentiation functions for calculating partial derivatives and gradients.

3. **Fixed Point Method:**
   - Function `R_cal` calculates the distance from the dipole to a point in space, considering the retardation effect.

4. **Scalar Potential (Phi):**
   - Function `Phi_charge` computes the scalar potential due to a single charge, and `Phi_tot` computes the total scalar potential from both positive and negative charges.

5. **Vector Potential (A):**
   - Function `Az` calculates the z-component of the vector potential.

6. **Electric and Magnetic Fields:**
   - Functions `E` and `B` compute the electric and magnetic fields, respectively, using the potentials.

7. **Poynting Vector (S):**
   - Function `S` computes the Poynting vector representing the power flux of the electromagnetic field.

8. **Mean Poynting Vector:**
   - Function `mean_S` calculates the time-averaged Poynting vector using the trapezoidal rule for numerical integration.

9. **Theoretical Mean Poynting Vector:**
   - Function `mean_S_theory` provides the theoretical value of the time-averaged Poynting vector for comparison.

10. **Correlation Between Intensity and Distance:**
    - The code generates data points for varying distances (r) while keeping the angle (theta) constant and fits a linear regression model to the log-transformed data.

11. **Correlation Between Intensity and Angle:**
    - The code generates data points for varying angles (theta) while keeping the distance (norm_r) constant and fits a sine-squared model to the data.

12. **Visualization:**
    - The results are plotted to visualize the relationships between intensity and distance, as well as intensity and angle.
"""


import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



# Dipole constants
q = 100 * const.e
T = 1e-7
w = 2 * const.pi / T
Z0 = 1   # Initial half lenght of dipole 

# Math tools--------------------------------------------------------
h = 1e-8

def d_dx(f, r, t):
    r_new = r.copy()
    r_new[0] += h
    df_dx = (f(r_new, t) - f(r, t)) / h
    return df_dx


def d_dy(f, r, t):
    r_new = r.copy()
    r_new[1] += h
    df_dy = (f(r_new, t) - f(r, t)) / h
    return df_dy


def d_dz(f, r, t):
    r_new = r.copy()
    r_new[2] += h
    df_dz = (f(r_new, t) - f(r, t)) / h
    return df_dz


def d_dt(f, r, t):
    df_dt = (f(r, t + h) - f(r, t)) / h
    return df_dt


def gradient(f, r, t):
    df_dx = d_dx(f, r, t)
    df_dy = d_dy(f, r, t)
    df_dz = d_dz(f, r, t) 
    return np.array([df_dx, df_dy, df_dz])


def curl_z(F, r, t):

    dFz_dx, dFz_dy, _ = gradient(F, r, t)
    
    curl_x = dFz_dy
    curl_y = - dFz_dx
    curl_z = 0
    
    return np.array([curl_x, curl_y, curl_z])

#-----------------------------------------------------------------

#Calculation R calligraphic using fixed point method
def R_cal(r, t, charge_sign):
    r = np.linalg.norm(r) 

    def equation(guess):
        return (r**2 + (Z0 * np.cos(w * (t - guess / const.c))) ** 2
                - charge_sign * 2 * r * Z0 * np.cos(w * (t - guess / const.c))) ** 0.5

    first_guess_point = r
    guess = equation(first_guess_point)

    for _ in range(30):
        guess = equation(guess)
    
    return guess


def R_cal_dot_v(r, t, charge_sign):
    r_cal = R_cal(r, t, charge_sign)
    t_r = t - r_cal / const.c
    r_ = np.linalg.norm(r)
    theta = np.arctan((r[0] ** 2 + r[1] ** 2) ** 0.5 / r[2])

    return -charge_sign * Z0 * w * np.sin(w * t_r) * (r_ * np.cos(theta) -
                                                      Z0 * np.cos(w * t_r))



# Calculation scalar potential for single charge
def Phi_charge(r, t, charge_sign):
    r_cal = R_cal(r, t, charge_sign)
    r_cal_dot_v = R_cal_dot_v(r, t, charge_sign)

    return charge_sign * q * const.c / (4 * const.pi * const.epsilon_0 * (r_cal * const.c -
                                                                          r_cal_dot_v))


def Phi_tot(r, t):
    Phi_positive = Phi_charge(r, t, 1)
    Phi_negative = Phi_charge(r, t, -1)
    return Phi_positive + Phi_negative


# Calculation vector potential
def A_integrand(z_prime, r, t):
    d = np.array([0, 0, z_prime])

    r_cal = R_cal(r, d)

    return np.sin(w * (t - r_cal / const.c)) / r_cal


def Az(r, t):
    r_cal_positive = R_cal(r, t, 1)
    r_cal_negative = R_cal(r, t, -1)

    t_r_psositive = t - r_cal_positive /const.c
    t_r_negative = t - r_cal_negative / const.c

    v_positive = - Z0 * w * np.sin(w * t_r_psositive)
    v_negative = + Z0 * w * np.sin(w * t_r_negative)

    phi_positive = Phi_charge(r, t, 1)
    phi_negative = Phi_charge(r, t, -1)

    return const.c ** (-2) * (phi_positive * v_positive + phi_negative * v_negative)



# Calculate Electric field
def E(r, t):
    return - gradient(Phi_tot, r, t) - np.array([0., 0., d_dt(Az, r, t)])


# Calculate Magnetic field
def B(r, t):
    return curl_z(Az, r, t)


# Calculate Poynting vector
def S(r, t):
    e = E(r, t)
    b = B(r, t)

    return (1 / const.mu_0) * np.cross(e, b)


 # Using trapezoidal rule to calculate mean S
def mean_S(r):  
    n = 300
    lower_limit = 0
    upper_limit = T
    delta_t = (upper_limit - lower_limit) / n

    # Calculate the sum of function values
    sum_f = 0
    for i in range(1, n):
        t_i = lower_limit + i * delta_t
        sum_f += S(r, t_i)
        
    # Apply trapezoidal rule formula
    integral_approx = delta_t * (0.5 * (S(r, lower_limit) +
                                          2 * sum_f + S(r, upper_limit)))
    
    return np.linalg.norm(integral_approx) / T


def mean_S_theory(r):
    theta = np.arctan((r[0] ** 2 + r[1] ** 2) ** 0.5 / r[2])

    s = const.mu_0 * (q * 2 * Z0)**2 * w**4 * np.sin(theta)**2 / (32 * const.pi**2 * const.c * np.linalg.norm(r)**2)
    return s



# Evaluation co-relation between I and r by keeping theta constant and changing r
# By multiplying 10 to x, y, z theta stays constant
r_start = np.array([10., 20. ,20.])
r_list = r_start.copy()
data_number = 7

for _ in range(data_number - 1):
    r_start *= 10
    r_list = np.vstack((r_list, r_start))


r_values = np.log(np.linalg.norm(r_list, axis=1))
I_values = np.log(np.array([mean_S(r_) for r_ in r_list]))
I_theory_values = np.log(np.array([mean_S_theory(r_) for r_ in r_list]))


# Perform linear regression using numpy.polyfit
coeff_I = np.polyfit(r_values, I_values, 1)  # Fit for I_values
coeff_I_theory = np.polyfit(r_values, I_theory_values, 1)  # Fit for I_theory_values

# Generate points for fitted lines
r_fit = np.linspace(min(r_values), max(r_values), 100)
I_fit = coeff_I[0] * r_fit + coeff_I[1]
I_theory_fit = coeff_I_theory[0] * r_fit + coeff_I_theory[1]

# Plotting
plt.figure(figsize=(10, 6))

# Plot I_values
plt.scatter(r_values, I_values, label='I_values', color='blue')
plt.plot(r_fit, I_fit, '--', color='blue', label=f'Fit for I_values: slope={coeff_I[0]:.2f}')

# Plot I_theory_values
plt.scatter(r_values, I_theory_values, label='I_theory_values', color='green')
plt.plot(r_fit, I_theory_fit, '--', color='green', label=f'Fit for I_theory_values: slope={coeff_I_theory[0]:.2f}')

plt.xlabel('log(r)')
plt.ylabel('log(Intensity)')
plt.title('co-relation between I and r')
plt.legend()
plt.grid(True)



# Evaluation co-relation between I and theta by keeping norm r constant and changing theta
# Datas of r are on a cricle in y-z plane with radius of norm_r
norm_r = 1000
theta_list = np.linspace(0, 2 * const.pi, 50)
x = 0
r_list2 = np.array([0., 0., 0.])

for theta in theta_list:
    y = norm_r * np.sin(theta)
    z = norm_r * np.cos(theta)
    r_ = np.array([x, y, z])
    r_list2 = np.vstack((r_list2, r_.copy()))

r_list2 = np.delete(r_list2, 0, axis=0) 
I_values2 = np.array([mean_S(r_) for r_ in r_list2])
I_theory_values2 = np.array([mean_S_theory(r_) for r_ in r_list2])


# Define the model function
def sin_squared(theta, a, b):
    return a * np.sin(theta)**2 + b

# Perform curve fitting for I_values2
popt_I, _ = curve_fit(sin_squared, theta_list, I_values2)

# Perform curve fitting for I_theory_values2
popt_I_theory, _ = curve_fit(sin_squared, theta_list, I_theory_values2)

# Generate points for fitted curves
theta_fit = np.linspace(0, 2*np.pi, 100)
I_fit = sin_squared(theta_fit, *popt_I)
I_theory_fit = sin_squared(theta_fit, *popt_I_theory)


# Plotting
plt.figure(figsize=(10, 6))

# Plot I_values2
plt.scatter(theta_list, I_values2, label='I', color='blue')
plt.plot(theta_fit, I_fit, '--', color='blue', label='Fit for I')

# Plot I_theory_values2
plt.scatter(theta_list, I_theory_values2, label='I theory', color='green')
plt.plot(theta_fit, I_theory_fit, '--', color='green', label='Fit for I theory')

plt.xlabel('Theta')
plt.ylabel('Intensity')
plt.title('co-relation between I and theta')
plt.legend()
plt.grid(True)
plt.show()


