import numpy as np
import matplotlib.pyplot as plt

#Constants of Circuit
C1 = 5.57e-9 #F
C2 = 21.32e-9 #F
L = 12e-3 #H
R = 1.5e3 #Ohms
R0 = 30.86 #Ohm
Ga = (-1)*0.879e-3 #S
Gb = (-1)*0.4124e-3 #S
E = 1 #V

#step size (decrease for better approximation)
h = 0.001
#number of steps (increase see behavior over longer time period)
num_steps = 100

#Characteristic equations of the circuit
def dV1dt(V1, V2):
    """Given instantanous V1, V2, and f(V1), returns the instantaneous change in V1"""
    return (1/C1) * (((V2 - V1)/R) - f(V1))

def dV2dt(V1, V2, I3):
    """Given instantanous V1, V2, and i3, returns the instantaneous change in V2"""
    return (1/C2) * (((V1 - V2)/R) - I3)

def dI3dt(V2, I3):
    """Given instantaneous V2 and I3, returns the instantaneous change in I3"""
    return -(1/L) * (V2 + R0 * I3)

def f(V):
    """Given a voltage, returns the VI characteristic of Chua's Diode"""
    return Gb * V + 0.5 * (Ga - Gb) * (np.abs(V + E) - np.abs(V - E))

#Approximate slopes over finite step, 4th order Runge-Kutta mehtod
def m(V1, V2, I3):
    """Given current V1, V2, and I3, returns approximate change in V1, V2, I3, over finite step (as a tuple)"""
    k1V1 = dV1dt(V1, V2)
    k1V2 = dV2dt(V1, V2, I3)
    k1I3 = dI3dt(V2, I3)

    k2V1 = dV1dt(V1 + k1V1*(h/2), V2 + k1V2*(h/2))
    k2V2 = dV2dt(V1 + k1V1*(h/2), V2 + k1V2*(h/2), I3 + k1I3*(h/2))
    k2I3 = dI3dt(V2 + k1V2*(h/2), I3 + k1I3*(h/2))

    k3V1 = dV1dt(V1 + k2V1*(h/2), V2 + k2V2*(h/2))
    k3V2 = dV2dt(V1 + k2V1*(h/2), V2 + k2V2*(h/2), I3 + k2I3*(h/2))
    k3I3 = dI3dt(V2 + k2V2*(h/2), I3 + k2I3*(h/2))

    k4V1 = dV1dt(V1 + k3V1*h, V2 + k3V2*h)
    k4V2 = dV2dt(V1 + k3V1*h, V2 + k3V2*h, I3 + k3I3*h)
    k4I3 = dI3dt(V2 + k3V2*h, I3 + k3I3*h)

    mV1 = k1V1/6 + k2V1/3 + k3V1/3  + k4V1/6
    mV2 = k1V2/6 + k2V2/3 + k3V2/3  + k4V2/6
    mI3 = k1I3/6 + k2I3/3 + k3I3/3  + k4I3/6

    return mV1, mV2, mI3

V1_over_time = np.empty(num_steps)
V2_over_time = np.empty(num_steps)
I3_over_time = np.empty(num_steps)

V1, V2, I3 = -10e-6, 1.3, 0.001

for i in range(num_steps):
    #Find approximate slopes
    mV1, mV2, mI3 = m(V1, V2, I3)
    #Update V1, V2, I3
    V1 = V1 + mV1 * h
    V2 = V2 + mV2 * h
    I3 = I3 + mI3 * h
    #Add new values to np.arrays
    V1_over_time[i], V2_over_time[i], I3_over_time[i] = V1, V2, I3

#print(V1_over_time, V2_over_time, I3_over_time)
 
#plt.plot(V1_over_time, V2_over_time)
#plt.show()

test_V1 = np.linspace(-5,5, 100)
plt.plot(test_V1, f(test_V1))
plt.title("v-i Characteristic of Chura's Diode")
plt.xlabel('V1 = V across diode')
plt.ylabel('f(V1) = Resistance of Diode')
textstr = "Ga: {}s\nGb: {}s\nE: {}V".format(Ga, Gb, E)
plt.annotate(textstr, xy=(0.75,0.85), xycoords= 'axes fraction')
plt.show()