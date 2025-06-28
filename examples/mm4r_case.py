import pandas as pd
import numpy as np

# Load and Prepare WLTP data
# The WLTP dataset must include: 'Time','Speed','Acceleration' 
df = pd.read_csv("WLTP data/Time-Speed-Profile_WLTP_Class3b.csv") 
df["v"] = df["Speed (km/h)"] # velocity data [m/s]
df["a"] = df["Acceleration (m/s2)"] # acceleration data [m/s2]
df["dt"] = df["Time (s)"].diff().fillna(1) # calculate time steps dt [s]

df_tractive = df[(df["a"] > 0) & (df["v"] > 0)].copy() # filter for positve tractive phases (a > 0, v > 0) 
df_braking = df[(df["a"] < 0) & (df["v"] > 0)].copy() # filter for negative braking phases (a < 0, v > 0)

# Define Constants and Parameters
m_ref             = 100.0        # standard reference mass [kg] used for calculation
m_vehicle         = 1595         # curb weight of Porsche 911 vehicle model [kg]
ds_ref            = 100.0        # standard reference distance [km] used for calculation
g                 = 9.81         # gravitational acceleration [m/s2]
f_R               = 0.01         # rolling-resistance coefficient [-]
f_Rot             = 0.10         # rotational mass factor [-]
rho_air           = 1.2          # air density [kg/m3]
cw                = 0.30         # drag coefficient [-]
A_frontal         = 2.07         # vehicle frontal area [m²]
ds                = 23.26        # distance covered by WLTC class 3b [km]
roll_pos_factor   = 0.55         # share of rolling resistance during positive-tractive phase [-]
mu                = 0.30         # regenerative-braking efficiency [-]
c_fuel_WLTP       = 10.35        # certified fuel consumption [L/100km]
c_el_WLTP         = 17.9         # certified electricity consumption [kWh/100km]
LHV_fuel          = 32.3         # lower heating value gasoline [MJ/L]

# Compute Mechanical Work Integrals ICV 
W_R = m_ref * g * f_R * ds * 1000 / 1e6  # Rolling resistance Work [MJ]

a_v_integral = np.sum(df_tractive["a"] * df_tractive["v"] * df_tractive["dt"]) # integral a*v*dt for positive traction phases 
W_A = m_ref * a_v_integral / 1e6 # Acceleration Work [MJ]
W_Rot = m_ref * f_Rot * a_v_integral / 1e6 # Rotational Work [MJ]

v3_integral_tractive = np.sum(df_tractive["v"]**3 * df_tractive["dt"]) # integral v^3*dt for positive traction phases 
W_L = 0.5 * rho_air * cw * A_frontal * v3_integral_tractive / 1e6 # Aerodynamic Work [MJ]

W_R_total = m_vehicle * g * f_R * ds * 1000 / 1e6 # Rolling resistance Work (entire vehicle) [MJ]
W_A_total = m_vehicle * a_v_integral / 1e6 # Acceleration Work (entire vehicle) [MJ]
W_Rot_total = m_vehicle * f_Rot * a_v_integral / 1e6 # Rotational Work (entire vehicle) [MJ]

# Compute Mechanical Work Integrals EV
W_R_brake = m_ref * g * f_R * ds * 1000 / 1e6 # Rolling resistance Work [MJ]

a_v_brake_integral = np.sum(df_braking["a"] * df_braking["v"] * df_braking["dt"]) # integral a*v*dt for negative braking phases 
W_A_brake = m_ref * a_v_brake_integral / 1e6 # Regenerative Acceleration Work [MJ]
W_Rot_brake = m_ref * f_Rot * a_v_brake_integral / 1e6 # Regenerative Rotational Work [MJ]

v3_integral_brake = np.sum(df_braking["v"]**3 * df_braking["dt"]) #integral v^3*dt for negative braking phases
W_L_brake = 0.5 * rho_air * cw * A_frontal * v3_integral_brake / 1e6 # Regenerative Aerodynamic Work [MJ]

W_R_brake_total = m_vehicle * g * f_R * ds * 1000 / 1e6 # Regenerative Rolling resistance Work (entire vehicle) [MJ]
W_A_brake_total = m_vehicle * a_v_brake_integral / 1e6 # Regenerative Acceleration Work (entire vehicle) [MJ]
W_Rot_brake_total = m_vehicle * f_Rot * a_v_brake_integral / 1e6 # Regenerative Rotational Work (entire vehicle) [MJ]

# Total mechanical Work ICV (100kg,100km)
W_mech_ICV_100kg_100km = ((roll_pos_factor * W_R) + W_A + W_Rot) * (ds_ref / ds)
print(f"Total mechanical work (ICV, 100kg, 100km): {W_mech_ICV_100kg_100km:.4f} MJ")

# Total mechanical Work EV (100kg,100km)
E_b = abs(W_R_brake + W_A_brake + W_Rot_brake + W_L_brake)
denominator = m_ref * np.sum(df_tractive["a"] * df_tractive["v"] * df_tractive["dt"])
phi = E_b / denominator # braking energy to kinetic energy ratio

W_A_EV = W_A * (1 - (phi * mu))
W_mech_EV_100kg_100km = ((roll_pos_factor * W_R) + W_A_EV + W_Rot) * (ds_ref / ds)
print(f"Total mechanical work (EV, 100kg, 100km):  {W_mech_EV_100kg_100km:.4f} MJ")

# Total mechanical Work ICV (vehicle mass, 100km)
W_mech_ICV_total_100km = ((roll_pos_factor * W_R_total) + W_A_total + W_Rot_total + W_L) * (ds_ref / ds)

# Total mechanical Work EV (vehicle mass, 100km)
E_b_total = abs(W_R_brake_total + W_A_brake_total + W_Rot_brake_total + W_L_brake)
denominator_total = m_vehicle * np.sum(df_tractive["a"] * df_tractive["v"] * df_tractive["dt"])
phi_total = E_b_total / denominator_total

W_A_EV_total = W_A_total * (1 - (phi_total * mu))
W_mech_EV_total_100km = ((roll_pos_factor * W_R_total) + W_A_EV_total + W_Rot_total + W_L) * (ds_ref / ds)

# Differential efficiency factors 
mu_diff_ICV = W_mech_ICV_total_100km / (c_fuel_WLTP * LHV_fuel)
mu_diff_EV = W_mech_EV_total_100km / (c_el_WLTP * 3.6)  # convert kWh to MJ
print(f"Differential efficiency factor (ICV): {mu_diff_ICV:.4f}")
print(f"Differential efficiency factor (EV):  {mu_diff_EV:.4f}")

# ERV calculations
ERV_ICV = W_mech_ICV_100kg_100km / (mu_diff_ICV * LHV_fuel)  # [L/(100km*100kg)]
ERV_EV = W_mech_EV_100kg_100km / mu_diff_EV  # [kWh/(100km*100kg)]
print(f"ERV (ICV): {ERV_ICV:.4f} L/(100km·100kg)")
print(f"ERV (EV):  {ERV_EV:.4f} kWh/(100km·100kg)")