import pandas as pd
import numpy as np

# ===============================
# LightImpact Core Module
# ===============================
# This module provides a scientific method to compute the Energy Reduction Value (ERV) for Internal Combustion Vehicles (ICVs) and Electric Vehicles (EVs), based on a WLTP driving cycle input.
# Structure_
# -Inputs:CSV file with time-series driving data (speed [m/s], acceleration [m/s²], time [s]); default or user-defined vehicle/ driving cycle parameters
# -Outputs: ERV for ICV and EV
# ===============================

# Fixed physical constants and parameters
FIXED_PARAMS = {
    "m_ref": 100.0,           # Reference mass [kg]
    "ds_ref": 100.0,          # Reference distance [km]
    "g": 9.81,                # Gravitational acceleration [m/s²]
    "rho_air": 1.2,           # Air density [kg/m³]
    "LHV_fuel": 32.3          # Lower heating value of gasoline [MJ/L]
}

# Case study specific parameters (default parameters, need to be adapted for specific use case)
VEHICLE_PARAMS = {
    "m_vehicle": 1595.0,      # Vehicle mass [kg]
    "f_r": 0.01,              # Rolling resistance coefficient [-]
    "f_Rot": 0.10,            # Rotational mass factor [-]
    "cw": 0.30,               # Drag coefficient [-]
    "A_frontal": 2.07,        # Vehicle frontal area [m²]
    "mu": 0.3                 # Regenerative braking efficiency [-]
}

# WLTP driving cycle specific parameters (WLTP class 3b parameters, need to be adapted for specific driving cycle and class)
WLTP_PARAMS = {
    "ds": 23.26,              # Distance covered by WLTP class 3b [km]
    "roll_pos_factor": 0.55,  # WLTP rolling resistance share during traction [-]
    "c_fuel_WLTP": 10.35,     # Certified fuel consumption [L/100km]
    "c_el_WLTP": 17.9         # Certified electricity consumption [kWh/100km]
}

# Load and preprocess WLTP driving cycle CSV file containing: Time, Speed, Acceleration (in SI units)
def load_wltp(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    required_cols = {"Speed (m/s)", "Acceleration (m/s2)", "Time (s)"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain the following columns: {required_cols}")

    df["v"] = df["Speed (m/s)"].astype(float) # velocity data [m/s]
    df["a"] = df["Acceleration (m/s2)"].astype(float) # acceleration data [m/s2]
    df["dt"] = df["Time (s)"].diff().fillna(1) # calculate time steps dt [s]

    df_tractive = df[(df["a"] > 0) & (df["v"] > 0)].copy() # filter for positve tractive phases (a > 0, v > 0) 
    df_braking = df[(df["a"] < 0) & (df["v"] > 0)].copy() # filter for negative braking phases (a < 0, v > 0)
    return df, df_tractive, df_braking

# ERV calculations
def calculate_erv(df_tractive, df_braking, p):

    # Unpack parameters
    m_ref = p["m_ref"]
    m_vehicle = p["m_vehicle"]
    g = p["g"]
    f_R = p["f_r"]
    f_Rot = p["f_Rot"]
    rho_air = p["rho_air"]
    cw = p["cw"]
    A_frontal = p["A_frontal"]
    mu = p["mu"]
    ds = p["ds"]
    ds_ref = p["ds_ref"]
    roll_pos_factor = p["roll_pos_factor"]
    c_fuel_WLTP = p["c_fuel_WLTP"]
    c_el_WLTP = p["c_el_WLTP"]
    LHV_fuel = p["LHV_fuel"]

    # Compute Mechanical Work ICV (for a standard mass of 100kg and distance of 100km)
    W_R = m_ref * g * f_R * ds * 1000 / 1e6 # Rolling resistance Work [MJ]
    a_v_integral = np.sum(df_tractive["a"] * df_tractive["v"] * df_tractive["dt"]) # integral a*v*dt for positive traction phases 
    W_A = m_ref * a_v_integral / 1e6 # Acceleration Work [MJ]
    W_Rot = m_ref * f_Rot * a_v_integral / 1e6 # Rotational Work [MJ]

    v3_integral_tractive = np.sum(df_tractive["v"]**3 * df_tractive["dt"]) # integral v^3*dt for positive traction phases 
    W_L = 0.5 * rho_air * cw * A_frontal * v3_integral_tractive / 1e6 # Aerodynamic Work [MJ]

    W_mech_ICV_100kg_100km = ((roll_pos_factor * W_R) + W_A + W_Rot) * (ds_ref / ds) # Total mechanical Work ICV (100kg,100km)

    # Compute Mechanical Work EV (for a standard mass of 100kg and distance of 100km)
    W_R_brake = m_ref * g * f_R * ds * 1000 / 1e6
    a_v_brake_integral = np.sum(df_braking["a"] * df_braking["v"] * df_braking["dt"]) # integral a*v*dt for negative braking phases 
    W_A_brake = m_ref * a_v_brake_integral / 1e6 # Regenerative Acceleration Work [MJ]
    W_Rot_brake = m_ref * f_Rot * a_v_brake_integral / 1e6 # Regenerative Rotational Work [MJ]
    v3_integral_brake = np.sum(df_braking["v"]**3 * df_braking["dt"]) #integral v^3*dt for negative braking phases
    W_L_brake = 0.5 * rho_air * cw * A_frontal * v3_integral_brake / 1e6 # Regenerative Aerodynamic Work [MJ]

    E_b = abs(W_R_brake + W_A_brake + W_Rot_brake + W_L_brake)
    phi = E_b / (m_ref * a_v_integral)
    W_A_EV = W_A * (1 - (phi * mu)) # Acceleration Work EV [MJ]

    W_mech_EV_100kg_100km = ((roll_pos_factor * W_R) + W_A_EV + W_Rot) * (ds_ref / ds) # Total mechanical Work EV (100kg,100km)

    # Compute Total Mechanical Work for ICV and EV (total vehicle mass)
    W_R_total = m_vehicle * g * f_R * ds * 1000 / 1e6 # Rolling resistance Work (entire vehicle) [MJ]
    W_A_total = m_vehicle * a_v_integral / 1e6 # Acceleration Work (entire vehicle) [MJ]
    W_Rot_total = m_vehicle * f_Rot * a_v_integral / 1e6 # Rotational Work (entire vehicle) [MJ]

    W_mech_ICV_total_100km = ((roll_pos_factor * W_R_total) + W_A_total + W_Rot_total + W_L) * (ds_ref / ds) # Total mechanical Work ICV (entire vehicle)

    W_R_brake_total = m_vehicle * g * f_R * ds * 1000 / 1e6 # Regenerative Rolling resistance Work (entire vehicle) [MJ]
    W_A_brake_total = m_vehicle * a_v_brake_integral / 1e6 # Regenerative Acceleration Work (entire vehicle) [MJ]
    W_Rot_brake_total = m_vehicle * f_Rot * a_v_brake_integral / 1e6 # Regenerative Rotational Work (entire vehicle) [MJ]

    E_b_total = abs(W_R_brake_total + W_A_brake_total + W_Rot_brake_total + W_L_brake)
    phi_total = E_b_total / (m_vehicle * a_v_integral)
    W_A_EV_total = W_A_total * (1 - (phi_total * mu)) # Acceleration Work EV (entire vehicle) [MJ]

    W_mech_EV_total_100km = ((roll_pos_factor * W_R_total) + W_A_EV_total + W_Rot_total + W_L) * (ds_ref / ds)  # Total mechanical Work EV (entire vehicle)

    # Differential efficiency factors
    mu_diff_ICV = W_mech_ICV_total_100km / (c_fuel_WLTP * LHV_fuel)
    mu_diff_EV = W_mech_EV_total_100km / (c_el_WLTP * 3.6)

    # ERV (for a standard mass of 100kg and distance of 100km)
    ERV_ICV = W_mech_ICV_100kg_100km / (mu_diff_ICV * LHV_fuel)
    ERV_EV = W_mech_EV_100kg_100km / mu_diff_EV

    return ERV_ICV, ERV_EV, mu_diff_ICV, mu_diff_EV

# Run and print results
def run_lightimpact_case(filepath, custom_params=None):
    p = {**FIXED_PARAMS, **VEHICLE_PARAMS, **WLTP_PARAMS}
    if custom_params:
        p.update(custom_params)

    df, df_tractive, df_braking = load_wltp(filepath)
    erv_icv, erv_ev, mu_icv, mu_ev = calculate_erv(df_tractive, df_braking, p)

    total_ICV_work = erv_icv * mu_icv * p["LHV_fuel"]
    total_EV_work = erv_ev * mu_ev

    print(f"Total mechanical work (ICV, 100kg, 100km): {total_ICV_work:.4f} MJ")
    print(f"Total mechanical work (EV, 100kg, 100km):  {total_EV_work:.4f} MJ")
    print(f"Differential efficiency factor (ICV) [µdiff]: {mu_icv:.4f}")
    print(f"Differential efficiency factor (EV) [µdiff]:  {mu_ev:.4f}")
    print(f"ERV (ICV): {erv_icv:.4f} L/(100km·100kg)")
    print(f"ERV (EV):  {erv_ev:.4f} kWh/(100km·100kg)")

# Run the module directly
if __name__ == "__main__":
    run_lightimpact_case("WLTP data/Time-Speed-Profile_WLTP_Class3b.csv")

