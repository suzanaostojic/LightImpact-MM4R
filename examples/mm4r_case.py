import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lightimpact.core import run_lightimpact_case

custom_params = {
    "m_vehicle": 1595.0,
    "f_r": 0.01,
    "f_Rot": 0.10,
    "cw": 0.30,
    "A_frontal": 2.07,
    "mu": 0.3,
    "ds": 23.26,
    "roll_pos_factor": 0.55,
    "c_fuel_WLTP": 10.35,
    "c_el_WLTP": 17.9
}

run_lightimpact_case("WLTP_data/Time-Speed-Profile_WLTP_Class3b.csv", custom_params)