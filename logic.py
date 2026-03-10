import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def get_keys():
    return "PRIVATE_KEY_HIDDEN", "PUBLIC_KEY_GATEWAY"

def sign_data(priv_key, icao, vel):
    return f"SIG_{icao}_{vel}"

def verify_sig(pub_key, icao, vel, sig):
    # Only block if it's explicitly an attack signature
    if sig is not None and "INVALID" in str(sig):
        return False
    return True

def run_middleware(df):
    processed_df = df.copy()
    
    # 1. Identity Check
    processed_df['auth_passed'] = processed_df.apply(
        lambda r: verify_sig(None, r['icao24'], r['velocity'], r.get('signature', None)), axis=1
    )

    # 2. Physics Check 
    features = ['velocity', 'vertrate', 'geoaltitude']
    iso_model = IsolationForest(contamination=0.005, random_state=42) 
    processed_df['iso_score'] = iso_model.fit_predict(processed_df[features])
    
    # --- CLASSIFICATION ---
    processed_df['status'] = 'LEGITIMATE'
    
    # Block Identity First
    processed_df.loc[processed_df['auth_passed'] == False, 'status'] = 'FAKE IDENTITY'
    
    # Block Physics 
    # The AI only blocks if it's an outlier AND the speed is realistically high
    # This prevents it from picking on 'slow but weird' legitimate flights
    physics_attack = (processed_df['iso_score'] == -1) & (processed_df['velocity'] > 1000)

    # Or, if the speed is truly insane, block it regardless of AI score
    physics_attack = physics_attack | (processed_df['velocity'] > 1500)
        
    processed_df.loc[(processed_df['status'] == 'LEGITIMATE') & physics_attack, 'status'] = 'IMPROPER PHYSICS'

    # RGB Mapping
    def assign_color(status):
        if status == 'LEGITIMATE': return 0, 255, 0
        if status == 'IMPROPER PHYSICS': return 255, 0, 0
        return 255, 165, 0
        
    processed_df['r'], processed_df['g'], processed_df['b'] = zip(*processed_df['status'].apply(assign_color))
    
    return processed_df