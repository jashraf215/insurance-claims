import pandas as pd
import numpy as np


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Basic cleaning
    df.drop(columns=['_c39'], inplace=True, errors='ignore')
    df.replace(['?', 'None'], np.nan, inplace=True)
    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

    # Drop unnecessary columns
    drop_cols = ['police_report_available', 'property_damage',
                 'policy_bind_date', 'policy_number']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Impute missing
    for col in ['collision_type', 'authorities_contacted']:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Drop sparse categorical features
    sparse = ['incident_date', 'incident_location', 'incident_state',
              'incident_city', 'auto_model', 'policy_state',
              'umbrella_limit', 'insured_hobbies']
    df.drop(columns=sparse, inplace=True, errors='ignore')

    # Simplify auto_make and insured_occupation
    auto_make_map = {
        'BMW': 'Germany', 'Mercedes': 'Germany', 'Audi': 'Germany',
        'Volkswagen': 'Germany', 'Toyota': 'Japan', 'Honda': 'Japan',
        'Nissan': 'Japan', 'Suburu': 'Japan', 'Accura': 'Japan',
        'Ford': 'USA', 'Chevrolet': 'USA', 'Dodge': 'USA',
        'Jeep': 'USA', 'Saab': 'Other'
    }

    insured_occupation_map = {
        'exec-managerial': 'White Collar', 'prof-specialty': 'White Collar',
        'tech-support': 'White Collar', 'adm-clerical': 'White Collar',
        'craft-repair': 'Blue Collar', 'machine-op-inspct': 'Blue Collar',
        'transport-moving': 'Blue Collar', 'handlers-cleaners': 'Blue Collar',
        'farming-fishing': 'Blue Collar', 'sales': 'Service',
        'other-service': 'Service', 'priv-house-serv': 'Service',
        'protective-serv': 'Service', 'armed-forces': 'Military'
    }

    if 'auto_make' in df.columns:
        df['auto_make'] = df['auto_make'].map(auto_make_map)
    if 'insured_occupation' in df.columns:
        df['insured_occupation'] = df['insured_occupation'].map(insured_occupation_map)

    return df
