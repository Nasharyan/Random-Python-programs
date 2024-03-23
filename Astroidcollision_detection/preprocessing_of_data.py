import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_data(file_path):

    # Load the dataset
    data = pd.read_csv(file_path, low_memory=False)

    # Assert uniqueness of 'full_name' column
    assert data['full_name'].nunique() == data.shape[0]

    # Mapping column names for better readability
    column_mapping = {
    'full_name': 'object_full_name_designation',
    'a': 'semi_major_axis',
    'e': 'eccentricity',
    'G': 'magnitude_slope_parameter',
    'i': 'inclination_deg',
    'om': 'longitude_of_the_ascending_node',
    'w': 'argument_of_perihelion',
    'q': 'perihelion_distance',
    'ad': 'aphelion_distance',
    'per_y': 'orbital_period',
    'data_arc': 'data_arc_span',
    'condition_code': 'orbit_condition_code',
    'n_obs_used': 'number_of_observations_used',
    'H': 'absolute_magnitude_parameter',
    'albedo': 'geometric_albedo',
    'rot_per': 'rotation_period',
    'GM': 'standard_gravitational_parameter',
    'BV': 'color_index_BV_magnitude_difference',
    'UB': 'color_index_UB_magnitude_difference',
    'IR': 'color_index_IR_magnitude_difference',
    'spec_B': 'spectral_taxonomic_type_SMASSII',
    'spec_T': 'spectral_taxonomic_type_Tholen',
    'neo': 'near_earth_object',
    'pha': 'physically_hazardous_asteroid',
    'moid': 'earth_minimum_orbit_intersection_distance'
}

    # Rename columns
    data.rename(columns=column_mapping, inplace=True)

    # Convert 'diameter' column to numeric, drop rows with NaN values in 'diameter'
    data['diameter'] = pd.to_numeric(data['diameter'], errors='coerce')
    data.dropna(subset=['diameter'], inplace=True)

    # Fill missing values in 'orbit_condition_code' with mode
    mode_condition_code = data['orbit_condition_code'].mode().iloc[0]
    data['orbit_condition_code'].fillna(mode_condition_code, inplace=True)

    # Convert 'orbit_condition_code' to string and filter out specific values
    data['orbit_condition_code'] = data['orbit_condition_code'].astype(int).astype(str)
    data = data[~data['orbit_condition_code'].isin(['E', 'D'])]

    # Drop columns with high NaN values
    nan_columns = ['magnitude_slope_parameter', 'standard_gravitational_parameter', 'extent', 
    'color_index_BV_magnitude_difference', 'color_index_UB_magnitude_difference', 
    'color_index_IR_magnitude_difference', 'spectral_taxonomic_type_SMASSII', 'spectral_taxonomic_type_Tholen'
]
    data.drop(columns=nan_columns, axis=1, inplace=True)

    # Fill NaNs in numeric columns with the median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # Fill NaNs in non-numeric columns with the mode
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        mode_value = data[col].mode().iloc[0]
        data[col].fillna(mode_value, inplace=True)

    # Define function to handle outliers
    def fix_outlier(column, data):
        IQR = data[column].quantile(0.75) - data[column].quantile(0.25)
        Lower_fence = data[column].quantile(0.25) - (IQR * 3)
        Upper_fence = data[column].quantile(0.75) + (IQR * 3)
        return data[data[column] <= Upper_fence]

    # Remove outliers in 'diameter' column
    data = fix_outlier('diameter', data)

    # Prepare data for modeling
    input_df = data.drop(columns=['diameter'], axis=1)
    target = 'diameter'
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns

    # Scale numeric features
    scaler = MinMaxScaler()
    input_df[numeric_cols] = scaler.fit_transform(input_df[numeric_cols])

    # Encode categorical features
    encoder = OneHotEncoder()
    input_df.drop(columns=['object_full_name_designation'], axis=1, inplace=True)
    categorical_cols = input_df.select_dtypes(exclude=[np.number]).columns.to_list()
    encoded_data = encoder.fit_transform(input_df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(input_features=categorical_cols)
    input_df[encoded_cols] = encoded_data.toarray()
    input_cols = numeric_cols.tolist() + encoded_cols.tolist()

    return input_df[input_cols], data[target]

# Usage:
# X, y = preprocess_data('Asteroid.csv')
