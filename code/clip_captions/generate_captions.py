import os
import pandas as pd
import re

# Define constants for sensor types and their descriptions
SENSOR_MAP = {
    'cegr': 'Mid-Wave Infrared (MWIR)',
    'i1co': 'Visible'
}

# Define mappings for scenario numbers to descriptions
SCENARIO_MAP = {
    '02003': 'during the day at 1000 meters', '02005': 'during the day at 1500 meters',
    '02007': 'during the day at 2000 meters', '02009': 'during the day at 2500 meters',
    '02011': 'during the day at 3000 meters', '02013': 'during the day at 3500 meters',
    '02015': 'during the day at 4000 meters', '02017': 'during the day at 4500 meters',
    '02019': 'during the day at 5000 meters', '01923': 'at night at 1000 meters',
    '01925': 'at night at 1500 meters', '01927': 'at night at 2000 meters',
    '01929': 'at night at 2500 meters', '01931': 'at night at 3000 meters',
    '01933': 'at night at 3500 meters', '01935': 'at night at 4000 meters',
    '01937': 'at night at 4500 meters', '01939': 'at night at 5000 meters',
    '02002': 'during the day at 500 meters', '02004': 'during the day at 1000 meters',
    '02006': 'during the day at 1500 meters', '02008': 'during the day at 2000 meters',
    '02010': 'during the day at 2500 meters', '02012': 'during the day at 3000 meters',
    '01926': 'at night at 500 meters', '01928': 'at night at 1000 meters',
    '01932': 'at night at 1500 meters', '01934': 'at night at 2000 meters',
    '01936': 'at night at 2500 meters', '01938': 'at night at 3000 meters'
}

# Define mappings for look numbers to vehicle or human descriptions
LOOK_NUMBER_MAP = {
    '0001': 'Pickup', '0002': 'Sport Utility Vehicle', '0005': 'BTR70 – Armored Personnel Carrier',
    '0006': 'BRDM2 – Infantry Scout Vehicle', '0009': 'BMP2 – Armored Personnel Carrier',
    '0010': 'T62 – Main Battle Tank', '0011': 'T72 – Main Battle Tank',
    '0012': 'ZSU23-4 - Anti-Aircraft Weapon', '0013': '2S3 – Self-Propelled Howitzer',
    '0014': 'MTLB – Armored Reconnaissance Vehicle Towing a D20 Artillery Piece'
}

# Set of human scenario codes
HUMAN_SCENARIO_CODES = {
    '02002', '02004', '02006', '02008', '02010', '02012',
    '01926', '01928', '01932', '01934', '01936', '01938'
}

def generate_natural_language_caption(sensor_code, scenario_code, look_number_code):
    """
    Generates a natural language caption for an image based on sensor code, scenario code,
    and look number code.
    
    Args:
        sensor_code (str): Code of the sensor.
        scenario_code (str): Code of the scenario.
        look_number_code (str): Code representing the specific look number.

    Returns:
        str: A descriptive caption based on the provided codes.
    """
    sensor = SENSOR_MAP[sensor_code]
    scenario = SCENARIO_MAP[scenario_code]

    if scenario_code in HUMAN_SCENARIO_CODES:
        target_type = 'Humans'
        action = 'moving in figure-8 at slow pace' if look_number_code == '0001' else 'moving in figure-8 at faster pace'
    else:
        target_type = LOOK_NUMBER_MAP.get(look_number_code, 'an unidentified target')
        action = 'driving in a circle'

    return f"Captured with a {sensor} sensor, the imagery depicts {target_type} {action} {scenario}."

def extract_info_from_filename(filename):
    """
    Extracts sensor code, scenario code, and look number code from the filename.
    
    Args:
        filename (str): The filename to extract information from.
        
    Returns:
        tuple: Sensor code, scenario code, and look number code if extracted, else 'Unknown' for each.
    """
    match = re.match(r'(cegr|i1co)(\d{5})_(\d{4})_(\d+).png', filename)
    return match.groups() if match else ('Unknown', 'Unknown', 'Unknown')

def create_clip_dataset(directory_paths):
    """
    Creates a dataset from images stored in specified directories.
    
    Args:
        directory_paths (list): A list of directory paths to process.
        
    Returns:
        DataFrame: A DataFrame containing paths and descriptions of images.
    """
    data = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif')
    for directory_path in directory_paths:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(valid_extensions):
                    full_path = os.path.join(root, file)
                    sensor_code, scenario_code, look_number_code = extract_info_from_filename(file)
                    if sensor_code != 'Unknown':
                        description = generate_natural_language_caption(sensor_code, scenario_code, look_number_code)
                        data.append([full_path, sensor_code, scenario_code, look_number_code, description])
                    else:
                        print(f"Skipping {file}: Unable to extract necessary codes.")

    return pd.DataFrame(data, columns=['Image Path', 'Sensor Code', 'Scenario Code', 'Look Number Code', 'Description'])

# Usage example
if __name__ == "__main__":
    directory_paths = ['/path/to/directory1', '/path/to/directory2']
    clip_dataset = create_clip_dataset(directory_paths)
    clip_dataset.to_csv('clip_dataset.csv', index=False)
    print("Dataset prepared and saved to 'clip_dataset.csv'.")
    print(clip_dataset.head())
