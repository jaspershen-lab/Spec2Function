import os
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pickle

def parse_ms_xml_folder(folder_path):
    """
    Parse a folder of XML files containing MS-MS data.

    Args:
        folder_path (str): path to the folder containing XML files

    Returns:
        tuple: (ms_data, meta_data)
            - ms_data: dict keyed by filename (without extension); value is a dict with mz, intensity and molecule_id
            - meta_data: DataFrame with metadata for each file
    """
    # Initialize data structures
    ms_data = {}
    meta_data_list = []

    # Get all XML files
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]

    for file_name in xml_files:
        file_path = os.path.join(folder_path, file_name)

        try:
            # Parse XML file
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Strip file extension
            file_name_without_ext = os.path.splitext(file_name)[0]

            # Extract MS-MS peak data
            mz_list = []
            intensity_list = []
            molecule_id_list = []

            for peak in root.findall('.//ms-ms-peak'):
                mz = peak.find('mass-charge')
                intensity = peak.find('intensity')
                molecule_id = peak.find('molecule-id')

                if mz is not None and intensity is not None:
                    mz_list.append(float(mz.text))
                    intensity_list.append(float(intensity.text))

                    # Extract molecule_id; None if marked nil
                    if molecule_id is not None and 'nil' not in molecule_id.attrib:
                        molecule_id_list.append(molecule_id.text)
                    else:
                        molecule_id_list.append(None)

            # Get database-id
            database_id_elem = root.find('database-id')
            database_id = database_id_elem.text if database_id_elem is not None and database_id_elem.text else np.nan

            # Get ionization-mode (Polarity)
            polarity_elem = root.find('ionization-mode')
            polarity = polarity_elem.text if polarity_elem is not None and polarity_elem.text else np.nan

            # Get precursor_mass (adduct-mass)
            adduct_mass_elem = root.find('adduct-mass')
            precursor_mass = adduct_mass_elem.text if adduct_mass_elem is not None and adduct_mass_elem.text else np.nan

            # Get splash-key
            splash_id_elem = root.find('splash-key')
            splash_id = splash_id_elem.text if splash_id_elem is not None and splash_id_elem.text else np.nan

            # Store MS data - keyed by filename without extension
            ms_data[file_name_without_ext] = {
                'mz': mz_list,
                'intensity': intensity_list,
                'molecule_id': database_id  # use database-id as molecule_id
            }

            # Store metadata - keyed by filename without extension
            meta_data_list.append({
                'file_name': file_name_without_ext,
                'HMDB.ID': database_id,
                'Polarity': polarity,
                'precursor_mass': precursor_mass,
                'splash_id': splash_id
            })

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # Build metadata DataFrame
    meta_data = pd.DataFrame(meta_data_list)

    return ms_data, meta_data

def save_ms_data(ms_data, output_file):
    """
    Save MS data to a pickle file.

    Args:
        ms_data (dict): MS data dictionary
        output_file (str): output file path
    """
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(ms_data, f)
    print(f"MS data saved to {output_file}")

def save_meta_data(meta_data, output_file):
    """
    Save metadata to a CSV file.

    Args:
        meta_data (DataFrame): metadata DataFrame
        output_file (str): output file path
    """
    meta_data.to_csv(output_file, index=False)
    print(f"Metadata saved to {output_file}")

def main():
    # Example usage
    folder_path = "/Users/cgxjdzz/Desktop/NTU phd/ms2_database_feifan/HMDB raw/hmdb_experimental_msms_spectra"  # replace with actual XML folder path
    output_dir = "/Users/cgxjdzz/Desktop/NTU phd/ms2_database_feifan/MS2BioText"  # replace with actual output directory path

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Parse XML files
    ms_data, meta_data = parse_ms_xml_folder(folder_path)

    # Print a sample of the results
    print("MS data sample:")
    for file_name, data in list(ms_data.items())[:1]:  # only print the first file's data
        print(f"File: {file_name}")
        print(f"Number of m/z values: {len(data['mz'])}")
        print(f"First 5 m/z values: {data['mz'][:5]}")
        print(f"First 5 intensities: {data['intensity'][:5]}")
        print(f"molecule_id: {data['molecule_id']}")
        print()

    print("Metadata:")
    print(meta_data.head())

    # Save data
    ms_data_file = os.path.join(output_dir, "new_ms_data.pkl")
    meta_data_file = os.path.join(output_dir, "new_meta_data.csv")

    save_ms_data(ms_data, ms_data_file)
    save_meta_data(meta_data, meta_data_file)

if __name__ == "__main__":
    main()
