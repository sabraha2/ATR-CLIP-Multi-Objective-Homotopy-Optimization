import os
import pandas as pd
from pathlib import Path
import argparse
import shutil

def prepare_data(csv_file, output_folder):
    """
    Prepares data by reading a CSV file and writing descriptions to text files,
    optionally copying images to the output folder.

    Parameters:
    csv_file (str): The path to the CSV file containing the dataset.
    output_folder (str): The directory where the new data will be stored.
    """
    try:
        # Load the dataset
        df = pd.read_csv(csv_file)

        # Create a new data folder
        data_folder = Path(output_folder)
        data_folder.mkdir(exist_ok=True, parents=True)

        # Process each record in the dataframe
        for index, row in df.iterrows():
            image_path = Path(row['Image Path'])
            image_name = image_path.stem
            description = row['Description']

            # Define the path for the new text file
            text_file_path = data_folder / f"{image_name}.txt"

            # Write the description to the text file
            with open(text_file_path, 'w') as file:
                file.write(description)

            # Optionally copy the image to the output folder
            # If image file doesn't exist, this will raise an exception
            # Uncomment the next line if image copying is desired
            # shutil.copy(image_path, data_folder / image_path.name)

        print(f"Data prepared successfully in the '{data_folder}' directory.")
    
    except FileNotFoundError:
        print(f"Error: The file {csv_file} does not exist.")
    except pd.errors.EmptyDataError:
        print("Error: The given file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for CLIP training")
    parser.add_argument('csv_file', type=str, help="Path to the CSV file containing the dataset")
    parser.add_argument('--output_folder', type=str, default='./DATA', 
                        help="Folder where the new data will be stored (default is './DATA')")

    args = parser.parse_args()

    prepare_data(args.csv_file, args.output_folder)
