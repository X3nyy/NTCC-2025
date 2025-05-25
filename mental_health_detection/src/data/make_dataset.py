import os
import pandas as pd
import requests

def download_dataset(url, save_path):
    """Download a dataset from a given URL and save it to the specified path."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Dataset downloaded and saved to {save_path}")
    else:
        print(f"Failed to download dataset from {url}. Status code: {response.status_code}")

def create_raw_data_directory(raw_data_path):
    """Create the raw data directory if it doesn't exist."""
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
        print(f"Created directory: {raw_data_path}")

def main():
    # Define the URL for the dataset
    dataset_url = "https://example.com/path/to/dataset.csv"  # Replace with actual dataset URL
    raw_data_path = os.path.join("data", "raw", "dataset.csv")  # Adjust the filename as needed

    # Create the raw data directory
    create_raw_data_directory(os.path.dirname(raw_data_path))

    # Download the dataset
    download_dataset(dataset_url, raw_data_path)

if __name__ == "__main__":
    main()