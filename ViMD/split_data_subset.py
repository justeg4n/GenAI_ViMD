import os
import pandas as pd
import glob
from tqdm import tqdm

def main():
    source_dir = r"D:\FPTU\Semester 7\DAT301m\Dataset\ViMD\data"
    dest_dir = r"D:\FPTU\Semester 7\DAT301m\Dataset\ViMD\data_subset_40"

    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Find all parquet files in the source directory
    parquet_files = glob.glob(os.path.join(source_dir, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files to process.")

    # Process each file
    for file_path in tqdm(parquet_files, desc="Processing files"):
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, file_name)
        
        # Skip if already exists
        if os.path.exists(dest_path):
            print(f"Skipping {file_name}, already exists.")
            continue

        try:
            df = pd.read_parquet(file_path)
            if 'province_name' in df.columns:
                stratify_col = 'province_name'
            elif 'province_code' in df.columns:
                stratify_col = 'province_code'
            else:
                print(f"Warning: No province column found in {file_name}. Performing random split.")
                df_sampled = df.sample(frac=0.4, random_state=42)
                df_sampled.to_parquet(dest_path, index=False)
                continue

            df_sampled = df.groupby(stratify_col, group_keys=False).apply(
                lambda x: x.sample(frac=0.4, random_state=42)
            )
            
            df_sampled.to_parquet(dest_path, index=False)
            
        except Exception as e:
            print(f"\nError processing {file_name}: {e}")

    print("\nData splitting complete!")
    print(f"Subset saved to: {dest_dir}")

if __name__ == "__main__":
    main()
