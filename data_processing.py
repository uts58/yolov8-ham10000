import pandas as pd
import os
import shutil


parent_path = "/mmfs1/scratch/utsha.saha/skncncr/data/test"
df = pd.read_csv(f"{parent_path}/ISIC2018_Task3_Test_GroundTruth.csv")


for items in df.columns:
	os.makedirs(f"{parent_path}/{items.strip()}", exist_ok=True)



for i, row in df.iterrows():
    filename = row["image"] + '.jpg'
    for col in df.columns:
        if row[col] == 1:
            source_path = f"{parent_path}/ISIC2018_Task3_Test_Input/{filename}"
            destination_dir = f"{parent_path}/{col}"
            shutil.move(source_path, destination_dir)
            print(f"Moved {filename} to {destination_dir}")

