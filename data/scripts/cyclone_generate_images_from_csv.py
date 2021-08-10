import argparse
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

## -----------------------------------------------------------------
## Argument parser
## -----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Create Image data from North Indian Ocean Cyclone Dataset")
parser.add_argument(
    '--src', type=str, 
    default='../data/South_Indian_Ocean/southindianocean_jtwc.csv', 
    help="Path to the source csv file"
)
parser.add_argument(
    '--dst', type=str, 
    default='../data/North_Indian_Ocean/images/', 
    help="Path to store the processed data"
)
parser.add_argument(
    '--format', type=str, 
    default='numpy', 
    help="Save as numpy or jpg?"
)

args = parser.parse_args()


## -----------------------------------------------------------------
## 1. READ DATA FILE
## -----------------------------------------------------------------
cyclone_df = pd.read_csv(
    args.src, delimiter = ",", header = None, 
    names =['id', 'timestamp', 'longitude', 'latitude', 'speed']
)

# Format the timestamp column
cyclone_df['timestamp'] = pd.to_datetime(cyclone_df['timestamp'], format='%Y%m%d%H')

# Convert logitude to float type
cyclone_df.loc[cyclone_df.longitude=='#VALUE!', 'longitude'] = np.nan
cyclone_df.dropna(inplace=True)

cyclone_df['longitude'] = cyclone_df['longitude'].astype(np.float16)
cyclone_df.reset_index(drop=True, inplace=True)



## -----------------------------------------------------------------
## 2. ASSIGN UNIQUE ID TO EACH TRACK
## -----------------------------------------------------------------

# Compare current and next id
curr = cyclone_df.loc[1:, 'id'].values
prev = cyclone_df.loc[:len(cyclone_df)-2, 'id'].values

# Indices where the id changes
change = np.concatenate([[True], curr != prev])
cyclone_df.loc[:, 'change'] = change

# Assign a new track_id to the row when change value is true
track_id = 0

def assign_cyclone_id(change):
    global track_id
    if change:
        track_id += 1
    return track_id

cyclone_df.loc[:, 'track_id'] = cyclone_df.change.apply(assign_cyclone_id)

# Min and max speed (for normalization)
min_speed, max_speed = cyclone_df.loc[cyclone_df.speed > 0].speed.min(), cyclone_df.speed.max()



## -----------------------------------------------------------------
## 3. Generate numpy files for each track
## -----------------------------------------------------------------

SCALE = 25
X_PAD = 10
Y_PAD = 10
BORDER = 5
SCALE_RADIUS = 0.05


if os.path.exists(args.dst):
    raise(ValueError(f"Path exists: {args.dst}"))
else:
    os.makedirs(args.dst)

for track_id in tqdm(cyclone_df.track_id.unique()):

    # Filter DF by track_id
    cyclone_df_by_track_id = cyclone_df[cyclone_df['track_id'] == track_id].sort_values(by=['timestamp'])

    # Calculate width and height
    min_lat, max_lat = cyclone_df_by_track_id.latitude.min(), cyclone_df_by_track_id.latitude.max()
    min_long, max_long = cyclone_df_by_track_id.longitude.min(), cyclone_df_by_track_id.longitude.max()

    height = SCALE * int(np.ceil(max_long - min_long)) + 2 * Y_PAD
    width = SCALE * int(np.ceil(max_lat - min_lat)) + 2 * X_PAD

    # Normalize speed
    cyclone_df_by_track_id.speed = (cyclone_df_by_track_id.speed - min_speed) / (max_speed - min_speed) * 255

    images = []

    for latitude, longitude, speed in cyclone_df_by_track_id[['latitude', 'longitude', 'speed']].values:
        
        shape = (
            int( ( 1 + SCALE_RADIUS ) * height ),
            int( ( 1 + SCALE_RADIUS ) * width ),
            3
        )
        blank_img = np.ones(shape, dtype=np.uint8) * 255

        if speed < 0:
            continue

        circle = cv2.circle(
            blank_img, 
            (
                int((latitude - min_lat) * SCALE + X_PAD), 
                int((longitude - min_long) * SCALE + Y_PAD)
            ), 
            int(0.05 * min(height, width)), 
            (255-int(speed), 128, int(speed)), 
            -1
        )

        # ADD BORDER
        border = cv2.copyMakeBorder(
            circle,
            top=BORDER,
            bottom=BORDER,
            left=BORDER,
            right=BORDER,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        images.append(border)

    
    # NUMPY ARRAY
    images = np.array(images)
    np.save(os.path.join(args.dst, f"{track_id}.npy"), images)






