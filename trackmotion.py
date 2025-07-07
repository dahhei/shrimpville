#!/usr/bin/env python3


# show motion plot on top of frame 0
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load the CSV
df = pd.read_csv("mask_centers.csv")

# Select object_id to visualize
object_id = 1
obj_df = df[df["object_id"] == object_id].sort_values("frame")

# Load frame 0 as background
import sys
if len(sys.argv) == 1:
    print("Need arguments passed to get video dir")
    quit(1)

frame_0_path = sys.argv[1] + "/00000.jpg"  # adjust path as needed
frame_img = Image.open(frame_0_path)

# Set up the plot
plt.figure(figsize=(10, 7))
plt.imshow(frame_img)
plt.plot(obj_df["center_x"], obj_df["center_y"], marker='o', linestyle='-', color='cyan', linewidth=2)

# Label each point with the frame number
for _, row in obj_df.iterrows():
    plt.text(row["center_x"], row["center_y"], str(row["frame"]), fontsize=8, color='black')

plt.title(f"Motion Path of Object {object_id} Over Frame 0")
plt.axis("off")  # optional: hides axis ticks
plt.tight_layout()
plt.show()


# show motion plot by itself
"""
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("mask_centers.csv")

# Filter for a specific object (e.g., object_id = 1)
object_id = 1
obj_df = df[df["object_id"] == object_id].sort_values("frame")

# Create the plot
plt.figure(figsize=(9, 6))
plt.plot(obj_df["center_x"], obj_df["center_y"], marker='o', linestyle='-', color='blue')

# Optionally label points by frame number
for i, row in obj_df.iterrows():
    plt.text(row["center_x"], row["center_y"], str(row["frame"]), fontsize=8, color='black')

# Invert y-axis if working in image coordinates
plt.gca().invert_yaxis()

# Plot formatting
plt.title(f"Object {object_id} Motion Path")
plt.xlabel("Center X")
plt.ylabel("Center Y")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
"""
