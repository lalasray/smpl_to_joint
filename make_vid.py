import os
import cv2

# Specify the path to the folder containing the images
folder_path = r"frames"

# Get a list of image filenames
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Sort the images based on the numerical part of the filename
image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by the number after 'frame_'

# Read the first image to get the width and height
first_image_path = os.path.join(folder_path, image_files[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape

# Set up the video writer to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for video
output_video_path = os.path.join(folder_path, 'output_video.mp4')
fps = 15  # Frames per second

# Create the VideoWriter object
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Loop through each image and write it to the video
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    out.write(image)

# Release the VideoWriter and close any OpenCV windows
out.release()
#cv2.destroyAllWindows()

print(f"Video saved at {output_video_path}")
