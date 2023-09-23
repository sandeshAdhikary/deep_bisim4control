import cv2
import os
import argparse
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder')
    args = parser.parse_args()

    image_folder = args.image_folder

    # Output video file name
    output_video = os.path.join(image_folder, 'video.mp4')

    # Set the frame rate (frames per second) and codec
    fps = 10  # You can adjust this value as needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format

    # Get the list of image files in the directory
    # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]  # Change the file extension as needed
    images = glob.glob(os.path.join(image_folder, '*.jpg'))

    # Sort the image files by their numerical names
    images.sort(key=lambda x: x.split('.')[0])

    # Get the first image to determine the size of the video frames
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Create a VideoWriter object
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Loop through the images and write each frame to the video
    for image in images:
        img = cv2.imread(image)
        video.write(img)

    # Release the video writer and close the video file
    video.release()

    print(f"Video '{output_video}' has been created.")