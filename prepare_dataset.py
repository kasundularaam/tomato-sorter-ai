import os
import cv2
from tqdm import tqdm
import glob

# Constants
INPUT_DIR = 'videos'
OUTPUT_DIR = 'dataset'
FRAME_INTERVAL = 15
RESIZE_WIDTH = None
RESIZE_HEIGHT = None
CLASSES = ['ripe', 'unripe', 'damaged']


def extract_frames(video_path, output_dir, frame_interval=FRAME_INTERVAL, resize=None):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get video filename without extension
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    print(f"Processing {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")

    # Initialize counters
    frame_count = 0
    success = True

    # Create progress bar
    pbar = tqdm(total=total_frames)

    # Extract frames
    while success:
        success, frame = video.read()

        if not success:
            break

        # Extract every nth frame
        if frame_count % frame_interval == 0:
            # Resize frame if specified
            if resize:
                frame = cv2.resize(frame, resize)

            # Save the frame as an image
            frame_filename = f"{video_filename}_frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

        frame_count += 1
        pbar.update(1)

    # Cleanup
    video.release()
    pbar.close()

    print(
        f"Extracted {frame_count // frame_interval} frames from {video_path}")


def process_videos():
    # Prepare resize dimensions if specified
    resize = None
    if RESIZE_WIDTH and RESIZE_HEIGHT:
        resize = (RESIZE_WIDTH, RESIZE_HEIGHT)

    # Create the output base directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each class
    for class_name in CLASSES:
        input_dir = os.path.join(INPUT_DIR, class_name)
        output_dir = os.path.join(OUTPUT_DIR, class_name)

        # Skip if input directory doesn't exist
        if not os.path.exists(input_dir):
            print(
                f"Warning: Input directory {input_dir} does not exist, skipping...")
            continue

        # Create the output directory for this class
        os.makedirs(output_dir, exist_ok=True)

        # Get all video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_dir, ext)))

        print(f"Found {len(video_files)} videos in {input_dir}")

        # Process each video
        for video_path in video_files:
            extract_frames(video_path, output_dir, FRAME_INTERVAL, resize)


def print_dataset_stats():
    print(f"Dataset structure:")

    for class_name in CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        if os.path.exists(class_dir):
            num_images = len([f for f in os.listdir(
                class_dir) if f.endswith('.jpg')])
            print(f"  {class_name}: {num_images} images")


def main():
    # Process all videos
    process_videos()

    print(f"Finished extracting frames to {OUTPUT_DIR}")

    # Print dataset statistics
    print_dataset_stats()


if __name__ == "__main__":
    main()
