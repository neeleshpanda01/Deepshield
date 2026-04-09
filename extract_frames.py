import os
import cv2
from pathlib import Path

# Define paths
DATA_REAL = Path("data/real")
DATA_FAKE = Path("data/fake")
FRAMES_REAL = Path("frames/real")
FRAMES_FAKE = Path("frames/fake")

# Create output directories
FRAMES_REAL.mkdir(parents=True, exist_ok=True)
FRAMES_FAKE.mkdir(parents=True, exist_ok=True)

# Video extensions to process
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

def extract_frames(video_path, output_dir, frame_interval=10):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract one frame every N frames
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    video_name = video_path.stem
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{saved_count:04d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count

def process_folder(source_dir, output_dir, category_name):
    """
    Process all videos in a source directory.
    
    Args:
        source_dir: Directory containing video files
        output_dir: Directory to save extracted frames
        category_name: Name of the category (real/fake) for logging
    """
    if not source_dir.exists():
        print(f"WARNING: {source_dir} does not exist")
        return
    
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(source_dir.glob(f"*{ext}"))
    
    if not video_files:
        print(f"No video files found in {source_dir}")
        return
    
    print(f"\nProcessing {category_name} videos...")
    total_frames = 0
    
    for video_path in sorted(video_files):
        frames_extracted = extract_frames(video_path, output_dir)
        total_frames += frames_extracted
        print(f"  {video_path.name}: {frames_extracted} frames extracted")
    
    print(f"Total frames extracted from {category_name}: {total_frames}")

# Main execution
if __name__ == "__main__":
    print("=== DeepShield Frame Extraction ===")
    
    process_folder(DATA_REAL, FRAMES_REAL, "REAL")
    process_folder(DATA_FAKE, FRAMES_FAKE, "FAKE")
    
    print("\n=== Extraction Complete ===")
    print(f"Real frames saved to: {FRAMES_REAL}")
    print(f"Fake frames saved to: {FRAMES_FAKE}")
