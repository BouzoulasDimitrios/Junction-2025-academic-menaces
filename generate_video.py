import cv2
import numpy as np
import os

def create_simple_mp4(filename='output.mp4', duration_seconds=10, fps=30, width=1280, height=720, color=(255, 0, 0), text=''):
    """
    Generates a simple MP4 file with colored frames and optional text.
    
    Args:
        filename: Output filename
        duration_seconds: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
        color: BGR color tuple
        text: Text to display on video
    """
    # Try H.264 codec first (best browser compatibility), fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"H.264 codec not available, trying mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter. Check codec support on your system.")
        return False

    num_frames = int(duration_seconds * fps)

    for frame_num in range(num_frames):
        # Create frame with solid color
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = color
        
        # Add text if provided
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 3
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Add frame counter in corner
        counter_text = f"Frame: {frame_num + 1}/{num_frames}"
        cv2.putText(frame, counter_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)

    out.release()
    print(f"✓ Created '{filename}' ({duration_seconds}s, {num_frames} frames)")
    return True

def generate_all_videos():
    """Generate all test videos for the drone analysis application."""
    
    # Create videos directory if it doesn't exist
    os.makedirs('videos', exist_ok=True)
    
    videos = [
        {
            'filename': 'videos/propeller-speed.mp4',
            'color': (0, 0, 255),  # Red (BGR)
            'text': 'Propeller Speed'
        },
        {
            'filename': 'videos/movement-prediction.mp4',
            'color': (0, 255, 0),  # Green (BGR)
            'text': 'Movement Prediction'
        },
        {
            'filename': 'videos/speed-prediction.mp4',
            'color': (255, 0, 0),  # Blue (BGR)
            'text': 'Speed Prediction'
        },
        {
            'filename': 'videos/boundary-detection.mp4',
            'color': (255, 0, 255),  # Magenta (BGR)
            'text': 'Boundary Detection'
        },
        {
            'filename': 'videos/noise-removal.mp4',
            'color': (0, 165, 255),  # Orange (BGR)
            'text': 'Noise Removal'
        }
    ]
    
    print("Generating test videos for drone analysis application...")
    print("=" * 60)
    
    success_count = 0
    for video in videos:
        if create_simple_mp4(
            filename=video['filename'],
            duration_seconds=10,
            fps=30,
            width=1280,
            height=720,
            color=video['color'],
            text=video['text']
        ):
            success_count += 1
    
    print("=" * 60)
    print(f"Generated {success_count}/{len(videos)} videos successfully!")
    
    if success_count == len(videos):
        print("\n✓ All videos created! You can now run the application.")
        print("  Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("\n⚠ Some videos failed to generate. Check codec support.")

if __name__ == "__main__":
    generate_all_videos()