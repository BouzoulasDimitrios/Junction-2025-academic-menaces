# Video Files Directory

This directory should contain the drone analysis video files referenced in `features.json`.

## Required Video Files

Based on the current configuration, the following video files are expected:

1. `propeller-speed.mp4` - Propeller Speed Analysis video
2. `movement-prediction.mp4` - Movement Prediction video
3. `speed-prediction.mp4` - Speed Prediction video
4. `boundary-detection.mp4` - Boundary Detection video
5. `noise-removal.mp4` - Noise Removal video

## Video Requirements

- **Format**: MP4 (H.264 codec recommended for broad browser support)
- **Duration**: All videos should have identical duration for synchronized playback
- **Resolution**: 1920x1080 (1080p) recommended, or 1280x720 (720p) minimum
- **Aspect Ratio**: 16:9
- **File Size**: Optimize for web delivery (compress if necessary)

## Adding Video Files

1. Place your MP4 video files in this directory
2. Ensure the filenames match those specified in `features.json`
3. Test playback in the application to verify compatibility

## Creating Test Videos

If you need placeholder videos for testing, you can create them using FFmpeg:

```bash
# Create a 10-second test video with a colored background and text
ffmpeg -f lavfi -i color=c=blue:s=1280x720:d=10 -vf "drawtext=text='Propeller Speed':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" -c:v libx264 -pix_fmt yuv420p propeller-speed.mp4

# Repeat for other features with different colors
ffmpeg -f lavfi -i color=c=green:s=1280x720:d=10 -vf "drawtext=text='Movement Prediction':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" -c:v libx264 -pix_fmt yuv420p movement-prediction.mp4

ffmpeg -f lavfi -i color=c=red:s=1280x720:d=10 -vf "drawtext=text='Speed Prediction':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" -c:v libx264 -pix_fmt yuv420p speed-prediction.mp4

ffmpeg -f lavfi -i color=c=purple:s=1280x720:d=10 -vf "drawtext=text='Boundary Detection':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" -c:v libx264 -pix_fmt yuv420p boundary-detection.mp4

ffmpeg -f lavfi -i color=c=orange:s=1280x720:d=10 -vf "drawtext=text='Noise Removal':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" -c:v libx264 -pix_fmt yuv420p noise-removal.mp4
```

## Troubleshooting

- If videos don't play, check browser console for errors
- Ensure video codec is H.264 (most compatible)
- Verify file permissions allow web server to read files
- Check that video paths in `features.json` match actual filenames
