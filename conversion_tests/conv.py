#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Common video extensions to process in directory mode
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mpg", ".mpeg", ".webm"}


def convert_to_high_profile(
    input_path: Path,
    output_path: Optional[Path] = None,
    crf: int = 18,
    preset: str = "slow",
) -> Path:
    """
    Convert a video to H.264 High Profile using ffmpeg.

    Parameters
    ----------
    input_path : Path
        Path to input video.
    output_path : Path | None
        Path to output video. If None, appends '.high.mp4' next to the input.
    crf : int
        Constant Rate Factor (18–23 is a good range; lower = better quality, larger file).
    preset : str
        x264 preset: ultrafast, superfast, veryfast, faster, fast, medium,
        slow, slower, veryslow. Slower = better compression.

    Returns
    -------
    Path
        Path to the output video.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + ".high.mp4")

    cmd = [
        "ffmpeg",
        "-y",                     # overwrite output
        "-i", str(input_path),    # input
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level:v", "4.1",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(crf),
        "-movflags", "+faststart",  # better streaming compatibility
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    return output_path


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    crf: int = 18,
    preset: str = "slow",
) -> None:
    """
    Convert all video files from input_dir to High Profile and save them
    in output_dir with the SAME filenames.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]

    if not files:
        print(f"No video files with extensions {sorted(VIDEO_EXTS)} found in {input_dir}")
        return

    print(f"Found {len(files)} video(s) in {input_dir}")
    for idx, src in enumerate(files, start=1):
        dst = output_dir / src.name
        print(f"[{idx}/{len(files)}] Converting {src.name} -> {dst}")
        convert_to_high_profile(src, dst, crf=crf, preset=preset)

    print("Batch conversion done.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert videos to H.264 High Profile.\n"
            "Modes:\n"
            "  1) Single file: convert one input file.\n"
            "  2) Batch: convert all videos from --input-dir to --output-dir."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Single-file mode
    parser.add_argument(
        "input",
        nargs="?",
        help="Input video file (single-file mode). Omit if using --input-dir.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output video file (single-file mode). Default: <input_stem>.high.mp4",
        default=None,
    )

    # Batch mode
    parser.add_argument(
        "--input-dir",
        help="Directory with input videos (batch mode).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where converted videos are stored (batch mode). "
             "Must be different from --input-dir.",
    )

    # Common options
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Constant Rate Factor for quality (lower = better, default: 18)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="slow",
        help=(
            "x264 preset (ultrafast, superfast, veryfast, faster, fast, "
            "medium, slow, slower, veryslow). Default: slow"
        ),
    )

    args = parser.parse_args()

    # Batch mode
    if args.input_dir is not None:
        if args.output_dir is None:
            parser.error("--output-dir is required when using --input-dir")

        in_dir = Path(args.input_dir).expanduser().resolve()
        out_dir = Path(args.output_dir).expanduser().resolve()

        if in_dir == out_dir:
            parser.error("input and output directories must be different")

        batch_convert(in_dir, out_dir, crf=args.crf, preset=args.preset)
        return

    # Single-file mode
    if args.input is None:
        parser.error(
            "You must either:\n"
            "  * Provide an input file (single-file mode), OR\n"
            "  * Use --input-dir and --output-dir (batch mode)."
        )

    in_file = Path(args.input).expanduser()
    out_file = Path(args.output).expanduser() if args.output else None

    out = convert_to_high_profile(
        input_path=in_file,
        output_path=out_file,
        crf=args.crf,
        preset=args.preset,
    )
    print(f"Converted file saved to: {out}")


if __name__ == "__main__":
    main()
