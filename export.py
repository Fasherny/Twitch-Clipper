#!/usr/bin/env python3
"""
Export module for BeastClipper
Handles multi-platform exports and Discord integration
"""

import os
import json
import time
import logging
import subprocess
from datetime import datetime
import tempfile

import requests
from requests_toolbelt import MultipartEncoder

from PyQt5.QtCore import QThread, pyqtSignal

# Configure logger
logger = logging.getLogger("BeastClipper")

# Find FFmpeg path (reusing from stream.py)
def find_ffmpeg():
    """Find FFmpeg executable."""
    potential_paths = [
        # Windows common locations
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Users\Ahsan Ali\Downloads\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe",
        os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), "FFmpeg", "bin", "ffmpeg.exe"),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), "FFmpeg", "bin", "ffmpeg.exe"),
    ]
    
    # Check if in PATH
    try:
        result = subprocess.run(["where", "ffmpeg"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except:
        pass
    
    # Check potential paths
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    # If we've reached here, return default and log warning
    logger.warning("FFmpeg not found in common locations. Using 'ffmpeg' command directly.")
    return "ffmpeg"

# Set FFmpeg path
FFMPEG_PATH = find_ffmpeg()


# ======================
# Export Presets
# ======================

class ExportPresets:
    """Manages export presets for different platforms."""
    
    PRESETS = {
        "youtube": {
            "format": "mp4",
            "resolution": "1080p",
            "fps": 30,
            "bitrate": "8M",
            "audio_bitrate": "192k",
            "description": "Optimized for YouTube uploads"
        },
        "twitch": {
            "format": "mp4",
            "resolution": "1080p",
            "fps": 60,
            "bitrate": "6M",
            "audio_bitrate": "160k",
            "description": "Optimized for Twitch clips"
        },
        "tiktok": {
            "format": "mp4",
            "resolution": "1080x1920",
            "fps": 30,
            "bitrate": "5M",
            "audio_bitrate": "128k",
            "description": "Optimized for TikTok (vertical video)"
        }
    }
    
    @staticmethod
    def get_preset(platform):
        """Get export settings for a specific platform."""
        return ExportPresets.PRESETS.get(platform, ExportPresets.PRESETS["youtube"])
    
    @staticmethod
    def get_platforms():
        """Get list of available platform presets."""
        return list(ExportPresets.PRESETS.keys())


# ======================
# Platform Exporter
# ======================

class ClipExporter(QThread):
    """Exports clips with platform-specific optimizations."""
    
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    export_complete = pyqtSignal(bool, str)  # Success status, output path
    error_occurred = pyqtSignal(str)
    
    def __init__(self, input_path, output_dir, platform="youtube", custom_settings=None):
        """
        Initialize clip exporter.
        
        Args:
            input_path: Path to input video file
            output_dir: Directory to save exported video
            platform: Target platform (youtube, twitter, etc.)
            custom_settings: Dict of custom settings to override presets
        """
        super().__init__()
        
        self.input_path = input_path
        self.output_dir = output_dir
        self.platform = platform
        
        # Get platform preset settings
        self.settings = ExportPresets.get_preset(platform).copy()
        
        # Override with any custom settings
        if custom_settings:
            self.settings.update(custom_settings)
        
        self.process = None
    
    def run(self):
        try:
            self.status_update.emit(f"Exporting clip for {self.platform}...")
            self.progress_update.emit(5)
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.platform}_{timestamp}.{self.settings['format']}"
            output_path = os.path.join(self.output_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get video duration
            duration_seconds = self._get_video_duration(self.input_path)
            
            # Check if video needs to be trimmed for platform limits
            if "max_duration" in self.settings and duration_seconds > self.settings["max_duration"]:
                self.status_update.emit(f"Video exceeds {self.platform} duration limit. Trimming...")
                trimmed_path = self._trim_video(self.input_path, self.settings["max_duration"])
                if trimmed_path:
                    self.input_path = trimmed_path
            
            # Build FFmpeg command for the platform
            cmd = [
                FFMPEG_PATH,
                "-y",  # Overwrite output
                "-i", self.input_path
            ]
            
            # Add video settings
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "medium",
                "-profile:v", "high",
                "-crf", "18",
                "-b:v", self.settings["bitrate"],
                "-maxrate", self.settings["bitrate"],
                "-bufsize", str(int(self.settings["bitrate"].replace("M", "")) * 2) + "M"
            ])
            
            # Add resolution setting if needed
            vf_filters = []
            if "resolution" in self.settings:
                if self.settings["resolution"] == "1080p":
                    vf_filters.append("scale=-1:1080")
                elif self.settings["resolution"] == "720p":
                    vf_filters.append("scale=-1:720")
                elif self.settings["resolution"] == "480p":
                    vf_filters.append("scale=-1:480")
                elif self.settings["resolution"] == "360p":
                    vf_filters.append("scale=-1:360")
            
            # Apply video filters if any
            if vf_filters:
                cmd.extend(["-vf", ",".join(vf_filters)])
            
            # Add fps setting
            if "fps" in self.settings:
                cmd.extend(["-r", str(self.settings["fps"])])
            
            # Add audio settings
            cmd.extend([
                "-c:a", "aac",
                "-b:a", self.settings["audio_bitrate"]
            ])
            
            # Add output file
            cmd.append(output_path)
            
            # Start the export process
            self.progress_update.emit(10)
            self.status_update.emit(f"Running FFmpeg export for {self.platform}...")
            
            logger.info(f"Export command: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor progress
            progress = 10
            for line in self.process.stderr:
                # Parse FFmpeg progress
                if "time=" in line:
                    try:
                        time_str = line.split("time=")[1].split()[0]
                        # Parse time format HH:MM:SS.ss
                        parts = time_str.split(':')
                        if len(parts) == 3:
                            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                            current_seconds = h * 3600 + m * 60 + s
                            if duration_seconds > 0:
                                progress = min(10 + int((current_seconds / duration_seconds) * 85), 95)
                                self.progress_update.emit(progress)
                    except Exception as e:
                        logger.debug(f"Error parsing FFmpeg progress: {e}")
                        # If we can't parse progress, just increment slowly
                        if progress < 90:
                            progress += 5
                            self.progress_update.emit(progress)
            
            # Wait for completion
            return_code = self.process.wait()
            
            # Check result
            if return_code == 0 and os.path.exists(output_path):
                self.progress_update.emit(100)
                self.status_update.emit(f"Export for {self.platform} complete!")
                self.export_complete.emit(True, output_path)
                
                # Clean up temp files
                if self.input_path != self.input_path and os.path.exists(self.input_path):
                    try:
                        os.remove(self.input_path)
                    except:
                        pass
            else:
                error_output = self.process.stderr.read() if self.process.stderr else ""
                self.error_occurred.emit(f"Export failed with code {return_code}: {error_output[:200]}")
                self.export_complete.emit(False, "")
                
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            self.error_occurred.emit(f"Export error: {str(e)}")
            self.export_complete.emit(False, "")
        
        finally:
            if self.process:
                try:
                    self.process.terminate()
                except:
                    pass
    
    def _get_video_duration(self, video_path):
        """Get video duration in seconds using FFmpeg."""
        try:
            cmd = [
                FFMPEG_PATH,
                "-i", video_path,
                "-f", "null",
                "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse duration from FFmpeg output
            for line in result.stderr.split('\n'):
                if "Duration" in line:
                    time_str = line.split("Duration: ")[1].split(",")[0].strip()
                    h, m, s = map(float, time_str.split(':'))
                    return h * 3600 + m * 60 + s
            
            return 0
        except Exception as e:
            logger.error(f"Error getting video duration: {e}")
            return 0
    
    def _trim_video(self, video_path, max_duration):
        """Trim video to specified duration."""
        try:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(
                temp_dir,
                f"trimmed_{os.path.basename(video_path)}"
            )
            
            # FFmpeg trim command
            cmd = [
                FFMPEG_PATH,
                "-y",
                "-i", video_path,
                "-t", str(max_duration),
                "-c:v", "copy",  # Fast copy without re-encoding
                "-c:a", "copy",
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if os.path.exists(output_path):
                return output_path
            
            return None
        except Exception as e:
            logger.error(f"Error trimming video: {e}")
            return None
            
    def stop(self):
        """Stop the export process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                if self.process:
                    self.process.kill()
