#!/usr/bin/env python3
"""
SIMPLIFIED Twitch Stream Buffer - FFmpeg Primary Method
Records segments directly using FFmpeg without complex dual-process approach
"""

import os
import time
import logging
import subprocess
import threading
import json
import shutil
import traceback
import signal
from collections import deque
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QObject

# Configure logger
logger = logging.getLogger("BeastClipper")

# Find FFmpeg executables - improved detection
def find_ffmpeg():
    """Find FFmpeg and FFprobe executables."""
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
            return result.stdout.strip().split('\n')[0], result.stdout.strip().split('\n')[0].replace('ffmpeg', 'ffprobe')
    except:
        pass
    
    # Check potential paths
    for path in potential_paths:
        if os.path.exists(path):
            probe_path = path.replace('ffmpeg', 'ffprobe')
            return path, probe_path
    
    # If we've reached here, return default and log warning
    logger.warning("FFmpeg not found in common locations. Using 'ffmpeg' command directly.")
    return "ffmpeg", "ffprobe"

# Set FFmpeg paths
FFMPEG_PATH, FFPROBE_PATH = find_ffmpeg()

try:
    from main import BASE_TEMP_DIR
except ImportError:
    BASE_TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(BASE_TEMP_DIR, exist_ok=True)


class StreamBuffer(QThread):
    """Simplified Twitch stream buffer using FFmpeg as primary recording method."""
    
    # Signals
    buffer_progress = pyqtSignal(int, int)  # current, total
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    stream_info_updated = pyqtSignal(dict)
    ad_status_update = pyqtSignal(str)
    
    def __init__(self, stream_url, buffer_duration=300, resolution="best", 
                 segment_length=10, temp_manager=None, buffer_directory=None):
        """Initialize simplified Twitch stream buffer."""
        super().__init__()
        
        self.stream_url = self._format_twitch_url(stream_url)
        self.buffer_duration = buffer_duration
        self.resolution = resolution
        self.segment_length = segment_length
        self.temp_manager = temp_manager
        
        # Buffer management
        self.segments = deque()
        self.segment_lock = threading.RLock()
        self.running = False
        
        # Recording state
        self.current_stream_url = None
        self.url_refresh_time = 0
        self.url_refresh_interval = 1800  # Refresh URL every 30 minutes
        self.segment_index = 0
        self.recording_thread = None
        
        # Use unified temp directory for buffer
        self.temp_dir = os.path.join(
            BASE_TEMP_DIR,
            "buffer",
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Ad detection state
        self.ad_detected = False
        self.last_successful_segment = 0
        self.consecutive_failures = 0
        
        logger.info(f"StreamBuffer initialized for Twitch: {self.stream_url}, buffer dir: {self.temp_dir}")
    
    def _format_twitch_url(self, url):
        """Format Twitch URL to ensure compatibility."""
        try:
            url = url.strip().lower()
            
            # Handle various Twitch URL formats
            if "twitch.tv" not in url:
                # Assume it's just a channel name
                return f"https://twitch.tv/{url}"
            
            # Extract channel name from URL
            if "/videos/" in url:
                raise ValueError("VOD URLs are not supported. Please use a live stream URL.")
            
            # Clean up the URL
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
            
            # Extract just the channel part
            parts = url.split("twitch.tv/")
            if len(parts) > 1:
                channel = parts[1].split('/')[0].split('?')[0]
                return f"https://twitch.tv/{channel}"
            
            return url
        except Exception as e:
            logger.error(f"Error formatting URL: {e}")
            return url
    
    def run(self):
        """Main buffer thread - simplified FFmpeg-based recording."""
        self.running = True
        
        try:
            # Validate stream first
            self.status_update.emit("Checking Twitch stream...")
            
            if not self._validate_stream():
                self.error_occurred.emit("Stream is offline or invalid. Please check the channel name.")
                return
            
            # Get stream information
            self.status_update.emit("Getting stream information...")
            stream_info = self._get_stream_info()
            
            if stream_info:
                self.stream_info_updated.emit(stream_info)
            
            # Get initial direct stream URL
            self.status_update.emit("Getting direct stream URL...")
            if not self._refresh_stream_url():
                self.error_occurred.emit("Could not get direct stream URL")
                return
            
            # Start FFmpeg segment recording
            self.status_update.emit("Starting segment recording...")
            self._start_segment_recording()
            
            # Main monitoring loop
            self.status_update.emit("Recording active!")
            self.ad_status_update.emit("✅ Recording started")
            
            while self.running:
                # Check if URL needs refreshing
                current_time = time.time()
                if current_time - self.url_refresh_time > self.url_refresh_interval:
                    self._refresh_stream_url()
                
                # Update progress and check health
                self._update_progress()
                self._check_recording_health()
                
                time.sleep(5)  # Check every 5 seconds
                
        except Exception as e:
            logger.error(f"Buffer thread error: {traceback.format_exc()}")
            self.error_occurred.emit(f"Buffer error: {str(e)}")
        
        finally:
            self.running = False
            self._cleanup()
            self.status_update.emit("Buffer stopped")
    
    def _validate_stream(self):
        """Validate that the stream exists and is live."""
        try:
            logger.info(f"Validating stream: {self.stream_url}")
            
            # Use streamlink to check if stream is live
            cmd = ["streamlink", "--stream-url", self.stream_url, "worst"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and "http" in result.stdout:
                logger.info("Stream validation successful")
                return True
            
            logger.warning(f"Stream validation failed: {result.stderr[:200] if result.stderr else 'No error output'}")
            return False
            
        except Exception as e:
            logger.error(f"Stream validation error: {str(e)}")
            return False
    
    def _get_stream_info(self):
        """Get basic stream information."""
        try:
            # Get JSON info from streamlink
            cmd = ["streamlink", "--json", self.stream_url]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    stream_info = {
                        'url': self.stream_url,
                        'channel': self.stream_url.split('/')[-1],
                        'qualities': list(data.get('streams', {}).keys()),
                        'title': data.get('title', 'Unknown')
                    }
                    logger.info(f"Got stream info: {stream_info['channel']}")
                    return stream_info
                except json.JSONDecodeError:
                    pass
            
            # Fallback - return basic info
            return {
                'url': self.stream_url,
                'channel': self.stream_url.split('/')[-1],
                'qualities': ['best'],
                'title': 'Unknown'
            }
            
        except Exception as e:
            logger.error(f"Error getting stream info: {str(e)}")
            return None
    
    def _refresh_stream_url(self):
        """Get/refresh the direct stream URL."""
        try:
            # Map resolution to streamlink quality
            quality_map = {
                "1080p": "1080p60,1080p,best",
                "720p": "720p60,720p,best", 
                "480p": "480p,best",
                "360p": "360p,worst",
                "best": "best"
            }
            quality = quality_map.get(self.resolution, "best")
            
            logger.info(f"Getting direct stream URL for quality: {quality}")
            
            # Get direct stream URL
            cmd = ["streamlink", "--stream-url", self.stream_url, quality]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and result.stdout.strip():
                self.current_stream_url = result.stdout.strip()
                self.url_refresh_time = time.time()
                logger.info(f"Got fresh stream URL: {self.current_stream_url[:50]}...")
                return True
            else:
                logger.error(f"Failed to get stream URL: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error refreshing stream URL: {e}")
            return False
    
    def _start_segment_recording(self):
        """Start the main segment recording thread."""
        if self.recording_thread and self.recording_thread.is_alive():
            return
        
        self.recording_thread = threading.Thread(target=self._segment_recording_loop, daemon=True)
        self.recording_thread.start()
    
    def _segment_recording_loop(self):
        """Main segment recording loop using FFmpeg."""
        while self.running:
            try:
                if not self.current_stream_url:
                    logger.warning("No stream URL available, refreshing...")
                    if not self._refresh_stream_url():
                        time.sleep(10)
                        continue
                
                # Record one segment
                success = self._record_single_segment()
                
                if success:
                    self.consecutive_failures = 0
                    self.last_successful_segment = time.time()
                    
                    # Check if we were in ad state and now have content
                    if self.ad_detected:
                        self.ad_detected = False
                        self.ad_status_update.emit("✅ Content resumed - recording active")
                else:
                    self.consecutive_failures += 1
                    
                    # Handle failures
                    if self.consecutive_failures >= 3:
                        logger.warning("Multiple segment failures, refreshing stream URL...")
                        if not self._refresh_stream_url():
                            time.sleep(10)
                            continue
                        self.consecutive_failures = 0
                    
                    # Might be ads or stream issue
                    if not self.ad_detected:
                        self.ad_detected = True
                        self.ad_status_update.emit("📺 Stream issue detected - attempting recovery...")
                    
                    time.sleep(2)  # Short delay before retry
                
            except Exception as e:
                logger.error(f"Error in segment recording loop: {e}")
                time.sleep(5)
    
    def _record_single_segment(self):
        """Record a single segment using FFmpeg."""
        try:
            segment_file = os.path.join(self.temp_dir, f"segment_{self.segment_index:05d}.ts")
            
            # FFmpeg command to record one segment with proper audio handling
            ffmpeg_cmd = [
                FFMPEG_PATH,
                "-hide_banner",
                "-loglevel", "warning",
                "-i", self.current_stream_url,
                "-t", str(self.segment_length),
                "-c:v", "copy",
                "-c:a", "aac",  # Re-encode audio to ensure compatibility
                "-b:a", "128k",  # Set audio bitrate
                "-ac", "2",      # Ensure stereo audio
                "-ar", "44100",  # Set audio sample rate
                "-f", "mpegts",
                "-y",
                segment_file
            ]
            
            logger.debug(f"Recording segment {self.segment_index}")
            
            # Run FFmpeg with timeout
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for segment with timeout
            try:
                process.wait(timeout=self.segment_length + 10)
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"Segment {self.segment_index} recording timed out")
                return False
            
            # Check if segment was created successfully
            if os.path.exists(segment_file) and os.path.getsize(segment_file) > 10000:  # At least 10KB
                # Add to segments deque
                with self.segment_lock:
                    self.segments.append({
                        'file': segment_file,
                        'index': self.segment_index,
                        'timestamp': time.time(),
                        'duration': self.segment_length,
                        'size': os.path.getsize(segment_file),
                        'method': 'ffmpeg_direct'
                    })
                    
                    self._prune_old_segments()
                
                logger.debug(f"Successfully recorded segment {self.segment_index}")
                self.segment_index += 1
                return True
            else:
                # Clean up failed segment
                try:
                    if os.path.exists(segment_file):
                        os.remove(segment_file)
                except:
                    pass
                
                logger.warning(f"Segment {self.segment_index} failed - file too small or missing")
                return False
                
        except Exception as e:
            logger.error(f"Error recording segment {self.segment_index}: {e}")
            return False
    
    def _prune_old_segments(self):
        """Remove old segments when buffer is full."""
        try:
            current_buffer_duration = len(self.segments) * self.segment_length
            
            while current_buffer_duration > self.buffer_duration and self.segments:
                old_segment = self.segments.popleft()
                try:
                    if os.path.exists(old_segment['file']):
                        os.remove(old_segment['file'])
                        logger.debug(f"Removed old segment: {old_segment['file']}")
                except Exception as e:
                    logger.error(f"Error removing old segment: {e}")
                
                current_buffer_duration = len(self.segments) * self.segment_length
                
        except Exception as e:
            logger.error(f"Error pruning segments: {e}")
    
    def _check_recording_health(self):
        """Check if recording is healthy and working."""
        current_time = time.time()
        
        # Check if we've had a successful segment recently
        if self.last_successful_segment > 0:
            time_since_last = current_time - self.last_successful_segment
            
            if time_since_last > 60:  # No successful segment in 1 minute
                logger.warning("No successful segments in 60 seconds, may need intervention")
                self.ad_status_update.emit("⚠️ Recording issues detected")
            elif time_since_last > 30:  # Warning at 30 seconds
                if not self.ad_detected:
                    self.ad_detected = True
                    self.ad_status_update.emit("📺 Possible ads or stream issues")
    
    def _update_progress(self):
        """Update progress indicators."""
        try:
            with self.segment_lock:
                current_duration = len(self.segments) * self.segment_length
            
            # Emit progress
            self.buffer_progress.emit(min(current_duration, self.buffer_duration), self.buffer_duration)
            
            # Update status based on current state
            if self.ad_detected:
                status_msg = f"Buffering: {current_duration}s / {self.buffer_duration}s (Stream issues)"
            else:
                status_msg = f"Buffering: {current_duration}s / {self.buffer_duration}s"
            
            self.status_update.emit(status_msg)
                
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    def get_buffer_status(self):
        """Get buffer status information."""
        with self.segment_lock:
            current_duration = len(self.segments) * self.segment_length
            return {
                'segments': len(self.segments),
                'duration': current_duration,
                'max_duration': self.buffer_duration,
                'segment_length': self.segment_length,
                'method': 'ffmpeg_primary',
                'ad_detected': self.ad_detected,
                'url_age': time.time() - self.url_refresh_time if self.url_refresh_time > 0 else 0
            }
    
    def get_segments_for_clip(self, start_time_ago, duration):
        """Get segments for creating a clip."""
        with self.segment_lock:
            if not self.segments:
                logger.warning("No segments available for clip creation")
                return []
            
            # Calculate how many segments we need
            segments_needed = max(1, int(duration / self.segment_length)) + 1
            
            # Calculate starting point (from the end, going backwards)
            start_segments_ago = max(1, int(start_time_ago / self.segment_length))
            
            # Get the relevant segments
            total_segments = len(self.segments)
            if total_segments < segments_needed:
                logger.warning(f"Not enough segments. Have {total_segments}, need {segments_needed}")
                return list(self.segments)
            
            # Calculate indices
            end_index = total_segments - start_segments_ago
            start_index = max(0, end_index - segments_needed)
            
            # Extract segments
            clip_segments = []
            for i in range(start_index, min(end_index, total_segments)):
                if 0 <= i < len(self.segments):
                    segment = self.segments[i]
                    # Verify file still exists
                    if os.path.exists(segment['file']):
                        clip_segments.append(segment)
            
            logger.info(f"Found {len(clip_segments)} segments for clip (requested {segments_needed})")
            return clip_segments
    
    def stop(self):
        """Stop the buffer thread."""
        logger.info("Stopping stream buffer...")
        self.running = False
        
        # Recording thread will stop on its own when self.running = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=10)
    
    def _cleanup(self):
        """Clean up resources."""
        try:
            # Register temp directory for cleanup
            if self.temp_manager:
                self.temp_manager.register_temp_file(self.temp_dir, lifetime=3600)
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Rest of the classes remain unchanged (StreamMonitor, ClipCreator, ClipEditor)
class StreamMonitor(QThread):
    """Monitors stream status and updates the UI."""
    
    # Signals
    stream_live = pyqtSignal(str)  # stream URL
    stream_offline = pyqtSignal(str)  # stream URL
    status_update = pyqtSignal(str)  # status message
    
    def __init__(self, config_manager):
        """Initialize stream monitor."""
        super().__init__()
        self.config_manager = config_manager
        self.running = False
        self.check_interval = 60  # Check every 60 seconds
        self.monitored_streams = []
        self.load_monitored_streams()
    
    def load_monitored_streams(self):
        """Load monitored streams from config."""
        self.monitored_streams = self.config_manager.get("monitored_streams", [])
    
    def save_monitored_streams(self):
        """Save monitored streams to config."""
        self.config_manager.set("monitored_streams", self.monitored_streams)
    
    def add_stream(self, stream_url):
        """Add stream to monitoring."""
        if stream_url not in self.monitored_streams:
            self.monitored_streams.append(stream_url)
            self.save_monitored_streams()
            return True
        return False
    
    def remove_stream(self, stream_url):
        """Remove stream from monitoring."""
        if stream_url in self.monitored_streams:
            self.monitored_streams.remove(stream_url)
            self.save_monitored_streams()
            return True
        return False
    
    def run(self):
        """Monitor streams for status changes."""
        self.running = True
        
        while self.running:
            for stream_url in self.monitored_streams:
                try:
                    if self._check_stream_live(stream_url):
                        self.stream_live.emit(stream_url)
                    else:
                        self.stream_offline.emit(stream_url)
                except Exception as e:
                    logger.error(f"Error checking stream {stream_url}: {e}")
            
            # Sleep for check interval
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _check_stream_live(self, stream_url):
        """Check if a stream is live using streamlink."""
        try:
            self.status_update.emit(f"Checking if {stream_url} is live...")
            
            # Format the URL properly if it's just a channel name
            if "twitch.tv" not in stream_url:
                stream_url = f"https://twitch.tv/{stream_url}"
            
            # Use streamlink to check stream status
            cmd = ["streamlink", "--stream-url", stream_url, "best"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Stream is live if we get a URL and return code is 0
            return result.returncode == 0 and "https://" in result.stdout
            
        except Exception as e:
            logger.error(f"Error checking stream status: {e}")
            return False
    
    def stop(self):
        """Stop monitoring."""
        self.running = False


class ClipCreator(QThread):
    """Creates clips from buffer segments."""
    
    # Signals
    progress_update = pyqtSignal(int)  # 0-100
    status_update = pyqtSignal(str)  # status message
    clip_created = pyqtSignal(str)  # output path
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, stream_buffer, start_time_ago, duration, output_path, format_type="mp4"):
        """Initialize clip creator."""
        super().__init__()
        self.stream_buffer = stream_buffer
        self.start_time_ago = start_time_ago
        self.duration = duration
        self.output_path = output_path
        self.format_type = format_type
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    def run(self):
        """Create clip from buffer segments."""
        try:
            self.status_update.emit(f"Creating clip: {os.path.basename(self.output_path)}")
            self.progress_update.emit(5)
            
            # Get segments for clip
            segments = self.stream_buffer.get_segments_for_clip(
                self.start_time_ago, self.duration
            )
            
            if not segments:
                self.error_occurred.emit("No segments found for specified time range")
                return
            
            # Create temp file list for FFmpeg
            temp_list_path = os.path.join(
                os.path.dirname(self.output_path),
                f"temp_list_{int(time.time())}.txt"
            )
            
            with open(temp_list_path, 'w') as f:
                for segment in segments:
                    f.write(f"file '{segment['file']}'\n")
            
            self.progress_update.emit(10)
            
            # Build FFmpeg command with proper audio handling
            cmd = [
                FFMPEG_PATH, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", temp_list_path,
                "-c:v", "libx264",  # Re-encode video for compatibility
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",      # Ensure AAC audio
                "-b:a", "128k",     # Set audio bitrate
                "-ac", "2",         # Stereo audio
                "-ar", "44100"      # Audio sample rate
            ]
            
            # Add format-specific options
            if self.format_type == "mp4":
                cmd.extend(["-movflags", "+faststart"])
            
            cmd.append(self.output_path)
            
            # Execute FFmpeg command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor progress
            for line in process.stderr:
                if "time=" in line:
                    try:
                        time_str = line.split("time=")[1].split()[0]
                        parts = time_str.split(':')
                        if len(parts) == 3:
                            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                            current_seconds = h * 3600 + m * 60 + s
                            progress = min(int((current_seconds / self.duration) * 90) + 10, 100)
                            self.progress_update.emit(progress)
                    except:
                        pass
            
            # Wait for process to complete
            process.wait()
            
            # Check if output file exists and has non-zero size
            if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0:
                self.progress_update.emit(100)
                self.clip_created.emit(self.output_path)
            else:
                self.error_occurred.emit("Failed to create clip. FFmpeg error or output file is empty.")
            
            # Clean up temp file
            try:
                os.remove(temp_list_path)
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error creating clip: {traceback.format_exc()}")
            self.error_occurred.emit(f"Error creating clip: {str(e)}")


class ClipEditor(QThread):
    """Edits clips with effects, transitions, etc."""
    
    # Signals
    progress_update = pyqtSignal(int)  # 0-100
    status_update = pyqtSignal(str)  # status message
    edit_complete = pyqtSignal(str)  # output path
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, input_path, output_path, edit_options=None):
        """Initialize clip editor."""
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.edit_options = edit_options or {}
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    def run(self):
        """Edit clip with specified options."""
        try:
            self.status_update.emit(f"Editing clip: {os.path.basename(self.output_path)}")
            self.progress_update.emit(5)
            
            # Build FFmpeg command based on edit options
            cmd = [FFMPEG_PATH, "-y", "-i", self.input_path]
            
            # Apply filters
            filters = []
            
            # Speed adjustment
            if "speed" in self.edit_options:
                speed = float(self.edit_options["speed"])
                if speed != 1.0:
                    # Audio tempo filter
                    filters.append(f"atempo={min(2.0, max(0.5, speed))}")
                    # Video setpts filter
                    filters.append(f"setpts={1/speed}*PTS")
            
            # Brightness/contrast
            if "brightness" in self.edit_options or "contrast" in self.edit_options:
                brightness = self.edit_options.get("brightness", 0)  # -1.0 to 1.0
                contrast = self.edit_options.get("contrast", 1)  # 0.0 to 2.0
                filters.append(f"eq=brightness={brightness}:contrast={contrast}")
            
            # Apply filters if any
            if filters:
                audio_filters = [f for f in filters if f.startswith("atempo")]
                video_filters = [f for f in filters if not f.startswith("atempo")]
                
                if audio_filters:
                    cmd.extend(["-af", ",".join(audio_filters)])
                if video_filters:
                    cmd.extend(["-vf", ",".join(video_filters)])
            
            # Format-specific options
            if self.output_path.endswith(".mp4"):
                cmd.extend(["-movflags", "+faststart"])
            
            # Output quality
            cmd.extend([
                "-c:v", self.edit_options.get("video_codec", "libx264"),
                "-crf", str(self.edit_options.get("quality", 23)),
                "-preset", self.edit_options.get("preset", "fast"),
                "-c:a", self.edit_options.get("audio_codec", "aac"),
                "-b:a", self.edit_options.get("audio_bitrate", "128k")
            ])
            
            # Output file
            cmd.append(self.output_path)
            
            # Execute FFmpeg command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Get input duration first
            input_duration = 0
            try:
                probe_cmd = [
                    FFPROBE_PATH, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json",
                    self.input_path
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                data = json.loads(result.stdout)
                input_duration = float(data["format"]["duration"])
            except:
                input_duration = 60  # Default assumption
            
            # Monitor progress
            for line in process.stderr:
                if "time=" in line:
                    try:
                        time_str = line.split("time=")[1].split()[0]
                        parts = time_str.split(':')
                        if len(parts) == 3:
                            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                            current_seconds = h * 3600 + m * 60 + s
                            progress = min(int((current_seconds / input_duration) * 90) + 10, 100)
                            self.progress_update.emit(progress)
                    except:
                        pass
            
            # Wait for process to complete
            process.wait()
            
            # Check if output file exists and has non-zero size
            if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0:
                self.progress_update.emit(100)
                self.edit_complete.emit(self.output_path)
            else:
                self.error_occurred.emit("Failed to edit clip. FFmpeg error or output file is empty.")
            
        except Exception as e:
            logger.error(f"Error editing clip: {traceback.format_exc()}")
            self.error_occurred.emit(f"Error editing clip: {str(e)}")