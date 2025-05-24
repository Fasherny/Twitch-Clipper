#!/usr/bin/env python3
"""
FIXED Twitch Stream Buffer with Ad Handling
Records at user's chosen quality by waiting out ads or recording through them
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
    """Twitch stream buffer that handles ads by recording through them."""
    
    # Signals
    buffer_progress = pyqtSignal(int, int)  # current, total
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    stream_info_updated = pyqtSignal(dict)
    ad_status_update = pyqtSignal(str)  # New signal for ad status
    
    def __init__(self, stream_url, buffer_duration=300, resolution="best", 
                 segment_length=10, temp_manager=None, buffer_directory=None):
        """Initialize Twitch stream buffer with enhanced ad handling."""
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
        self.recording_process = None
        
        # Use unified temp directory for buffer
        self.temp_dir = os.path.join(
            BASE_TEMP_DIR,
            "buffer",
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Ad handling state
        self.ad_detected = False
        self.ad_segments = []  # Track which segments contain ads
        self.recording_started = False
        self.segment_index = 0
        
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
        """Main buffer thread - records through ads instead of avoiding them."""
        self.running = True
        
        try:
            # Validate stream first
            self.status_update.emit("Checking Twitch stream...")
            
            if not self._validate_stream_simple():
                self.error_occurred.emit("Stream is offline or invalid. Please check the channel name.")
                return
            
            # Get basic stream information
            self.status_update.emit("Getting stream information...")
            stream_info = self._get_stream_info_simple()
            
            if stream_info:
                self.stream_info_updated.emit(stream_info)
            
            # Start recording with user's chosen quality
            self.status_update.emit(f"Starting recording at {self.resolution}...")
            
            # Map resolution to streamlink quality
            quality_map = {
                "1080p": "1080p60,1080p,best",
                "720p": "720p60,720p,best",
                "480p": "480p,best",
                "360p": "360p,worst",
                "best": "best"
            }
            quality = quality_map.get(self.resolution, "best")
            
            # Start recording - will record through ads
            if self._start_recording_through_ads(quality):
                logger.info("Recording started successfully")
            else:
                self.error_occurred.emit("Failed to start recording")
                
        except Exception as e:
            logger.error(f"Buffer thread error: {traceback.format_exc()}")
            self.error_occurred.emit(f"Buffer error: {str(e)}")
        
        finally:
            self.running = False
            self._cleanup()
            self.status_update.emit("Buffer stopped")
    
    def _validate_stream_simple(self):
        """Simple, fast stream validation."""
        try:
            logger.info(f"Quick validation for: {self.stream_url}")
            
            # Simple validation command
            cmd = ["streamlink", self.stream_url, "worst", "--stream-url"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "http" in result.stdout:
                logger.info("Stream validation successful")
                return True
            
            logger.warning(f"Stream validation failed: {result.stderr[:200] if result.stderr else 'No error output'}")
            return False
            
        except Exception as e:
            logger.error(f"Stream validation error: {str(e)}")
            return False
    
    def _get_stream_info_simple(self):
        """Get basic stream information without complex operations."""
        try:
            # Simple JSON info command
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
    
    def _start_recording_through_ads(self, quality):
        """Start recording that will continue through ads."""
        try:
            # Create special output template that handles interruptions
            output_template = os.path.join(self.temp_dir, "stream.m3u8")
            
            # Use HLS passthrough to get the raw stream including ads
            cmd = [
                "streamlink",
                "--force",
                "--retry-streams", "10",
                "--retry-open", "10",
                "--stream-timeout", "120",
                "--hls-live-restart",
                "--hls-timeout", "120",
                "--hls-playlist-reload-attempts", "30",
                "--hls-segment-attempts", "10",
                "--hls-segment-timeout", "60",
                "--hls-segment-ignore-names", ".*preroll.*,.*midroll.*",  # Try to filter ad segments
                "--twitch-disable-hosting",
                "--twitch-disable-reruns",
                "--player-passthrough", "hls",  # Pass through HLS directly
                "--output", output_template,
                self.stream_url,
                quality
            ]
            
            logger.info(f"Starting recording with command: {' '.join(cmd)}")
            
            # Start the recording process
            if os.name == 'nt':  # Windows
                self.recording_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # Unix/Linux
                self.recording_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    preexec_fn=os.setsid
                )
            
            # Start direct segment recording in parallel
            segment_thread = threading.Thread(target=self._record_segments_directly, args=(quality,))
            segment_thread.daemon = True
            segment_thread.start()
            
            # Monitor the process
            monitor_thread = threading.Thread(target=self._monitor_recording_with_ad_detection)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Wait to see if recording starts
            time.sleep(5)
            
            # Check if we're getting content
            if self.recording_process.poll() is None:
                self.status_update.emit("Recording active - will record through any ads")
                self.ad_status_update.emit("✅ Recording at chosen quality")
                return True
            else:
                logger.error("Recording process failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False
    
    def _record_segments_directly(self, quality):
        """Record segments directly using ffmpeg to bypass ad detection."""
        try:
            # Get direct stream URL
            cmd = ["streamlink", "--stream-url", self.stream_url, quality]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0 or not result.stdout:
                logger.error("Could not get direct stream URL")
                return
            
            stream_url = result.stdout.strip()
            logger.info(f"Got direct stream URL: {stream_url[:50]}...")
            
            # Record segments using ffmpeg
            while self.running:
                segment_file = os.path.join(self.temp_dir, f"segment_{self.segment_index:05d}.ts")
                
                # Record one segment
                ffmpeg_cmd = [
                    FFMPEG_PATH,
                    "-i", stream_url,
                    "-t", str(self.segment_length),
                    "-c", "copy",
                    "-bsf:a", "aac_adtstoasc",
                    "-f", "mpegts",
                    "-y",
                    segment_file
                ]
                
                logger.debug(f"Recording segment {self.segment_index}")
                
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait for segment to complete
                process.wait()
                
                # Check if segment was created and has content
                if os.path.exists(segment_file) and os.path.getsize(segment_file) > 10000:
                    with self.segment_lock:
                        self.segments.append({
                            'file': segment_file,
                            'index': self.segment_index,
                            'timestamp': time.time(),
                            'duration': self.segment_length,
                            'size': os.path.getsize(segment_file),
                            'method': 'direct_ffmpeg'
                        })
                        
                        self._prune_old_segments()
                    
                    logger.info(f"Added segment {self.segment_index}")
                    self.recording_started = True
                    self.segment_index += 1
                else:
                    logger.warning(f"Segment {self.segment_index} was empty or failed")
                    time.sleep(2)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Error in direct segment recording: {e}")
    
    def _monitor_recording_with_ad_detection(self):
        """Monitor recording and detect when ads are playing."""
        last_check = time.time()
        consecutive_no_growth = 0
        
        while self.running and self.recording_process:
            try:
                # Check if process is still running
                if self.recording_process.poll() is not None:
                    logger.warning("Recording process ended")
                    break
                
                # Read stderr for ad detection
                if self.recording_process.stderr:
                    try:
                        # Non-blocking read
                        line = self.recording_process.stderr.readline()
                        if line:
                            line_lower = line.lower()
                            if any(word in line_lower for word in ["commercial", "ad-free", "subscribe", "preroll", "midroll"]):
                                self.ad_detected = True
                                self.ad_status_update.emit("📺 Ads detected - recording continues...")
                            elif "segment" in line_lower and "downloaded" in line_lower:
                                if self.ad_detected:
                                    self.ad_detected = False
                                    self.ad_status_update.emit("✅ Content resumed - recording active")
                    except:
                        pass
                
                # Check buffer growth
                current_time = time.time()
                if current_time - last_check >= 5:
                    with self.segment_lock:
                        segment_count = len(self.segments)
                    
                    if segment_count == 0 and not self.recording_started:
                        consecutive_no_growth += 1
                        if consecutive_no_growth > 6:  # 30 seconds no segments
                            self.ad_status_update.emit("⏳ Waiting for stream content...")
                    else:
                        consecutive_no_growth = 0
                        if not self.recording_started:
                            self.recording_started = True
                            self.ad_status_update.emit("✅ Recording started successfully")
                    
                    last_check = current_time
                
                # Update progress
                self._update_progress()
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error monitoring recording: {e}")
                time.sleep(2)
    
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
    
    def _update_progress(self):
        """Update progress indicators."""
        try:
            with self.segment_lock:
                current_duration = len(self.segments) * self.segment_length
            
            # Emit progress
            self.buffer_progress.emit(min(current_duration, self.buffer_duration), self.buffer_duration)
            
            # Update status
            if self.ad_detected:
                status_msg = f"Buffering: {current_duration}s / {self.buffer_duration}s (Recording through ads)"
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
                'method': 'direct_recording',
                'ad_detected': self.ad_detected
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
        self.running = False
        
        # Stop the recording process
        if self.recording_process:
            try:
                # Try graceful termination first
                if os.name == 'nt':
                    self.recording_process.terminate()
                else:
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(self.recording_process.pid), signal.SIGTERM)
                    else:
                        self.recording_process.terminate()
                
                # Wait a bit for graceful shutdown
                try:
                    self.recording_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't respond
                    if os.name == 'nt':
                        self.recording_process.kill()
                    else:
                        if hasattr(os, 'killpg'):
                            os.killpg(os.getpgid(self.recording_process.pid), signal.SIGKILL)
                        else:
                            self.recording_process.kill()
                    
                logger.info("Recording process stopped")
            except Exception as e:
                logger.error(f"Error stopping recording process: {e}")
    
    def _cleanup(self):
        """Clean up resources."""
        try:
            # Stop recording process
            if self.recording_process:
                self.stop()
            
            # Register temp directory for cleanup
            if self.temp_manager:
                self.temp_manager.register_temp_file(self.temp_dir, lifetime=3600)
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


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
            
            # Build FFmpeg command
            cmd = [
                FFMPEG_PATH, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", temp_list_path,
                "-c:v", "copy",
                "-c:a", "copy"
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