#!/usr/bin/env python3
"""
Real-time viral moment detection for BeastClipper
Detects potentially viral moments from live streaming buffer
"""

import time
import logging
import re
import subprocess
import os
import tempfile
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

# Configure logger
logger = logging.getLogger("BeastClipper")


class ViralMomentDetector(QThread):
    """Detects potential viral moments in a stream buffer in real-time."""
    
    moment_detected = pyqtSignal(dict)  # Emits viral moment info (timestamp, score, description)
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, stream_buffer, sensitivity=0.7, check_interval=5):
        """
        Initialize viral moment detector.
        
        Args:
            stream_buffer: StreamBuffer instance to monitor
            sensitivity: Detection sensitivity (0.0 to 1.0)
            check_interval: How often to check for viral moments (seconds)
        """
        super().__init__()
        
        self.stream_buffer = stream_buffer
        self.sensitivity = sensitivity
        self.check_interval = check_interval
        
        # Detection state
        self.running = False
        self.frame_diff_history = deque(maxlen=30)  # Store recent frame differences
        self.audio_peaks = deque(maxlen=10)  # Store recent audio peaks
        self.motion_scores = deque(maxlen=10)  # Store recent motion scores
        
        # Detected moments to avoid duplicates (timestamp -> score)
        self.detected_moments = {}
        
        # Temp files for detection
        self.temp_dir = tempfile.mkdtemp(prefix="beastclipper_detection_")
        
    def run(self):
        """Main detection thread."""
        self.running = True
        self.status_update.emit("Starting viral moment detection...")
        
        try:
            prev_frame = None
            frame_count = 0
            last_analysis_time = 0
            
            while self.running:
                current_time = time.time()
                
                # Only analyze every check_interval seconds
                if current_time - last_analysis_time < self.check_interval:
                    time.sleep(0.5)
                    continue
                
                last_analysis_time = current_time
                
                # Get recent segments from buffer
                buffer_status = self.stream_buffer.get_buffer_status()
                
                if buffer_status['segments'] == 0:
                    # No buffer data yet
                    time.sleep(1)
                    continue
                
                self.status_update.emit("Analyzing recent stream content...")
                
                # Get a short chunk to analyze
                # We'll use 10 seconds from the most recent part of the buffer
                segments = self.stream_buffer.get_segments_for_clip(10, 10)
                
                if not segments:
                    time.sleep(1)
                    continue
                
                # Create a temporary file to analyze
                temp_file = os.path.join(self.temp_dir, f"analysis_{int(time.time())}.ts")
                
                # Concatenate segments
                with open(temp_file, 'wb') as outfile:
                    for segment in segments:
                        with open(segment['file'], 'rb') as infile:
                            outfile.write(infile.read())
                
                # Analyze for scene changes
                scene_changes = self._detect_scene_changes(temp_file)
                
                # Analyze for audio peaks
                audio_peaks = self._detect_audio_peaks(temp_file)
                
                # Analyze for motion
                motion_scores = self._detect_motion(temp_file)
                
                # Combine signals for viral moment detection
                if scene_changes or audio_peaks or motion_scores:
                    self._combine_signals(scene_changes, audio_peaks, motion_scores, current_time)
                
                # Clean up
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                # Sleep before next analysis
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in viral moment detection: {str(e)}")
            self.error_occurred.emit(f"Detection Error: {str(e)}")
        
        finally:
            self.running = False
            self.status_update.emit("Viral moment detection stopped")
            
            # Clean up temp directory
            try:
                for file in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, file))
                os.rmdir(self.temp_dir)
            except:
                pass
    
    def _detect_scene_changes(self, video_file):
        """Detect scene changes in video file."""
        try:
            scene_changes = []
            
            # Open video
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                return []
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default FPS
            
            # Process frames
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every 5th frame for performance
                if frame_count % 5 != 0:
                    frame_count += 1
                    continue
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect scene changes
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(gray, prev_frame)
                    diff_mean = np.mean(diff)
                    self.frame_diff_history.append(diff_mean)
                    
                    # Check for scene change
                    if len(self.frame_diff_history) >= 10:
                        avg_diff = sum(self.frame_diff_history) / len(self.frame_diff_history)
                        if diff_mean > avg_diff * (1 + self.sensitivity):
                            timestamp = frame_count / fps
                            score = min(diff_mean / 255.0, 1.0)
                            scene_changes.append((timestamp, score))
                
                # Store current frame for next iteration
                prev_frame = gray
                frame_count += 1
            
            # Clean up
            cap.release()
            
            return scene_changes
            
        except Exception as e:
            logger.error(f"Error detecting scene changes: {str(e)}")
            return []
    
    def _detect_audio_peaks(self, video_file):
        """Detect audio peaks in video file."""
        try:
            audio_peaks = []
            
            # Use ffmpeg to extract audio levels
            cmd = [
                "ffmpeg",
                "-i", video_file,
                "-af", "loudnorm=print_format=json,volumedetect",
                "-f", "null", "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse audio level information from output
            output = result.stderr
            level_pattern = re.compile(r"max_volume: ([-\d.]+) dB")
            matches = level_pattern.findall(output)
            
            # Also look for "silence" sections
            silence_pattern = re.compile(r"silence_start: ([\d.]+)")
            silence_matches = silence_pattern.findall(output)
            
            # Process audio results
            if matches:
                # Convert dB levels to normalized scores
                for db_str in matches:
                    # Convert dB to a score (normalize -30dB to 0dB range to 0-1)
                    db = float(db_str)
                    score = min(max((db + 30) / 30, 0), 1)
                    if score > 0.6:  # Only track significant audio peaks
                        audio_peaks.append((0, score))  # We don't have timestamp info here
            
            # Add silence start points (could be interesting transitions)
            for silence_time in silence_matches:
                timestamp = float(silence_time)
                audio_peaks.append((timestamp, 0.7))
            
            return audio_peaks
            
        except Exception as e:
            logger.error(f"Error detecting audio peaks: {str(e)}")
            return []
    
    def _detect_motion(self, video_file):
        """Detect significant motion in video file."""
        try:
            motion_scores = []
            
            # Open video
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                return []
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default FPS
            
            # Process frames
            prev_frame = None
            frame_count = 0
            skip_frames = max(1, int(fps / 6))  # Process 6 frames per second
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every Nth frame for performance
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Check for motion
                if prev_frame is not None:
                    # Apply thresholding to find areas with significant change
                    diff = cv2.absdiff(gray, prev_frame)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Look for large motion areas
                    significant_motion = False
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > (frame.shape[0] * frame.shape[1] * 0.05):  # At least 5% of frame
                            significant_motion = True
                            break
                    
                    if significant_motion:
                        timestamp = frame_count / fps
                        motion_score = sum(cv2.contourArea(c) for c in contours) / (frame.shape[0] * frame.shape[1])
                        motion_score = min(motion_score, 1.0)
                        motion_scores.append((timestamp, motion_score))
                
                # Store current frame for next iteration
                prev_frame = gray
                frame_count += 1
            
            # Clean up
            cap.release()
            
            return motion_scores
            
        except Exception as e:
            logger.error(f"Error detecting motion: {str(e)}")
            return []
    
    def _combine_signals(self, scene_changes, audio_peaks, motion_scores, current_time):
        """Combine different signals to detect viral moments."""
        # Boost scores where multiple signals align
        combined_scores = []
        
        # Process scene changes as primary indicators
        for timestamp, score in scene_changes:
            # Check if there are nearby audio peaks to boost the score
            for _, audio_score in audio_peaks:
                # We don't have exact timestamps for audio, so just boost if there are peaks
                score = min(1.0, score + (audio_score * 0.3))
            
            # Check if there's significant motion around this time
            for motion_time, motion_score in motion_scores:
                if abs(timestamp - motion_time) < 2:  # Within 2 seconds
                    score = min(1.0, score + (motion_score * 0.2))
            
            combined_scores.append((timestamp, score))
        
        # Add any significant motion moments that weren't already covered
        for timestamp, score in motion_scores:
            if score > self.sensitivity * 1.2:  # Higher threshold for motion-only moments
                # Check if we already have a moment at this timestamp
                has_existing = False
                for existing_time, _ in combined_scores:
                    if abs(timestamp - existing_time) < 2:
                        has_existing = True
                        break
                
                if not has_existing:
                    combined_scores.append((timestamp, score))
        
        # Filter by sensitivity threshold
        filtered_moments = [(time, score) for time, score in combined_scores if score > self.sensitivity]
        
        # Convert to real stream timestamps (relative to current time)
        buffer_status = self.stream_buffer.get_buffer_status()
        buffer_duration = buffer_status['duration']
        
        for rel_timestamp, score in filtered_moments:
            # Convert clip-relative timestamp to buffer-relative timestamp
            buffer_timestamp = buffer_duration - (10 - rel_timestamp)
            
            # Only report if not recently detected (within 5 seconds)
            is_duplicate = False
            for existing_time in self.detected_moments:
                if abs(buffer_timestamp - existing_time) < 5:
                    is_duplicate = True
                    break
            
            if not is_duplicate and buffer_timestamp > 0:
                # Store this moment
                self.detected_moments[buffer_timestamp] = score
                
                # Create moment info
                moment_info = {
                    'timestamp': buffer_timestamp,
                    'score': score,
                    'time_ago': buffer_timestamp,  # Seconds ago from now
                    'detected_at': current_time,
                    'description': self._get_moment_description(score)
                }
                
                # Emit the moment
                self.moment_detected.emit(moment_info)
                logger.info(f"Viral moment detected: {buffer_timestamp:.1f}s ago (Score: {score:.2f})")
    
    def _get_moment_description(self, score):
        """Generate a description based on the moment score."""
        if score > 0.9:
            return "Exceptional moment detected!"
        elif score > 0.8:
            return "Very high activity detected"
        elif score > 0.7:
            return "Significant moment detected"
        else:
            return "Interesting moment detected"
    
    def stop(self):
        """Stop the detector."""
        self.running = False


class ViralMomentManager:
    """Manages detected viral moments for UI display and clipping."""
    
    def __init__(self, max_moments=10):
        """Initialize the viral moment manager."""
        self.max_moments = max_moments
        self.moments = []  # List of moment dictionaries ordered by detection time
    
    def add_moment(self, moment_info):
        """Add a new viral moment."""
        # Add to list
        self.moments.append(moment_info)
        
        # Keep only the latest max_moments
        if len(self.moments) > self.max_moments:
            self.moments.pop(0)
    
    def get_moments(self):
        """Get all stored moments."""
        return self.moments
    
    def get_moment_by_index(self, index):
        """Get a specific moment by index."""
        if 0 <= index < len(self.moments):
            return self.moments[index]
        return None
    
    def clear_moments(self):
        """Clear all stored moments."""
        self.moments.clear()
    
    def update_times_ago(self):
        """Update the time_ago field for all moments based on current time."""
        current_time = time.time()
        for moment in self.moments:
            moment['time_ago'] = current_time - moment['detected_at'] + moment['time_ago']
            moment['detected_at'] = current_time
