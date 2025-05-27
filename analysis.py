#!/usr/bin/env python3
"""
Content analysis module for BeastClipper
Handles viral moment detection and chat activity monitoring
"""

import time
import logging
import re
import subprocess
from collections import deque

import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from PyQt5.QtCore import QThread, pyqtSignal
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
import tempfile

# Configure logger
logger = logging.getLogger("BeastClipper")


# ======================
# Content Analyzer
# ======================

class ContentAnalyzer(QThread):
    """Analyzes video content to detect potentially viral moments."""
    
    analysis_complete = pyqtSignal(list)  # List of potential viral moments (start_time, end_time, score)
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    
    def __init__(self, video_file, sensitivity=0.7, min_clip_length=10, max_clip_length=60):
        """
        Initialize content analyzer for viral moment detection.
        
        Args:
            video_file: Path to video file to analyze
            sensitivity: Detection sensitivity (0.0 to 1.0)
            min_clip_length: Minimum viral clip length in seconds
            max_clip_length: Maximum viral clip length in seconds
        """
        super().__init__()
        self.video_file = video_file
        self.sensitivity = sensitivity
        self.min_clip_length = min_clip_length
        self.max_clip_length = max_clip_length
    
    def run(self):
        try:
            self.status_update.emit("Loading video for analysis...")
            self.progress_update.emit(5)
            
            # Open video file
            cap = cv2.VideoCapture(self.video_file)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default FPS if not detected
            
            video_duration = total_frames / fps
            logger.info(f"Analyzing video: {total_frames} frames, {fps} FPS, {video_duration:.2f} seconds")
            
            # Initialize variables for analysis
            self.status_update.emit("Analyzing video for viral moments...")
            prev_frame = None
            frame_diff_history = deque(maxlen=int(fps * 5))  # Store 5 seconds of frame differences
            audio_peaks = []
            scene_changes = []
            interesting_moments = []
            
            # Process video frames
            frame_count = 0
            skip_frames = max(1, int(fps / 10))  # Analyze at most 10 frames per second
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every Nth frame for performance
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                
                # Update progress
                progress = min(int((frame_count / total_frames) * 80) + 5, 85)
                if frame_count % (skip_frames * 10) == 0:
                    self.progress_update.emit(progress)
                    self.status_update.emit(f"Analyzing frame {frame_count}/{total_frames}...")
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect scene changes
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(gray, prev_frame)
                    diff_mean = np.mean(diff)
                    frame_diff_history.append(diff_mean)
                    
                    # Check for scene change
                    if len(frame_diff_history) >= 10:
                        avg_diff = sum(frame_diff_history) / len(frame_diff_history)
                        if diff_mean > avg_diff * (1 + self.sensitivity):
                            timestamp = frame_count / fps
                            scene_changes.append((timestamp, diff_mean / 255.0))
                            logger.debug(f"Detected scene change at {timestamp:.2f}s (score: {diff_mean / 255.0:.2f})")
                    
                    # Check for motion (interesting moments)
                    if frame_count % (skip_frames * 5) == 0:  # Less frequent check
                        # Apply thresholding to find areas with significant change
                        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Look for large motion areas (potentially interesting)
                        significant_motion = False
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > (frame.shape[0] * frame.shape[1] * 0.05):  # At least 5% of frame
                                significant_motion = True
                                break
                        
                        if significant_motion:
                            timestamp = frame_count / fps
                            motion_score = sum(cv2.contourArea(c) for c in contours) / (frame.shape[0] * frame.shape[1])
                            interesting_moments.append((timestamp, min(motion_score, 1.0)))
                            logger.debug(f"Detected significant motion at {timestamp:.2f}s (score: {motion_score:.2f})")
                
                # Store current frame for next iteration
                prev_frame = gray
                frame_count += 1
            
            # Extract audio for analysis
            self.status_update.emit("Analyzing audio...")
            self.progress_update.emit(90)
            
            # Use ffmpeg to extract audio levels
            audio_analysis_cmd = [
                "ffmpeg",
                "-i", self.video_file,
                "-af", "loudnorm=print_format=json,volumedetect",
                "-f", "null", "-"
            ]
            
            result = subprocess.run(audio_analysis_cmd, capture_output=True, text=True)
            
            # Parse audio level information from output
            output = result.stderr
            level_pattern = re.compile(r"max_volume: ([-\d.]+) dB")
            matches = level_pattern.findall(output)
            
            # Also look for "silence" sections (could be interesting transition points)
            silence_pattern = re.compile(r"silence_start: ([\d.]+)")
            silence_matches = silence_pattern.findall(output)
            
            # Process audio results
            if matches:
                # Convert dB levels to normalized scores and create timestamp mapping
                # This is simplified; ideally we'd have timestamps for each measurement
                if len(matches) > 1:
                    # Estimate timestamps by distributing evenly across the video
                    step = video_duration / len(matches)
                    for i, level_db in enumerate(matches):
                        timestamp = i * step
                        # Convert dB to a score (normalize -20dB to 0dB range to 0-1)
                        db = float(level_db)
                        score = min(max((db + 30) / 30, 0), 1)  # Normalize to 0-1
                        if score > 0.6:  # Only track significant audio peaks
                            audio_peaks.append((timestamp, score))
            
            # Add silence start points (could be interesting transitions)
            for silence_time in silence_matches:
                timestamp = float(silence_time)
                audio_peaks.append((timestamp, 0.7))  # Moderate score for silence transitions
            
            # Combine scene changes, motion and audio peaks to identify potential viral moments
            self.status_update.emit("Identifying potential viral moments...")
            self.progress_update.emit(95)
            
            viral_moments = []
            
            # Process scene changes as primary indicators
            for timestamp, score in scene_changes:
                # Check if there are nearby audio peaks to boost the score
                for audio_time, audio_score in audio_peaks:
                    if abs(timestamp - audio_time) < 3:  # Within 3 seconds
                        # Boost score if audio peak nearby
                        score = min(1.0, score + (audio_score * 0.3))
                
                # Check if there's significant motion around this time
                for motion_time, motion_score in interesting_moments:
                    if abs(timestamp - motion_time) < 2:  # Within 2 seconds
                        score = min(1.0, score + (motion_score * 0.2))
                
                # Add to viral moments if score is high enough
                if score > self.sensitivity:
                    # Always create a valid window of min_clip_length
                    start_time = max(0, timestamp - self.min_clip_length / 2)
                    end_time = min(start_time + self.min_clip_length, video_duration)
                    if end_time > start_time and (end_time - start_time) >= self.min_clip_length:
                        overlaps = False
                        for i, (existing_start, existing_end, _) in enumerate(viral_moments):
                            if (start_time <= existing_end and end_time >= existing_start):
                                overlaps = True
                                if score > viral_moments[i][2]:
                                    viral_moments[i] = (min(start_time, existing_start), max(end_time, existing_end), score)
                                break
                        if not overlaps:
                            viral_moments.append((start_time, end_time, score))
            
            # Add any significant motion moments that weren't already covered
            for timestamp, score in interesting_moments:
                if score > self.sensitivity * 1.2:  # Higher threshold for motion-only moments
                    start_time = max(0, timestamp - self.min_clip_length / 2)
                    end_time = min(start_time + self.min_clip_length, video_duration)
                    if end_time > start_time and (end_time - start_time) >= self.min_clip_length:
                        overlaps = False
                        for existing_start, existing_end, _ in viral_moments:
                            if (start_time <= existing_end and end_time >= existing_start):
                                overlaps = True
                                break
                        if not overlaps:
                            viral_moments.append((start_time, end_time, score))
            
            # Sort by score (descending)
            viral_moments.sort(key=lambda x: x[2], reverse=True)
            
            # Clean up
            cap.release()
            
            # Log results
            logger.info(f"Found {len(viral_moments)} potential viral moments")
            for i, (start, end, score) in enumerate(viral_moments):
                logger.info(f"Moment {i+1}: {start:.2f}s to {end:.2f}s (Score: {score:.2f})")
            
            self.progress_update.emit(100)
            self.status_update.emit("Analysis complete")
            
            # If no viral moments found but we have scene changes, lower threshold and try again
            if not viral_moments and scene_changes:
                reduced_sensitivity = max(self.sensitivity * 0.7, 0.3)  # Reduce by 30% but not below 0.3
                logger.info(f"No viral moments found with sensitivity {self.sensitivity}, trying with {reduced_sensitivity}")
                
                # Process again with lower threshold
                for timestamp, score in scene_changes:
                    if score > reduced_sensitivity:
                        start_time = max(0, timestamp - self.min_clip_length / 2)
                        end_time = min(start_time + self.min_clip_length, video_duration)
                        if end_time > start_time and (end_time - start_time) >= self.min_clip_length:
                            viral_moments.append((start_time, end_time, score))
                
                viral_moments.sort(key=lambda x: x[2], reverse=True)
            
            self.analysis_complete.emit(viral_moments)
            
        except Exception as e:
            logger.error(f"Error in content analysis: {str(e)}")
            self.status_update.emit(f"Analysis Error: {str(e)}")
            self.analysis_complete.emit([])


# =================
# Chat Monitor
# =================

class ChatMonitor(QThread):
    """Monitors stream chat for activity spikes that might indicate viral moments."""
    
    chat_activity_update = pyqtSignal(int)  # Recent message count
    viral_moment_detected = pyqtSignal(int)  # Peak score (0-100)
    status_update = pyqtSignal(str)
    
    def __init__(self, stream_url, threshold=20, check_interval=5):
        super().__init__()
        self.stream_url = stream_url
        self.threshold = threshold
        self.check_interval = check_interval
        self.running = True
        self.driver = None
        self.recent_message_count = 0
        self.message_history = deque(maxlen=12)  # Store 1 minute of message counts
    
    def run(self):
        try:
            self.status_update.emit(f"Starting chat monitor for {self.stream_url}")
            
            # Setup browser for chat monitoring
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--mute-audio")
            
            self.driver = webdriver.Chrome(options=options)
            
            # Navigate to the stream URL - modify for platform (Twitch, YouTube, etc.)
            self.driver.get(self.stream_url)
            
            # Wait for chat to load
            time.sleep(10)
            
            previous_messages = set()
            
            # Main monitoring loop
            while self.running:
                try:
                    # Different platforms have different chat selectors
                    messages = []
                    
                    # Try different chat element selectors (YouTube, Twitch, etc.)
                    if "youtube.com" in self.stream_url:
                        # YouTube Live chat
                        chat_elements = self.driver.find_elements(By.CSS_SELECTOR, "yt-live-chat-text-message-renderer")
                        for elem in chat_elements:
                            message_text = elem.text
                            if message_text:
                                messages.append(message_text)
                    
                    elif "twitch.tv" in self.stream_url:
                        # Twitch chat
                        chat_elements = self.driver.find_elements(By.CSS_SELECTOR, ".chat-line__message")
                        for elem in chat_elements:
                            message_text = elem.text
                            if message_text:
                                messages.append(message_text)
                    
                    else:
                        # Generic approach - look for common chat elements
                        chat_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                            ".chat-message, .chat-line, .message, .chat-item, .comment-item")
                        for elem in chat_elements:
                            message_text = elem.text
                            if message_text:
                                messages.append(message_text)
                    
                    # Find new messages
                    message_set = set(messages)
                    new_messages = message_set - previous_messages
                    previous_messages = message_set
                    
                    # Calculate recent message count
                    self.recent_message_count = len(new_messages)
                    self.message_history.append(self.recent_message_count)
                    
                    # Calculate average and detect spikes
                    if len(self.message_history) >= 6:
                        avg_count = sum(self.message_history) / len(self.message_history)
                        if self.recent_message_count > avg_count * 2 and self.recent_message_count > self.threshold:
                            # Potential viral moment - calculate score
                            score = min(int((self.recent_message_count / self.threshold) * 100), 100)
                            self.viral_moment_detected.emit(score)
                            self.status_update.emit(f"Viral moment detected! Chat activity: {self.recent_message_count} messages")
                    
                    # Emit activity update
                    self.chat_activity_update.emit(self.recent_message_count)
                    
                    # Wait before checking again
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Error monitoring chat: {str(e)}")
                    self.status_update.emit(f"Chat monitor error: {str(e)}")
                    time.sleep(15)  # Longer delay after error
            
        except Exception as e:
            logger.error(f"Error in chat monitor: {str(e)}")
            self.status_update.emit(f"Chat monitor error: {str(e)}")
        
        finally:
            if self.driver:
                self.driver.quit()
    
    def stop(self):
        self.running = False
        if self.driver:
            self.driver.quit()


# ======================
# Streaming Analyzer
# ======================

class StreamingAnalyzer(QThread):
    """Enhanced viral moment detector for streaming video content."""
    analysis_complete = pyqtSignal(list)  # List of (start_time, end_time, score)
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, video_url, sensitivity=0.5, min_clip_length=5, max_clip_length=60, debug_mode=False):
        super().__init__()
        self.video_url = video_url
        self.sensitivity = sensitivity
        self.min_clip_length = min_clip_length
        self.max_clip_length = max_clip_length
        self.debug_mode = debug_mode
        self.temp_dir = os.path.join(tempfile.gettempdir(), f"viral_detector_{int(time.time())}")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.plot_data = {
            'frame_times': [],
            'frame_diffs': [],
            'motion_times': [],
            'motion_values': [],
            'audio_times': [],
            'audio_values': [],
            'scene_change_times': [],
            'scene_change_values': [],
            'viral_moments': []
        }

    def run(self):
        try:
            self.status_update.emit("Getting video stream URL...")
            self.progress_update.emit(5)
            video_source, video_info = self._get_video_source_and_info()
            if not video_source or not video_info:
                self.error_occurred.emit("Could not retrieve video stream URL or info")
                return
            self.status_update.emit("Analyzing video for viral moments...")
            self.progress_update.emit(10)
            audio_file = self._extract_audio(video_source)
            scene_changes, motion_scores = self._analyze_video_content(video_source, video_info)
            audio_peaks = self._analyze_audio_content(audio_file) if audio_file else []
            self.status_update.emit("Detecting viral moments...")
            viral_moments = self._detect_viral_moments(scene_changes, motion_scores, audio_peaks, video_info)
            self.progress_update.emit(100)
            self.status_update.emit("Analysis complete")
            self.analysis_complete.emit(viral_moments)
            if self.debug_mode:
                self._create_visualization(scene_changes, motion_scores, audio_peaks, viral_moments, video_info)
        except Exception as e:
            logger.error(f"Error in streaming analysis: {str(e)}")
            self.error_occurred.emit(f"Analysis Error: {str(e)}")

    def _get_video_source_and_info(self):
        # Use yt-dlp to get direct stream URL and ffprobe for info
        try:
            import subprocess, json
            # Get direct stream URL
            result = subprocess.run([
                "yt-dlp", "--format", "best[height<=1080]", "--get-url", self.video_url
            ], capture_output=True, text=True, check=True)
            direct_url = result.stdout.strip().split('\n')[0]
            # Get video info
            ffprobe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", direct_url
            ]
            info_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            if info_result.returncode != 0:
                return None, None
            data = json.loads(info_result.stdout)
            video_stream = None
            audio_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video' and not video_stream:
                    video_stream = stream
                elif stream.get('codec_type') == 'audio' and not audio_stream:
                    audio_stream = stream
            if not video_stream:
                return None, None
            duration = float(data.get('format', {}).get('duration', 0))
            fps = 0
            if 'r_frame_rate' in video_stream:
                try:
                    num, den = map(int, video_stream['r_frame_rate'].split('/'))
                    fps = num / den
                except:
                    pass
            if fps <= 0:
                fps = 30
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            info = {
                'duration': duration,
                'fps': fps,
                'width': width,
                'height': height,
                'has_audio': audio_stream is not None
            }
            return direct_url, info
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return None, None

    def _extract_audio(self, video_source):
        try:
            audio_file = os.path.join(self.temp_dir, "audio.wav")
            cmd = [
                "ffmpeg", "-y", "-i", video_source, "-ac", "1", "-ar", "44100", "-vn", audio_file
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0 or not os.path.exists(audio_file):
                return None
            return audio_file
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            return None

    def _analyze_video_content(self, video_source, video_info):
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise Exception("Could not open video source")
            fps = video_info['fps']
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = int(video_info['duration'] * fps)
            prev_frame = None
            frame_count = 0
            skip_frames = max(1, int(fps / 8))
            frame_diff_history = deque(maxlen=30)
            scene_changes = []
            motion_scores = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                timestamp = frame_count / fps
                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    diff_mean = np.mean(diff)
                    frame_diff_history.append(diff_mean)
                    if len(frame_diff_history) >= 10:
                        avg_diff = sum(frame_diff_history) / len(frame_diff_history)
                        std_diff = np.std(list(frame_diff_history))
                        threshold = avg_diff + (std_diff * 2 * self.sensitivity)
                        if diff_mean > threshold:
                            score = min((diff_mean - avg_diff) / (std_diff + 0.1), 1.0)
                            scene_changes.append((timestamp, score))
                    if frame_count % (skip_frames * 2) == 0:
                        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            total_area = sum(cv2.contourArea(c) for c in contours)
                            frame_area = frame.shape[0] * frame.shape[1]
                            motion_score = min(total_area / frame_area, 1.0)
                            if motion_score > 0.03:
                                motion_scores.append((timestamp, motion_score))
                prev_frame = gray
                frame_count += 1
            cap.release()
            return scene_changes, motion_scores
        except Exception as e:
            logger.error(f"Error in video analysis: {str(e)}")
            return [], []

    def _analyze_audio_content(self, audio_file):
        try:
            sample_rate, audio_data = wavfile.read(audio_file)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(float)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            window_size = int(sample_rate * 0.5)
            hop_size = int(window_size / 2)
            energy_values = []
            timestamps = []
            for i in range(0, len(audio_data) - window_size, hop_size):
                window = audio_data[i:i+window_size]
                energy = np.sqrt(np.mean(window**2))
                timestamp = i / sample_rate
                energy_values.append(energy)
                timestamps.append(timestamp)
            audio_peaks = []
            if len(energy_values) > 10:
                window_length = min(30, len(energy_values) // 10)
                threshold = np.mean(energy_values) + (np.std(energy_values) * 1.5 * self.sensitivity)
                min_distance = int(sample_rate / hop_size * 0.5)
                peaks, _ = find_peaks(energy_values, height=threshold, distance=min_distance)
                for peak_idx in peaks:
                    peak_time = timestamps[peak_idx]
                    peak_value = energy_values[peak_idx]
                    normalized_score = min((peak_value - np.mean(energy_values)) / (np.std(energy_values) + 0.01), 1.0)
                    audio_peaks.append((peak_time, normalized_score))
            return audio_peaks
        except Exception as e:
            logger.error(f"Error in audio analysis: {str(e)}")
            return []

    def _detect_viral_moments(self, scene_changes, motion_scores, audio_peaks, video_info):
        try:
            duration = video_info['duration']
            segment_size = 0.5
            num_segments = int(duration / segment_size) + 1
            combined_scores = np.zeros(num_segments)
            for time, score in scene_changes:
                segment_idx = int(time / segment_size)
                if 0 <= segment_idx < num_segments:
                    amplified_score = score * (2.0 - self.sensitivity)
                    combined_scores[segment_idx] = max(combined_scores[segment_idx], amplified_score)
            for time, score in audio_peaks:
                segment_idx = int(time / segment_size)
                if 0 <= segment_idx < num_segments:
                    amplified_score = score * (2.0 - self.sensitivity)
                    combined_scores[segment_idx] = max(combined_scores[segment_idx], amplified_score)
            for time, score in motion_scores:
                segment_idx = int(time / segment_size)
                if 0 <= segment_idx < num_segments:
                    if combined_scores[segment_idx] > 0:
                        combined_scores[segment_idx] = min(combined_scores[segment_idx] + (score * 0.3), 1.0)
                    else:
                        if score > 0.2:
                            combined_scores[segment_idx] = score * 0.4
            window_size = int(3.0 / segment_size)
            if window_size > 0 and len(combined_scores) > window_size:
                smoothed = np.convolve(combined_scores, np.ones(window_size)/window_size, mode='same')
                combined_scores = smoothed
            threshold = 0.5 * self.sensitivity
            is_viral = False
            start_idx = 0
            potential_moments = []
            for i in range(num_segments):
                if combined_scores[i] >= threshold and not is_viral:
                    is_viral = True
                    start_idx = i
                elif (combined_scores[i] < threshold or i == num_segments-1) and is_viral:
                    is_viral = False
                    end_idx = i
                    start_time = start_idx * segment_size
                    end_time = end_idx * segment_size
                    if end_time - start_time < self.min_clip_length:
                        padding = (self.min_clip_length - (end_time - start_time)) / 2
                        start_time = max(0, start_time - padding)
                        end_time = min(duration, end_time + padding)
                    if end_time - start_time > self.max_clip_length:
                        peak_idx = start_idx + np.argmax(combined_scores[start_idx:end_idx+1])
                        peak_time = peak_idx * segment_size
                        half_max = self.max_clip_length / 2
                        start_time = max(0, peak_time - half_max)
                        end_time = min(duration, peak_time + half_max)
                    moment_score = np.mean(combined_scores[start_idx:end_idx+1])
                    potential_moments.append((start_time, end_time, moment_score))
            if potential_moments:
                potential_moments.sort(key=lambda x: x[0])
                merged_moments = []
                current = potential_moments[0]
                for i in range(1, len(potential_moments)):
                    current_start, current_end, current_score = current
                    next_start, next_end, next_score = potential_moments[i]
                    if next_start <= current_end:
                        end_time = max(current_end, next_end)
                        score = max(current_score, next_score)
                        current = (current_start, end_time, score)
                    else:
                        merged_moments.append(current)
                        current = potential_moments[i]
                merged_moments.append(current)
                final_moments = []
                for start_time, end_time, score in merged_moments:
                    if end_time - start_time >= self.min_clip_length and score >= 0.4 * self.sensitivity:
                        final_moments.append((start_time, end_time, score))
                final_moments.sort(key=lambda x: x[2], reverse=True)
                return final_moments[:10]
            return []
        except Exception as e:
            logger.error(f"Error in viral moment detection: {str(e)}")
            return []

    def _create_visualization(self, scene_changes, motion_scores, audio_peaks, viral_moments, video_info):
        try:
            plt.figure(figsize=(15, 10))
            if scene_changes:
                plt.subplot(4, 1, 1)
                times, scores = zip(*scene_changes)
                plt.scatter(times, scores, color='red', s=50, label='Scene Changes')
                plt.ylabel('Scene Change Score')
                plt.title('Scene Changes')
                plt.legend()
                plt.grid(True, alpha=0.3)
            if motion_scores:
                plt.subplot(4, 1, 2)
                times, scores = zip(*motion_scores)
                plt.plot(times, scores, 'g-', label='Motion Scores')
                plt.ylabel('Motion Score')
                plt.title('Motion Detection')
                plt.grid(True, alpha=0.3)
            if audio_peaks:
                plt.subplot(4, 1, 3)
                times, scores = zip(*audio_peaks)
                plt.scatter(times, scores, color='purple', marker='o', alpha=0.7, label='Audio Peaks')
                plt.ylabel('Audio Score')
                plt.title('Audio Peaks')
                plt.grid(True, alpha=0.3)
            plt.subplot(4, 1, 4)
            if viral_moments:
                time_axis = np.linspace(0, video_info['duration'], 1000)
                moment_scores = np.zeros_like(time_axis)
                for start, end, score in viral_moments:
                    start_idx = int((start / video_info['duration']) * 1000)
                    end_idx = int((end / video_info['duration']) * 1000)
                    start_idx = max(0, min(start_idx, 999))
                    end_idx = max(0, min(end_idx, 999))
                    if start_idx <= end_idx:
                        moment_scores[start_idx:end_idx+1] = score
                plt.fill_between(time_axis, moment_scores, alpha=0.3, color='orange', label='Viral Moments')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Score')
            plt.title('Detected Viral Moments')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.suptitle(f"Viral Moment Analysis - Sensitivity: {self.sensitivity}", fontsize=16)
            plt.subplots_adjust(top=0.92)
            output_file = os.path.join(self.temp_dir, f"viral_analysis_{int(time.time())}.png")
            plt.savefig(output_file, dpi=100)
            plt.close()
            logger.info(f"Visualization saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
