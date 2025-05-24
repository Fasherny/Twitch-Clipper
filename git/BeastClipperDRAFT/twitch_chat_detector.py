#!/usr/bin/env python3
"""
Universal Twitch Chat Viral Detector
Detects viral moments using modern Twitch chat patterns across all streamers
"""

import time
import logging
import re
from collections import deque, Counter
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger("BeastClipper")


class TwitchChatViralDetector(QThread):
    """Detects viral moments from Twitch chat using modern slang and reaction patterns."""
    
    # Signals
    viral_moment_detected = pyqtSignal(dict)
    chat_activity_update = pyqtSignal(dict)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, stream_url, sensitivity=0.7):
        """
        Initialize Twitch chat viral detector.
        
        Args:
            stream_url: Twitch stream URL
            sensitivity: Detection sensitivity (0.1 = very strict, 0.9 = very loose)
        """
        super().__init__()
        
        self.stream_url = stream_url
        self.sensitivity = sensitivity
        self.running = False
        self.driver = None
        
        # Chat monitoring
        self.message_history = deque(maxlen=100)  # Last 100 messages
        self.message_timestamps = deque(maxlen=100)  # When messages occurred
        self.baseline_messages_per_second = 0
        self.baseline_established = False
        
        # Viral indicators - MODERN Twitch slang
        self.viral_keywords = {
            # Direct clip requests
            'clip_requests': ['clip that', 'clip it', 'clipper', 'someone clip', 'clip this'],
            
            # Modern reactions
            'hype_words': ['w', 'big w', 'massive w', 'huge w', 'yooo', 'sheesh', 'fire', 'bussin'],
            
            # Shock/surprise
            'shock_words': ['bruh', 'nah', 'ayo', 'pause', 'sus', 'foul', 'violation', 'caught'],
            
            # Laughter
            'laugh_words': ['lmao', 'lmfao', 'dead', 'crying', 'skull', 'im done', 'not the'],
            
            # Disbelief
            'disbelief': ['no way', 'no shot', 'chat is this real', 'bro what', 'what the', 'how'],
            
            # Agreement/hype
            'agreement': ['fr', 'no cap', 'facts', 'real', 'true', 'based', 'valid'],
            
            # Loss reactions  
            'loss_words': ['l', 'big l', 'massive l', 'down bad', 'rip', 'cooked', 'finished']
        }
        
        # Emote patterns (modern emotes)
        self.viral_emotes = [
            'kappa', 'lul', 'omegalul', 'pepehands', 'monkas', 'sadge', 'copium',
            'ez', 'gg', 'oof', 'pog', 'hype', 'modcheck', 'aware', 'surely'
        ]
        
        # Text patterns
        self.excitement_patterns = [
            r'^[A-Z\s!?]{8,}$',  # ALL CAPS MESSAGES
            r'[!]{3,}',          # Multiple exclamation marks
            r'[?]{3,}',          # Multiple question marks  
            r'[A-Za-z]\1{3,}',   # Repeated letters (yooooo, brooooo)
            r'😭{2,}',           # Crying emoji spam
            r'🔥{2,}',           # Fire emoji spam
            r'💀{2,}',           # Skull emoji spam
        ]
        
        # Detection thresholds based on sensitivity
        self._calculate_thresholds()
    
    def _calculate_thresholds(self):
        """Calculate detection thresholds based on sensitivity."""
        # Lower sensitivity = higher thresholds (more strict)
        base_multiplier = 2.0 + (1.0 - self.sensitivity) * 3.0
        
        self.thresholds = {
            'message_spike_multiplier': base_multiplier,     # How much above baseline
            'viral_keyword_count': max(2, int(5 * self.sensitivity)),  # Min viral words needed
            'excitement_pattern_count': max(1, int(3 * self.sensitivity)),  # Min excitement patterns
            'sustained_activity_seconds': max(3, int(8 * (1.0 - self.sensitivity))),  # How long activity must last
            'min_unique_users': max(2, int(4 * self.sensitivity))  # Min different users reacting
        }
        
        logger.info(f"Chat detection thresholds: {self.thresholds}")
    
    def run(self):
        """Main detection loop - runs every 1-2 seconds."""
        self.running = True
        
        try:
            # Setup browser
            self.status_update.emit("Setting up chat monitor...")
            self._setup_browser()
            
            # Navigate to stream
            self.status_update.emit("Connecting to Twitch chat...")
            self.driver.get(self.stream_url)
            time.sleep(5)  # Let page load
            
            # Dismiss any overlays
            self._dismiss_overlays()
            
            # Build baseline for 30 seconds
            self.status_update.emit("Learning chat baseline...")
            self._build_baseline()
            
            # Main detection loop
            self.status_update.emit("Chat viral detection active!")
            last_check = time.time()
            
            while self.running:
                current_time = time.time()
                
                # Check every 1.5 seconds for responsiveness
                if current_time - last_check >= 1.5:
                    try:
                        # Get new messages
                        new_messages = self._get_new_messages()
                        
                        if new_messages:
                            # Add to history
                            for msg in new_messages:
                                self.message_history.append(msg)
                                self.message_timestamps.append(current_time)
                            
                            # Analyze for viral patterns
                            viral_score = self._analyze_viral_patterns(new_messages)
                            
                            # Check if this is a viral moment
                            if self._is_viral_moment(viral_score, new_messages):
                                self._emit_viral_moment(viral_score, new_messages)
                            
                            # Update activity stats
                            self._update_activity_stats()
                        
                        last_check = current_time
                        
                    except Exception as e:
                        logger.error(f"Chat analysis error: {e}")
                        time.sleep(2)
                
                time.sleep(0.5)  # Small delay to prevent excessive CPU usage
                
        except Exception as e:
            logger.error(f"Chat detector error: {str(e)}")
            self.error_occurred.emit(f"Chat Detection Error: {str(e)}")
        
        finally:
            self._cleanup()
    
    def _setup_browser(self):
        """Setup Chrome browser for chat monitoring."""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        self.driver = webdriver.Chrome(options=options)
        
        # Make it harder to detect
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    def _dismiss_overlays(self):
        """Dismiss Twitch overlays that might block chat access."""
        overlay_selectors = [
            '[data-a-target="player-overlay-mature-accept"]',
            '[data-a-target="content-classification-gate-overlay-start-watching-button"]',
            '.consent-banner button',
            '.mature-content-overlay button',
            '[data-test-selector="mature-gate-button"]'
        ]
        
        for selector in overlay_selectors:
            try:
                element = WebDriverWait(self.driver, 2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                element.click()
                time.sleep(1)
            except:
                continue
    
    def _get_new_messages(self):
        """Get new chat messages from Twitch."""
        try:
            # Twitch chat selectors
            chat_selectors = [
                ".chat-line__message",
                "[data-a-target='chat-line-message']",
                ".chat-line__message-body"
            ]
            
            messages = []
            
            for selector in chat_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in elements[-20:]:  # Only check last 20 messages
                        try:
                            # Get username and message text
                            username_elem = element.find_element(By.CSS_SELECTOR, ".chat-author__display-name")
                            message_elem = element.find_element(By.CSS_SELECTOR, "[data-a-target='chat-line-message-body']")
                            
                            username = username_elem.text.strip()
                            message_text = message_elem.text.strip().lower()
                            
                            if username and message_text:
                                # Create message object
                                message = {
                                    'username': username,
                                    'text': message_text,
                                    'timestamp': time.time(),
                                    'element_id': id(element)  # Unique identifier
                                }
                                
                                # Only add if we haven't seen this exact message recently
                                if not self._is_duplicate_message(message):
                                    messages.append(message)
                        
                        except:
                            continue
                    
                    if messages:
                        break  # Found messages with this selector
                        
                except:
                    continue
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    def _is_duplicate_message(self, message):
        """Check if we've already processed this message."""
        # Check last 10 messages for duplicates
        for prev_msg in list(self.message_history)[-10:]:
            if (prev_msg['username'] == message['username'] and 
                prev_msg['text'] == message['text'] and
                abs(prev_msg['timestamp'] - message['timestamp']) < 5):
                return True
        return False
    
    def _build_baseline(self):
        """Build baseline chat activity over 30 seconds."""
        baseline_start = time.time()
        baseline_messages = []
        
        while time.time() - baseline_start < 30:
            if not self.running:
                return
                
            messages = self._get_new_messages()
            baseline_messages.extend(messages)
            
            for msg in messages:
                self.message_history.append(msg)
                self.message_timestamps.append(time.time())
            
            time.sleep(2)
        
        # Calculate baseline messages per second
        if baseline_messages:
            self.baseline_messages_per_second = len(baseline_messages) / 30.0
        else:
            self.baseline_messages_per_second = 0.5  # Default low baseline
        
        self.baseline_established = True
        logger.info(f"Chat baseline established: {self.baseline_messages_per_second:.2f} msg/sec")
    
    def _analyze_viral_patterns(self, messages):
        """Analyze messages for viral indicators and return a score."""
        if not messages:
            return 0.0
        
        score = 0.0
        viral_indicators = {
            'clip_requests': 0,
            'hype_words': 0,
            'shock_words': 0,
            'laugh_words': 0,
            'disbelief': 0,
            'agreement': 0,
            'loss_words': 0,
            'excitement_patterns': 0,
            'viral_emotes': 0,
            'unique_users': set()
        }
        
        for message in messages:
            text = message['text']
            username = message['username']
            viral_indicators['unique_users'].add(username)
            
            # Check viral keywords
            for category, keywords in self.viral_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        viral_indicators[category] += 1
            
            # Check viral emotes
            for emote in self.viral_emotes:
                if emote in text:
                    viral_indicators['viral_emotes'] += 1
            
            # Check excitement patterns
            for pattern in self.excitement_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    viral_indicators['excitement_patterns'] += 1
        
        # Calculate score based on indicators
        # Clip requests are strongest indicator
        score += viral_indicators['clip_requests'] * 0.4
        
        # Hype and shock words
        score += (viral_indicators['hype_words'] + viral_indicators['shock_words']) * 0.2
        
        # Laughter and disbelief
        score += (viral_indicators['laugh_words'] + viral_indicators['disbelief']) * 0.15
        
        # Excitement patterns
        score += viral_indicators['excitement_patterns'] * 0.15
        
        # User diversity bonus
        unique_user_count = len(viral_indicators['unique_users'])
        if unique_user_count >= self.thresholds['min_unique_users']:
            score += 0.1
        
        # Normalize score
        score = min(score, 1.0)
        
        return score
    
    def _is_viral_moment(self, viral_score, messages):
        """Determine if current activity constitutes a viral moment."""
        # Must have minimum viral score
        if viral_score < 0.3:
            return False
        
        # Check message velocity spike
        current_time = time.time()
        recent_messages = [
            msg for msg in self.message_history 
            if current_time - msg['timestamp'] <= 5  # Last 5 seconds
        ]
        
        current_rate = len(recent_messages) / 5.0  # Messages per second
        
        # Must exceed baseline by threshold multiplier
        if self.baseline_established:
            required_rate = self.baseline_messages_per_second * self.thresholds['message_spike_multiplier']
            if current_rate < required_rate:
                return False
        
        # Check for sustained activity
        sustained_activity = self._check_sustained_activity()
        if not sustained_activity:
            return False
        
        # Passed all checks!
        return True
    
    def _check_sustained_activity(self):
        """Check if there's been sustained elevated activity."""
        current_time = time.time()
        lookback_seconds = self.thresholds['sustained_activity_seconds']
        
        # Count messages in each second over the lookback period
        activity_by_second = {}
        
        for msg in self.message_history:
            msg_time = msg['timestamp']
            if current_time - msg_time <= lookback_seconds:
                second = int(msg_time)
                activity_by_second[second] = activity_by_second.get(second, 0) + 1
        
        # Check if at least half the seconds had elevated activity
        elevated_seconds = 0
        required_messages_per_second = self.baseline_messages_per_second * 1.5
        
        for second_count in activity_by_second.values():
            if second_count >= required_messages_per_second:
                elevated_seconds += 1
        
        required_elevated_seconds = max(1, lookback_seconds // 2)
        return elevated_seconds >= required_elevated_seconds
    
    def _emit_viral_moment(self, viral_score, messages):
        """Emit viral moment detection signal."""
        moment_info = {
            'timestamp': 5,  # Assume moment happened ~5 seconds ago
            'score': viral_score,
            'detected_at': time.time(),
            'time_ago': 5,
            'description': self._generate_description(messages),
            'confidence': self._calculate_confidence(viral_score),
            'message_count': len(messages),
            'unique_users': len(set(msg['username'] for msg in messages))
        }
        
        self.viral_moment_detected.emit(moment_info)
        logger.info(f"CHAT VIRAL MOMENT: Score {viral_score:.2f} - {moment_info['description']}")
    
    def _generate_description(self, messages):
        """Generate description of what made this moment viral."""
        # Count different types of reactions
        reactions = Counter()
        
        for message in messages:
            text = message['text']
            
            # Categorize the message
            if any(word in text for word in self.viral_keywords['clip_requests']):
                reactions['clip_requests'] += 1
            elif any(word in text for word in self.viral_keywords['hype_words']):
                reactions['hype'] += 1
            elif any(word in text for word in self.viral_keywords['shock_words']):
                reactions['shock'] += 1
            elif any(word in text for word in self.viral_keywords['laugh_words']):
                reactions['laughter'] += 1
            elif any(word in text for word in self.viral_keywords['disbelief']):
                reactions['disbelief'] += 1
        
        # Generate description based on top reactions
        if reactions['clip_requests'] >= 2:
            return "Chat demanding clips!"
        elif reactions['laughter'] >= 3:
            return "Chat erupting in laughter"
        elif reactions['shock'] >= 3:
            return "Chat in shock/disbelief"
        elif reactions['hype'] >= 3:
            return "Chat hyped up"
        else:
            return "High chat activity spike"
    
    def _calculate_confidence(self, viral_score):
        """Calculate confidence level."""
        if viral_score >= 0.8:
            return "Very High"
        elif viral_score >= 0.6:
            return "High"
        elif viral_score >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _update_activity_stats(self):
        """Update and emit current activity statistics."""
        current_time = time.time()
        
        # Recent message count (last 10 seconds)
        recent_count = len([
            msg for msg in self.message_history 
            if current_time - msg['timestamp'] <= 10
        ])
        
        # Current rate vs baseline
        current_rate = recent_count / 10.0
        spike_multiplier = current_rate / (self.baseline_messages_per_second + 0.01)
        
        activity_stats = {
            'recent_messages': recent_count,
            'current_rate': current_rate,
            'baseline_rate': self.baseline_messages_per_second,
            'spike_multiplier': spike_multiplier
        }
        
        self.chat_activity_update.emit(activity_stats)
    
    def update_sensitivity(self, new_sensitivity):
        """Update detection sensitivity."""
        self.sensitivity = new_sensitivity
        self._calculate_thresholds()
        logger.info(f"Chat sensitivity updated to {new_sensitivity}")
    
    def stop(self):
        """Stop the chat detector."""
        self.running = False
    
    def _cleanup(self):
        """Clean up browser resources."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None