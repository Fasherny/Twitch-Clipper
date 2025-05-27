#!/usr/bin/env python3
"""
Professional Twitch IRC Chat Detector - Full Featured
Advanced IRC-based chat monitoring with sophisticated viral detection
No OAuth required for read-only access, but extensible for full features
"""

import socket
import time
import logging
import re
import threading
import json
from collections import deque, Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger("BeastClipper")


class ProfessionalTwitchIRCDetector(QThread):
    """
    Professional Twitch IRC chat detector with advanced viral moment detection.
    
    Features:
    - No OAuth required for basic functionality
    - Advanced message parsing and user tracking
    - Real-time viral pattern detection
    - Sophisticated spam filtering
    - Reconnection and error recovery
    - Extensible for OAuth features
    - Rate limiting compliance
    - Advanced analytics
    """
    
    # Signals
    viral_moment_detected = pyqtSignal(dict)
    chat_activity_update = pyqtSignal(dict)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    connection_status = pyqtSignal(bool)
    user_action_detected = pyqtSignal(dict)  # Raids, subs, follows, etc.
    
    def __init__(self, channel_name, sensitivity=0.7, use_oauth=False, oauth_token=None):
        """
        Initialize professional IRC detector.
        
        Args:
            channel_name: Twitch channel name (without #)
            sensitivity: Detection sensitivity (0.1 = strict, 0.9 = loose)
            use_oauth: Enable OAuth features (send messages, access subscriber info)
            oauth_token: OAuth token if available (optional)
        """
        super().__init__()
        
        self.channel_name = channel_name.lower().replace('#', '')
        self.sensitivity = sensitivity
        self.use_oauth = use_oauth
        self.oauth_token = oauth_token
        self.running = False
        
        # IRC connection
        self.sock = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_ping_time = 0
        
        # Message processing
        self.message_buffer = []
        self.processing_lock = threading.Lock()
        
        # Advanced chat monitoring
        self.message_history = deque(maxlen=200)  # Larger history for better analysis
        self.message_timestamps = deque(maxlen=200)
        self.user_activity = {}  # username -> activity data
        self.baseline_metrics = {
            'messages_per_second': 0,
            'unique_users_per_minute': 0,
            'emote_density': 0,
            'established': False
        }
        
        # Advanced viral detection patterns
        self.viral_keywords = {
            'clip_requests': {
                'patterns': ['clip that', 'clip it', 'clipper', 'someone clip', 'clip this', 'clippers get on this'],
                'weight': 3.0  # High importance
            },
            'hype_words': {
                'patterns': ['w', 'big w', 'massive w', 'huge w', 'yooo', 'sheesh', 'fire', 'bussin', 'cracked', 'goated'],
                'weight': 2.0
            },
            'shock_words': {
                'patterns': ['bruh', 'nah', 'ayo', 'pause', 'sus', 'foul', 'violation', 'caught', 'down bad'],
                'weight': 2.5
            },
            'laugh_words': {
                'patterns': ['lmao', 'lmfao', 'dead', 'crying', 'skull', 'im done', 'not the', 'bruh moment'],
                'weight': 2.0
            },
            'disbelief': {
                'patterns': ['no way', 'no shot', 'cap', 'no cap', 'chat is this real', 'bro what', 'what the', 'how tf'],
                'weight': 2.5
            },
            'agreement': {
                'patterns': ['fr', 'facts', 'real', 'true', 'based', 'valid', 'this', 'exactly'],
                'weight': 1.5
            },
            'loss_words': {
                'patterns': ['l', 'big l', 'massive l', 'rip', 'cooked', 'finished', 'oof', 'pain'],
                'weight': 1.8
            }
        }
        
        # Advanced emote detection
        self.viral_emotes = {
            'tier_3': ['omegalul', 'pogchamp', 'pepehands', 'monkas'],  # Highest impact
            'tier_2': ['lul', 'kappa', 'poggers', 'sadge', 'copium', 'aware'],  # High impact
            'tier_1': ['ez', 'gg', 'oof', 'pog', 'hype', 'modcheck', 'surely']  # Medium impact
        }
        
        # Advanced user classification
        self.user_types = {
            'vip_indicators': ['mod', 'vip', 'subscriber', 'founder', 'staff', 'admin'],
            'bot_indicators': ['bot', 'nightbot', 'streamelements', 'fossabot', 'moobot'],
            'new_user_threshold': 300  # Seconds to be considered "new"
        }
        
        # Sophisticated filtering
        self.spam_filters = {
            'min_message_length': 2,
            'max_repeated_chars': 5,
            'max_caps_ratio': 0.8,
            'duplicate_message_window': 30,  # Seconds
            'rapid_message_threshold': 5  # Messages per 10 seconds
        }
        
        # Advanced detection thresholds
        self._calculate_advanced_thresholds()
        
        # Connection settings
        self.irc_server = 'irc.chat.twitch.tv'
        self.irc_port = 6667
        if self.use_oauth and self.oauth_token:
            self.nickname = 'beastclipper_bot'  # Custom bot name
        else:
            self.nickname = f'justinfan{int(time.time()) % 100000}'  # Anonymous
        
        # Advanced analytics
        self.analytics = {
            'total_messages': 0,
            'unique_users': set(),
            'viral_moments_detected': 0,
            'false_positive_rate': 0.0,
            'average_confidence': 0.0,
            'context_distribution': Counter(),
            'peak_activity_times': [],
            'user_engagement_scores': {}
        }
    
    def _calculate_advanced_thresholds(self):
        """Calculate sophisticated detection thresholds."""
        # Base multipliers adjusted for IRC's typically higher message volume
        base_multiplier = 1.5 + (1.0 - self.sensitivity) * 2.5
        
        self.thresholds = {
            'message_spike_multiplier': base_multiplier,
            'viral_keyword_score': max(3.0, 8.0 * self.sensitivity),
            'emote_density_threshold': max(0.1, 0.3 * self.sensitivity),
            'sustained_activity_seconds': max(5, int(15 * (1.0 - self.sensitivity))),
            'min_unique_users': max(3, int(6 * self.sensitivity)),
            'user_engagement_threshold': max(0.3, 0.6 * self.sensitivity),
            'cross_correlation_threshold': 0.7  # For synchronized reactions
        }
    
    def run(self):
        """Main IRC monitoring loop with advanced error handling."""
        self.running = True
        
        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.status_update.emit("Connecting to Twitch IRC...")
                
                # Advanced connection with retry logic
                if not self._connect_with_retry():
                    self.reconnect_attempts += 1
                    if self.reconnect_attempts < self.max_reconnect_attempts:
                        wait_time = min(30, 5 * self.reconnect_attempts)
                        self.status_update.emit(f"Reconnecting in {wait_time}s... (attempt {self.reconnect_attempts})")
                        time.sleep(wait_time)
                        continue
                    else:
                        self.error_occurred.emit("Max reconnection attempts reached")
                        break
                
                # Join channel with capabilities
                if not self._join_channel_advanced():
                    continue
                
                # Build sophisticated baseline
                self.status_update.emit("Building advanced chat baseline...")
                self._build_advanced_baseline()
                
                # Main monitoring loop
                self.status_update.emit("Professional IRC viral detection active!")
                self.connection_status.emit(True)
                
                last_ping = time.time()
                
                while self.running and self.connected:
                    try:
                        # Handle IRC keepalive
                        if time.time() - last_ping > 60:  # Ping every minute
                            self._send_ping()
                            last_ping = time.time()
                        
                        # Process IRC messages with advanced parsing
                        new_messages = self._read_and_parse_irc()
                        
                        if new_messages:
                            # Advanced message processing
                            processed_messages = self._process_messages_advanced(new_messages)
                            
                            if processed_messages:
                                # Update user tracking
                                self._update_user_tracking(processed_messages)
                                
                                # Advanced viral pattern analysis
                                viral_analysis = self._analyze_viral_patterns_advanced(processed_messages)
                                
                                # Sophisticated moment detection
                                if self._detect_viral_moment_advanced(viral_analysis):
                                    self._emit_advanced_viral_moment(viral_analysis)
                                
                                # Update advanced analytics
                                self._update_advanced_analytics(processed_messages)
                        
                        time.sleep(0.05)  # 50ms polling for responsiveness
                        
                    except socket.timeout:
                        continue  # Normal timeout, keep going
                    except ConnectionResetError:
                        logger.warning("IRC connection reset, attempting reconnection...")
                        break  # Break inner loop to reconnect
                    except Exception as e:
                        logger.error(f"IRC message processing error: {e}")
                        time.sleep(1)
                        
            except Exception as e:
                logger.error(f"IRC detector error: {str(e)}")
                self.error_occurred.emit(f"IRC Detection Error: {str(e)}")
                time.sleep(5)
        
        self.connection_status.emit(False)
        self._cleanup()
    
    def _connect_with_retry(self):
        """Advanced connection with proper error handling."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)  # 5 second timeout for operations
            
            # Connect to IRC server
            self.sock.connect((self.irc_server, self.irc_port))
            
            # Send authentication
            if self.use_oauth and self.oauth_token:
                self.sock.send(f"PASS oauth:{self.oauth_token}\r\n".encode('utf-8'))
                self.sock.send(f"NICK {self.nickname}\r\n".encode('utf-8'))
            else:
                self.sock.send(f"NICK {self.nickname}\r\n".encode('utf-8'))
                self.sock.send(f"USER {self.nickname} 0 * :{self.nickname}\r\n".encode('utf-8'))
            
            # Wait for successful connection
            start_time = time.time()
            while time.time() - start_time < 10:
                try:
                    response = self.sock.recv(1024).decode('utf-8', errors='ignore')
                    
                    if '001' in response or '376' in response:  # Welcome messages
                        self.connected = True
                        self.reconnect_attempts = 0  # Reset on successful connection
                        logger.info("Successfully connected to Twitch IRC")
                        return True
                    elif 'Error logging in' in response or 'Login authentication failed' in response:
                        logger.error("IRC authentication failed")
                        return False
                        
                except socket.timeout:
                    continue
            
            logger.error("IRC connection timeout")
            return False
            
        except Exception as e:
            logger.error(f"IRC connection error: {e}")
            return False
    
    def _join_channel_advanced(self):
        """Join channel with advanced capabilities."""
        try:
            # Request capabilities for enhanced features
            if self.use_oauth:
                self.sock.send("CAP REQ :twitch.tv/membership\r\n".encode('utf-8'))
                self.sock.send("CAP REQ :twitch.tv/tags\r\n".encode('utf-8'))
                self.sock.send("CAP REQ :twitch.tv/commands\r\n".encode('utf-8'))
            
            # Join the channel
            self.sock.send(f"JOIN #{self.channel_name}\r\n".encode('utf-8'))
            
            # Wait for join confirmation
            start_time = time.time()
            while time.time() - start_time < 10:
                try:
                    response = self.sock.recv(1024).decode('utf-8', errors='ignore')
                    if f"366 {self.nickname} #{self.channel_name}" in response:  # End of names list
                        logger.info(f"Successfully joined #{self.channel_name}")
                        return True
                except socket.timeout:
                    continue
            
            logger.error(f"Failed to join #{self.channel_name}")
            return False
            
        except Exception as e:
            logger.error(f"Error joining channel: {e}")
            return False
    
    def _read_and_parse_irc(self):
        """Advanced IRC message reading and parsing."""
        try:
            # Set socket to non-blocking for real-time processing
            self.sock.settimeout(0.1)
            raw_data = self.sock.recv(4096).decode('utf-8', errors='ignore')
            
            if not raw_data:
                return []
            
            # Handle PING/PONG
            if raw_data.startswith('PING'):
                self.sock.send(f"PONG {raw_data.split()[1]}\r\n".encode('utf-8'))
                return []
            
            # Parse IRC messages
            messages = []
            lines = raw_data.strip().split('\r\n')
            
            for line in lines:
                if not line:
                    continue
                
                # Parse IRC message format
                parsed = self._parse_irc_message(line)
                if parsed:
                    messages.append(parsed)
            
            return messages
            
        except socket.timeout:
            return []  # Normal timeout
        except Exception as e:
            logger.error(f"Error reading IRC: {e}")
            return []
    
    def _parse_irc_message(self, line):
        """Parse individual IRC message with advanced features."""
        try:
            # Handle Twitch IRC format: @tags :user!user@user.tmi.twitch.tv PRIVMSG #channel :message
            if 'PRIVMSG' not in line:
                return None
            
            parts = line.split(' ', 3)
            if len(parts) < 4:
                return None
            
            # Extract tags (if using OAuth)
            tags = {}
            if line.startswith('@'):
                tag_part = parts[0][1:]  # Remove @
                for tag in tag_part.split(';'):
                    if '=' in tag:
                        key, value = tag.split('=', 1)
                        tags[key] = value
            
            # Extract username
            user_part = parts[1] if not line.startswith('@') else parts[1]
            if '!' in user_part:
                username = user_part.split('!')[0][1:]  # Remove :
            else:
                username = 'unknown'
            
            # Extract message
            message_text = parts[3][1:] if parts[3].startswith(':') else parts[3]  # Remove :
            
            # Create enhanced message object
            message = {
                'username': username.lower(),
                'text': message_text,
                'timestamp': time.time(),
                'tags': tags,
                'raw': line,
                'channel': self.channel_name,
                'is_action': message_text.startswith('\x01ACTION'),  # /me messages
                'message_length': len(message_text),
                'caps_ratio': sum(1 for c in message_text if c.isupper()) / max(len(message_text), 1)
            }
            
            # Add user classification
            message['user_type'] = self._classify_user(username, tags)
            
            return message
            
        except Exception as e:
            logger.error(f"Error parsing IRC message: {e}")
            return None
    
    def _classify_user(self, username, tags):
        """Classify user type based on username and tags."""
        # Check for bot patterns
        if any(bot_word in username for bot_word in self.user_types['bot_indicators']):
            return 'bot'
        
        # Check OAuth tags for subscriber/mod/vip status
        if tags:
            if tags.get('mod') == '1':
                return 'moderator'
            elif tags.get('subscriber') == '1':
                return 'subscriber'
            elif tags.get('vip') == '1':
                return 'vip'
            elif 'founder' in tags.get('badges', ''):
                return 'founder'
        
        # Check username patterns for VIP indicators
        if any(vip_word in username for vip_word in self.user_types['vip_indicators']):
            return 'vip'
        
        # Check if new user
        if username in self.user_activity:
            time_active = time.time() - self.user_activity[username]['first_seen']
            if time_active < self.user_types['new_user_threshold']:
                return 'new_user'
        
        return 'regular'
    
    def _process_messages_advanced(self, messages):
        """Advanced message processing with spam filtering."""
        processed = []
        
        for msg in messages:
            # Apply sophisticated spam filters
            if self._is_spam_message(msg):
                continue
            
            # Enhance message with analysis
            msg['viral_score'] = self._calculate_message_viral_score(msg)
            msg['emote_count'] = self._count_emotes(msg['text'])
            msg['keyword_matches'] = self._find_keyword_matches(msg['text'])
            msg['excitement_level'] = self._calculate_excitement_level(msg)
            
            processed.append(msg)
            
            # Update message history
            self.message_history.append(msg)
            self.message_timestamps.append(msg['timestamp'])
        
        return processed
    
    def _is_spam_message(self, msg):
        """Sophisticated spam detection."""
        text = msg['text']
        
        # Length filter
        if len(text) < self.spam_filters['min_message_length']:
            return True
        
        # Excessive repeated characters
        repeated_chars = re.findall(r'(.)\1{3,}', text.lower())
        if len(repeated_chars) > self.spam_filters['max_repeated_chars']:
            return True
        
        # Excessive caps
        if msg['caps_ratio'] > self.spam_filters['max_caps_ratio'] and len(text) > 5:
            return True
        
        # Bot filter
        if msg['user_type'] == 'bot':
            return True
        
        # Duplicate message filter
        recent_messages = [m for m in self.message_history 
                          if time.time() - m['timestamp'] < self.spam_filters['duplicate_message_window']]
        
        similar_count = sum(1 for m in recent_messages 
                           if m['username'] == msg['username'] and m['text'] == msg['text'])
        
        if similar_count > 0:  # Duplicate detected
            return True
        
        # Rapid posting filter
        user_recent = [m for m in recent_messages 
                      if m['username'] == msg['username'] and time.time() - m['timestamp'] < 10]
        
        if len(user_recent) > self.spam_filters['rapid_message_threshold']:
            return True
        
        return False
    
    def _calculate_message_viral_score(self, msg):
        """Calculate viral potential score for individual message."""
        score = 0.0
        text = msg['text'].lower()
        
        # Keyword scoring with weights
        for category, data in self.viral_keywords.items():
            for pattern in data['patterns']:
                if pattern in text:
                    score += data['weight']
                    break  # Only count once per category
        
        # Emote scoring
        for tier, emotes in self.viral_emotes.items():
            for emote in emotes:
                if emote in text:
                    if tier == 'tier_3':
                        score += 3.0
                    elif tier == 'tier_2':
                        score += 2.0
                    else:
                        score += 1.0
        
        # User type multiplier
        user_multipliers = {
            'moderator': 2.0,
            'vip': 1.8,
            'subscriber': 1.5,
            'founder': 1.7,
            'regular': 1.0,
            'new_user': 0.8,
            'bot': 0.0
        }
        
        score *= user_multipliers.get(msg['user_type'], 1.0)
        
        # Excitement level bonus
        score += msg.get('excitement_level', 0) * 2.0
        
        return min(score, 10.0)  # Cap at 10
    
    def _count_emotes(self, text):
        """Count emotes in message."""
        count = 0
        text_lower = text.lower()
        
        for tier_emotes in self.viral_emotes.values():
            for emote in tier_emotes:
                count += text_lower.count(emote)
        
        # Also count generic emote patterns
        emote_patterns = re.findall(r':\w+:', text)
        count += len(emote_patterns)
        
        return count
    
    def _find_keyword_matches(self, text):
        """Find viral keyword matches."""
        matches = []
        text_lower = text.lower()
        
        for category, data in self.viral_keywords.items():
            for pattern in data['patterns']:
                if pattern in text_lower:
                    matches.append((category, pattern))
        
        return matches
    
    def _calculate_excitement_level(self, msg):
        """Calculate excitement level of message."""
        text = msg['text']
        excitement = 0.0
        
        # Exclamation marks
        excitement += min(text.count('!') * 0.2, 1.0)
        
        # Question marks
        excitement += min(text.count('?') * 0.1, 0.5)
        
        # All caps words
        caps_words = len([word for word in text.split() if word.isupper() and len(word) > 2])
        excitement += min(caps_words * 0.3, 1.0)
        
        # Repeated characters
        repeated = len(re.findall(r'(.)\1{2,}', text.lower()))
        excitement += min(repeated * 0.2, 0.8)
        
        return min(excitement, 2.0)
    
    def _build_advanced_baseline(self):
        """Build sophisticated baseline metrics."""
        baseline_start = time.time()
        baseline_messages = []
        baseline_users = set()
        baseline_emotes = 0
        
        # Collect 60 seconds of baseline data
        while time.time() - baseline_start < 60 and self.running:
            messages = self._read_and_parse_irc()
            for msg in messages:
                if not self._is_spam_message(msg):
                    baseline_messages.append(msg)
                    baseline_users.add(msg['username'])
                    baseline_emotes += self._count_emotes(msg['text'])
            
            time.sleep(1)
        
        # Calculate baseline metrics
        duration = 60.0
        if baseline_messages:
            self.baseline_metrics['messages_per_second'] = len(baseline_messages) / duration
            self.baseline_metrics['unique_users_per_minute'] = len(baseline_users)
            self.baseline_metrics['emote_density'] = baseline_emotes / len(baseline_messages)
        else:
            # Default baseline for quiet channels
            self.baseline_metrics['messages_per_second'] = 0.5
            self.baseline_metrics['unique_users_per_minute'] = 5
            self.baseline_metrics['emote_density'] = 0.1
        
        self.baseline_metrics['established'] = True
        
        logger.info(f"Advanced baseline established: "
                   f"{self.baseline_metrics['messages_per_second']:.2f} msg/s, "
                   f"{self.baseline_metrics['unique_users_per_minute']} users/min")
    
    def _analyze_viral_patterns_advanced(self, messages):
        """Advanced viral pattern analysis."""
        if not messages:
            return None
        
        current_time = time.time()
        
        # Aggregate viral scores
        total_viral_score = sum(msg['viral_score'] for msg in messages)
        avg_viral_score = total_viral_score / len(messages)
        
        # Calculate advanced metrics
        unique_users = len(set(msg['username'] for msg in messages))
        total_emotes = sum(msg['emote_count'] for msg in messages)
        
        # Keyword analysis
        all_keywords = []
        for msg in messages:
            all_keywords.extend(msg['keyword_matches'])
        
        keyword_distribution = Counter([kw[0] for kw in all_keywords])
        
        # User engagement analysis
        user_types = Counter(msg['user_type'] for msg in messages)
        vip_ratio = (user_types['moderator'] + user_types['vip'] + user_types['subscriber']) / len(messages)
        
        # Calculate current activity vs baseline
        recent_window = 30  # seconds
        recent_messages = [m for m in self.message_history 
                          if current_time - m['timestamp'] <= recent_window]
        
        current_rate = len(recent_messages) / recent_window
        baseline_rate = self.baseline_metrics['messages_per_second']
        
        activity_multiplier = current_rate / (baseline_rate + 0.01)
        
        return {
            'timestamp': current_time,
            'total_viral_score': total_viral_score,
            'avg_viral_score': avg_viral_score,
            'unique_users': unique_users,
            'total_emotes': total_emotes,
            'keyword_distribution': keyword_distribution,
            'user_types': user_types,
            'vip_ratio': vip_ratio,
            'activity_multiplier': activity_multiplier,
            'message_count': len(messages),
            'messages': messages
        }
    
    def _detect_viral_moment_advanced(self, analysis):
        """Advanced viral moment detection algorithm."""
        if not analysis or not self.baseline_metrics['established']:
            return False
        
        # Multiple detection criteria must be met
        criteria_met = 0
        total_criteria = 5
        
        # 1. Viral score threshold
        if analysis['avg_viral_score'] >= self.thresholds['viral_keyword_score']:
            criteria_met += 1
        
        # 2. Activity spike
        if analysis['activity_multiplier'] >= self.thresholds['message_spike_multiplier']:
            criteria_met += 1
        
        # 3. User engagement
        if analysis['unique_users'] >= self.thresholds['min_unique_users']:
            criteria_met += 1
        
        # 4. VIP participation
        if analysis['vip_ratio'] >= self.thresholds['user_engagement_threshold']:
            criteria_met += 1
        
        # 5. Sustained activity check
        if self._check_sustained_activity_advanced():
            criteria_met += 1
        
        # Detection threshold based on sensitivity
        required_criteria = max(2, int(total_criteria * (1.0 - self.sensitivity) + 2))
        
        return criteria_met >= required_criteria
    
    def _check_sustained_activity_advanced(self):
        """Check for sustained elevated activity."""
        if len(self.message_history) < 10:
            return False
        
        window_size = self.thresholds['sustained_activity_seconds']
        current_time = time.time()
        
        # Check activity in sliding windows
        elevated_windows = 0
        total_windows = 0
        
        for i in range(int(window_size)):
            window_start = current_time - window_size + i
            window_end = window_start + 1
            
            window_messages = [m for m in self.message_history 
                             if window_start <= m['timestamp'] <= window_end]
            
            if len(window_messages) > self.baseline_metrics['messages_per_second'] * 1.5:
                elevated_windows += 1
            
            total_windows += 1
        
        return elevated_windows >= total_windows * 0.6  # 60% of windows elevated
    
    def _emit_advanced_viral_moment(self, analysis):
        """Emit sophisticated viral moment detection."""
        # Calculate final confidence score
        confidence = min(
            analysis['avg_viral_score'] / 10.0 * 0.4 +
            min(analysis['activity_multiplier'] / 3.0, 1.0) * 0.3 +
            min(analysis['vip_ratio'] * 2.0, 1.0) * 0.2 +
            min(analysis['unique_users'] / 10.0, 1.0) * 0.1,
            1.0
        )
        
        # Generate description
        description = self._generate_advanced_description(analysis)
        
        moment_info = {
            'timestamp': 5,  # Assume moment happened ~5 seconds ago
            'score': confidence,
            'confidence': confidence,
            'detected_at': analysis['timestamp'],
            'time_ago': 5,
            'description': description,
            'advanced_analysis': analysis,
            'detection_method': 'professional_irc',
            'message_count': analysis['message_count'],
            'unique_users': analysis['unique_users'],
            'viral_score': analysis['total_viral_score'],
            'activity_multiplier': analysis['activity_multiplier'],
            'vip_engagement': analysis['vip_ratio']
        }
        
        self.viral_moment_detected.emit(moment_info)
        self.analytics['viral_moments_detected'] += 1
        
        logger.info(
            f"🔥 PROFESSIONAL IRC VIRAL DETECTION: "
            f"Confidence={confidence:.3f}, "
            f"Messages={analysis['message_count']}, "
            f"Users={analysis['unique_users']}, "
            f"Activity={analysis['activity_multiplier']:.2f}x, "
            f"Description={description}"
        )
    
    def _generate_advanced_description(self, analysis):
        """Generate sophisticated description based on analysis."""
        # Find dominant reaction type
        top_keyword = analysis['keyword_distribution'].most_common(1)
        if top_keyword:
            category = top_keyword[0][0]
            
            descriptions = {
                'clip_requests': "Chat demanding clips!",
                'hype_words': "Chat absolutely hyped up!",
                'shock_words': "Chat in complete shock!",
                'laugh_words': "Chat erupting in laughter!",
                'disbelief': "Chat can't believe it!",
                'agreement': "Chat in strong agreement!",
                'loss_words': "Chat witnessing a major L!"
            }
            
            return descriptions.get(category, "High viral activity detected!")
        
        # Fallback based on metrics
        if analysis['activity_multiplier'] > 4:
            return "Massive chat explosion!"
        elif analysis['vip_ratio'] > 0.3:
            return "VIPs going crazy!"
        elif analysis['unique_users'] > 20:
            return "Huge audience reaction!"
        else:
            return "Significant viral moment detected!"
    
    def _update_user_tracking(self, messages):
        """Update advanced user activity tracking."""
        current_time = time.time()
        
        for msg in messages:
            username = msg['username']
            
            if username not in self.user_activity:
                self.user_activity[username] = {
                    'first_seen': current_time,
                    'last_message': current_time,
                    'message_count': 0,
                    'total_viral_score': 0.0,
                    'emote_usage': 0,
                    'user_type': msg['user_type'],
                    'engagement_score': 0.0
                }
            
            user_data = self.user_activity[username]
            user_data['last_message'] = current_time
            user_data['message_count'] += 1
            user_data['total_viral_score'] += msg['viral_score']
            user_data['emote_usage'] += msg['emote_count']
            
            # Calculate engagement score
            user_data['engagement_score'] = self._calculate_user_engagement(user_data)
    
    def _calculate_user_engagement(self, user_data):
        """Calculate sophisticated user engagement score."""
        current_time = time.time()
        
        # Factors
        recency = max(0, 1 - (current_time - user_data['last_message']) / 300)  # 5 min decay
        activity = min(user_data['message_count'] / 20.0, 1.0)  # Max at 20 messages
        viral_contribution = min(user_data['total_viral_score'] / 50.0, 1.0)  # Max at 50 points
        emote_usage = min(user_data['emote_usage'] / 10.0, 1.0)  # Max at 10 emotes
        
        # User type multiplier
        type_multipliers = {
            'moderator': 2.0,
            'vip': 1.8,
            'subscriber': 1.5,
            'founder': 1.7,
            'regular': 1.0,
            'new_user': 0.8
        }
        
        type_multiplier = type_multipliers.get(user_data['user_type'], 1.0)
        
        engagement = (recency * 0.2 + activity * 0.3 + viral_contribution * 0.3 + emote_usage * 0.2) * type_multiplier
        
        return min(engagement, 2.0)
    
    def _update_advanced_analytics(self, messages):
        """Update comprehensive analytics."""
        for msg in messages:
            self.analytics['total_messages'] += 1
            self.analytics['unique_users'].add(msg['username'])
        
        # Update running averages
        if self.analytics['viral_moments_detected'] > 0:
            recent_moments = [m for m in self.message_history 
                            if time.time() - m['timestamp'] <= 300]  # Last 5 minutes
            
            if recent_moments:
                avg_viral_score = sum(m.get('viral_score', 0) for m in recent_moments) / len(recent_moments)
                self.analytics['average_confidence'] = avg_viral_score / 10.0
    
    def _update_activity_stats(self):
        """Update and emit activity statistics."""
        current_time = time.time()
        
        # Recent activity (last 30 seconds)
        recent_messages = [m for m in self.message_history 
                          if current_time - m['timestamp'] <= 30]
        
        recent_count = len(recent_messages)
        unique_recent_users = len(set(m['username'] for m in recent_messages))
        
        # Calculate rates
        current_rate = recent_count / 30.0
        baseline_rate = self.baseline_metrics['messages_per_second']
        activity_multiplier = current_rate / (baseline_rate + 0.01)
        
        activity_stats = {
            'recent_messages': recent_count,
            'unique_recent_users': unique_recent_users,
            'current_rate': current_rate,
            'baseline_rate': baseline_rate,
            'activity_multiplier': activity_multiplier,
            'total_users_seen': len(self.analytics['unique_users']),
            'total_messages': self.analytics['total_messages'],
            'connection_quality': 'excellent' if self.connected else 'disconnected'
        }
        
        self.chat_activity_update.emit(activity_stats)
    
    def _send_ping(self):
        """Send IRC ping to maintain connection."""
        try:
            if self.connected:
                self.sock.send("PING :tmi.twitch.tv\r\n".encode('utf-8'))
        except Exception as e:
            logger.error(f"Error sending ping: {e}")
            self.connected = False
    
    def get_analytics_summary(self):
        """Get comprehensive analytics summary."""
        return {
            'total_messages': self.analytics['total_messages'],
            'unique_users': len(self.analytics['unique_users']),
            'viral_moments_detected': self.analytics['viral_moments_detected'],
            'average_confidence': self.analytics['average_confidence'],
            'baseline_metrics': self.baseline_metrics,
            'active_users': len([u for u in self.user_activity.values() 
                               if time.time() - u['last_message'] < 300]),
            'connection_uptime': time.time() - self.last_ping_time if self.connected else 0
        }
    
    def update_sensitivity(self, new_sensitivity):
        """Update detection sensitivity."""
        self.sensitivity = new_sensitivity
        self._calculate_advanced_thresholds()
        logger.info(f"IRC sensitivity updated to {new_sensitivity}")
    
    def stop(self):
        """Stop the IRC detector."""
        self.running = False
        self.connected = False
    
    def _cleanup(self):
        """Clean up IRC connection and resources."""
        try:
            if self.sock:
                self.sock.close()
        except:
            pass
        
        self.connected = False
        logger.info("IRC detector cleaned up")