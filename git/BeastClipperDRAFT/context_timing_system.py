#!/usr/bin/env python3
"""
Context Detection & Perfect Timing System
Determines stream context and optimizes clip timing for maximum viral potential
"""

import time
import logging
import re
import numpy as np
from collections import deque, Counter
from datetime import datetime

logger = logging.getLogger("BeastClipper")


class ContextDetector:
    """Detects the current context/category of stream content."""
    
    def __init__(self):
        self.context_history = deque(maxlen=100)
        self.confidence_threshold = 0.7
        
        # Context indicators
        self.gaming_indicators = {
            'chat_keywords': [
                'gg', 'ez', 'rip', 'respawn', 'kill', 'death', 'win', 'lose', 'clutch',
                'aim', 'bot', 'hacker', 'lag', 'fps', 'ping', 'server', 'match',
                'round', 'score', 'rank', 'elo', 'mmr', 'tier', 'division'
            ],
            'excitement_patterns': [
                'ace', 'clutch', 'insane play', 'sick', 'nasty', 'clean',
                'what a play', 'cracked', 'goated', 'diff'
            ],
            'game_specific': {
                'fps': ['headshot', 'frag', 'spawn', 'camp', 'rush', 'rotate'],
                'moba': ['gank', 'ward', 'baron', 'dragon', 'tower', 'minion'],
                'battle_royale': ['zone', 'storm', 'circle', 'loot', 'third party'],
                'minecraft': ['creeper', 'diamond', 'netherite', 'enchant', 'villager']
            }
        }
        
        self.chatting_indicators = {
            'chat_keywords': [
                'story', 'tell us', 'what happened', 'remember when', 'one time',
                'funny thing', 'listen', 'speaking of', 'remind me', 'anyways'
            ],
            'reaction_words': [
                'really', 'seriously', 'no way', 'what', 'damn', 'crazy',
                'wild', 'facts', 'true', 'cap', 'lying'
            ],
            'personal_topics': [
                'family', 'girlfriend', 'boyfriend', 'work', 'school', 'college',
                'money', 'car', 'house', 'vacation', 'food', 'movie', 'music'
            ]
        }
        
        self.react_indicators = {
            'chat_keywords': [
                'pause', 'rewind', 'skip', 'cringe', 'react', 'watch',
                'video', 'clip', 'tiktok', 'youtube', 'twitter'
            ],
            'sync_reactions': [
                'same time', 'together', 'both', 'synchronized', 'at once'
            ],
            'content_types': [
                'meme', 'compilation', 'highlight', 'trailer', 'music video',
                'news', 'drama', 'expose', 'roast'
            ]
        }
        
        self.irl_indicators = {
            'chat_keywords': [
                'outside', 'walking', 'driving', 'restaurant', 'store',
                'people', 'stranger', 'public', 'street', 'building'
            ],
            'location_words': [
                'here', 'there', 'over there', 'behind', 'front', 'left', 'right',
                'upstairs', 'downstairs', 'inside', 'outside'
            ]
        }
    
    def analyze_context_from_chat(self, recent_messages):
        """Analyze recent chat messages to determine context."""
        if not recent_messages:
            return 'unknown', 0.0
        
        # Combine all message text
        combined_text = ' '.join([msg.get('text', '') for msg in recent_messages]).lower()
        
        # Count indicators for each context
        context_scores = {
            'gaming': self._calculate_gaming_score(combined_text, recent_messages),
            'chatting': self._calculate_chatting_score(combined_text, recent_messages),
            'react': self._calculate_react_score(combined_text, recent_messages),
            'irl': self._calculate_irl_score(combined_text, recent_messages)
        }
        
        # Find highest scoring context
        best_context = max(context_scores.items(), key=lambda x: x[1])
        
        # Add to history
        self.context_history.append({
            'timestamp': time.time(),
            'context': best_context[0],
            'confidence': best_context[1],
            'scores': context_scores
        })
        
        return best_context[0], best_context[1]
    
    def _calculate_gaming_score(self, text, messages):
        """Calculate gaming context score."""
        score = 0.0
        
        # Basic gaming keywords
        keyword_count = sum(1 for keyword in self.gaming_indicators['chat_keywords'] if keyword in text)
        score += min(keyword_count * 0.1, 0.4)
        
        # Excitement patterns
        excitement_count = sum(1 for pattern in self.gaming_indicators['excitement_patterns'] if pattern in text)
        score += min(excitement_count * 0.15, 0.3)
        
        # Game-specific terms
        for game_type, keywords in self.gaming_indicators['game_specific'].items():
            game_count = sum(1 for keyword in keywords if keyword in text)
            if game_count >= 2:
                score += 0.2
                break
        
        # Quick succession messages (typical in gaming)
        if len(messages) >= 5:
            timestamps = [msg.get('timestamp', 0) for msg in messages[-5:]]
            if timestamps and max(timestamps) - min(timestamps) < 10:  # 5 messages in 10 seconds
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_chatting_score(self, text, messages):
        """Calculate just chatting context score."""
        score = 0.0
        
        # Conversation keywords
        keyword_count = sum(1 for keyword in self.chatting_indicators['chat_keywords'] if keyword in text)
        score += min(keyword_count * 0.15, 0.4)
        
        # Reaction words
        reaction_count = sum(1 for word in self.chatting_indicators['reaction_words'] if word in text)
        score += min(reaction_count * 0.1, 0.3)
        
        # Personal topics
        personal_count = sum(1 for topic in self.chatting_indicators['personal_topics'] if topic in text)
        score += min(personal_count * 0.1, 0.2)
        
        # Question patterns (typical in conversations)
        question_count = text.count('?')
        score += min(question_count * 0.05, 0.1)
        
        return min(score, 1.0)
    
    def _calculate_react_score(self, text, messages):
        """Calculate react content context score."""
        score = 0.0
        
        # React keywords
        keyword_count = sum(1 for keyword in self.react_indicators['chat_keywords'] if keyword in text)
        score += min(keyword_count * 0.2, 0.5)
        
        # Synchronized reactions
        sync_count = sum(1 for pattern in self.react_indicators['sync_reactions'] if pattern in text)
        score += min(sync_count * 0.2, 0.3)
        
        # Content type mentions
        content_count = sum(1 for content in self.react_indicators['content_types'] if content in text)
        score += min(content_count * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def _calculate_irl_score(self, text, messages):
        """Calculate IRL context score."""
        score = 0.0
        
        # IRL keywords
        keyword_count = sum(1 for keyword in self.irl_indicators['chat_keywords'] if keyword in text)
        score += min(keyword_count * 0.2, 0.4)
        
        # Location words
        location_count = sum(1 for word in self.irl_indicators['location_words'] if word in text)
        score += min(location_count * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def get_current_context(self):
        """Get current context with confidence."""
        if not self.context_history:
            return 'unknown', 0.0
        
        # Get recent contexts (last 2 minutes)
        recent_time = time.time() - 120
        recent_contexts = [
            ctx for ctx in self.context_history 
            if ctx['timestamp'] >= recent_time
        ]
        
        if not recent_contexts:
            return self.context_history[-1]['context'], self.context_history[-1]['confidence']
        
        # Calculate weighted average of recent contexts
        context_weights = Counter()
        total_weight = 0
        
        for ctx in recent_contexts:
            weight = ctx['confidence'] * (1 + (ctx['timestamp'] - recent_time) / 120)  # More recent = higher weight
            context_weights[ctx['context']] += weight
            total_weight += weight
        
        if total_weight == 0:
            return 'unknown', 0.0
        
        # Get most likely context
        best_context = context_weights.most_common(1)[0]
        confidence = best_context[1] / total_weight
        
        return best_context[0], confidence


class PerfectTimingOptimizer:
    """Optimizes clip timing for maximum viral potential."""
    
    def __init__(self):
        self.signal_history = deque(maxlen=300)  # 5 minutes at 1-second intervals
        self.context_timing_rules = {
            'gaming': {
                'pre_setup_seconds': 4,    # Show setup before the play
                'post_reaction_seconds': 3, # Show immediate reaction
                'min_length': 8,
                'max_length': 25,
                'peak_detection_window': 3
            },
            'chatting': {
                'pre_setup_seconds': 6,    # More context needed
                'post_reaction_seconds': 4, # Show chat reaction
                'min_length': 10,
                'max_length': 30,
                'peak_detection_window': 5
            },
            'react': {
                'pre_setup_seconds': 3,    # Show what they're reacting to
                'post_reaction_seconds': 5, # Show full reaction
                'min_length': 12,
                'max_length': 35,
                'peak_detection_window': 4
            },
            'irl': {
                'pre_setup_seconds': 5,    # Show environment context
                'post_reaction_seconds': 4, # Show what happens after
                'min_length': 15,
                'max_length': 40,
                'peak_detection_window': 6
            }
        }
    
    def add_signal_data(self, timestamp, chat_score, video_score, social_proof):
        """Add signal data point for timing analysis."""
        self.signal_history.append({
            'timestamp': timestamp,
            'chat_score': chat_score,
            'video_score': video_score,
            'social_proof': social_proof,
            'combined_score': (chat_score * 0.6 + video_score * 0.3 + social_proof * 0.1)
        })
    
    def find_optimal_clip_timing(self, peak_timestamp, context, base_duration=20):
        """Find optimal start and end times for a clip around a peak moment."""
        rules = self.context_timing_rules.get(context, self.context_timing_rules['chatting'])
        
        # Find the actual peak within the detection window
        true_peak = self._find_true_peak(peak_timestamp, rules['peak_detection_window'])
        
        # Find natural break points
        start_boundary = self._find_start_boundary(true_peak, rules['pre_setup_seconds'])
        end_boundary = self._find_end_boundary(true_peak, rules['post_reaction_seconds'])
        
        # Calculate initial timing
        clip_start = true_peak - start_boundary
        clip_end = true_peak + end_boundary
        
        # Adjust for content flow
        clip_start, clip_end = self._adjust_for_content_flow(clip_start, clip_end, context)
        
        # Ensure length constraints
        clip_length = clip_end - clip_start
        if clip_length < rules['min_length']:
            # Extend clip to minimum length
            extension = (rules['min_length'] - clip_length) / 2
            clip_start -= extension
            clip_end += extension
        elif clip_length > rules['max_length']:
            # Trim to maximum length, keeping peak centered
            trim = (clip_length - rules['max_length']) / 2
            clip_start += trim
            clip_end -= trim
        
        # Final validation
        clip_start = max(0, clip_start)
        
        return {
            'start': clip_start,
            'end': clip_end,
            'peak': true_peak,
            'duration': clip_end - clip_start,
            'pre_context': true_peak - clip_start,
            'post_context': clip_end - true_peak,
            'confidence': self._calculate_timing_confidence(clip_start, clip_end, true_peak)
        }
    
    def _find_true_peak(self, estimated_peak, window_seconds):
        """Find the actual peak moment within a time window."""
        # Get signals within the window
        window_start = estimated_peak - window_seconds
        window_end = estimated_peak + window_seconds
        
        window_signals = [
            signal for signal in self.signal_history
            if window_start <= signal['timestamp'] <= window_end
        ]
        
        if not window_signals:
            return estimated_peak
        
        # Find highest combined score
        best_signal = max(window_signals, key=lambda x: x['combined_score'])
        return best_signal['timestamp']
    
    def _find_start_boundary(self, peak_timestamp, max_pre_seconds):
        """Find optimal start point by analyzing signal buildup."""
        # Look for the beginning of activity buildup
        search_start = peak_timestamp - max_pre_seconds - 5  # Extra buffer
        search_signals = [
            signal for signal in self.signal_history
            if search_start <= signal['timestamp'] <= peak_timestamp
        ]
        
        if len(search_signals) < 3:
            return max_pre_seconds  # Default fallback
        
        # Find where activity started increasing
        baseline_score = np.mean([s['combined_score'] for s in search_signals[:3]])
        
        for i, signal in enumerate(search_signals[3:], 3):
            if signal['combined_score'] > baseline_score * 1.3:  # 30% increase
                buildup_start = peak_timestamp - signal['timestamp']
                return min(buildup_start + 2, max_pre_seconds)  # Add 2s buffer
        
        return max_pre_seconds
    
    def _find_end_boundary(self, peak_timestamp, max_post_seconds):
        """Find optimal end point by analyzing reaction decay."""
        # Look for when excitement dies down
        search_end = peak_timestamp + max_post_seconds + 5  # Extra buffer
        search_signals = [
            signal for signal in self.signal_history
            if peak_timestamp <= signal['timestamp'] <= search_end
        ]
        
        if len(search_signals) < 3:
            return max_post_seconds  # Default fallback
        
        # Find peak score in the immediate aftermath
        peak_score = max([s['combined_score'] for s in search_signals[:3]])
        
        # Find where it drops significantly
        for i, signal in enumerate(search_signals[2:], 2):
            if signal['combined_score'] < peak_score * 0.4:  # Dropped to 40% of peak
                decay_end = signal['timestamp'] - peak_timestamp
                return min(decay_end + 1, max_post_seconds)  # Add 1s buffer
        
        return max_post_seconds
    
    def _adjust_for_content_flow(self, start_time, end_time, context):
        """Adjust timing based on content flow patterns."""
        # Gaming: Avoid cutting mid-play
        if context == 'gaming':
            # Extend slightly if we might be cutting off a multi-kill or clutch
            duration = end_time - start_time
            if duration < 15:  # Short clips might miss follow-up action
                end_time += 3
        
        # Chatting: Ensure we don't cut off mid-sentence
        elif context == 'chatting':
            # Look for natural speech pauses
            start_time = self._find_speech_pause(start_time, direction='backward')
            end_time = self._find_speech_pause(end_time, direction='forward')
        
        # React: Sync with content being reacted to
        elif context == 'react':
            # Ensure we catch both the stimulus and reaction
            start_time -= 1  # Show a bit more of what they're reacting to
        
        return start_time, end_time
    
    def _find_speech_pause(self, timestamp, direction='forward'):
        """Find natural speech pause near timestamp."""
        # This would analyze audio for natural breaks
        # For now, return slight adjustment
        if direction == 'forward':
            return timestamp + 0.5
        else:
            return timestamp - 0.5
    
    def _calculate_timing_confidence(self, start_time, end_time, peak_time):
        """Calculate confidence in the timing selection."""
        # Check signal distribution around the clip
        clip_signals = [
            signal for signal in self.signal_history
            if start_time <= signal['timestamp'] <= end_time
        ]
        
        if not clip_signals:
            return 0.5
        
        # Peak should be well-centered and significant
        peak_position = (peak_time - start_time) / (end_time - start_time)
        position_score = 1.0 - abs(peak_position - 0.5)  # Closer to center = better
        
        # Peak should be significantly higher than surrounding activity
        peak_signal = next((s for s in clip_signals if s['timestamp'] == peak_time), None)
        if not peak_signal:
            return 0.5
        
        avg_score = np.mean([s['combined_score'] for s in clip_signals])
        peak_prominence = peak_signal['combined_score'] / (avg_score + 0.1)
        prominence_score = min(peak_prominence / 2.0, 1.0)
        
        # Combine scores
        confidence = (position_score * 0.4 + prominence_score * 0.6)
        return min(confidence, 1.0)
    
    def analyze_moment_buildup(self, peak_timestamp, context):
        """Analyze how the viral moment built up for better understanding."""
        rules = self.context_timing_rules.get(context, self.context_timing_rules['chatting'])
        
        # Look at 30 seconds before peak
        analysis_start = peak_timestamp - 30
        analysis_signals = [
            signal for signal in self.signal_history
            if analysis_start <= signal['timestamp'] <= peak_timestamp
        ]
        
        if not analysis_signals:
            return {'buildup_type': 'unknown', 'buildup_duration': 0, 'intensity_curve': []}
        
        # Analyze buildup pattern
        scores = [s['combined_score'] for s in analysis_signals]
        timestamps = [s['timestamp'] for s in analysis_signals]
        
        # Find where buildup started
        baseline = np.mean(scores[:5]) if len(scores) >= 5 else scores[0]
        buildup_start_idx = 0
        
        for i, score in enumerate(scores):
            if score > baseline * 1.5:  # 50% increase marks buildup start
                buildup_start_idx = i
                break
        
        buildup_duration = peak_timestamp - timestamps[buildup_start_idx] if buildup_start_idx < len(timestamps) else 0
        
        # Classify buildup type
        buildup_type = 'gradual'
        if buildup_duration < 5:
            buildup_type = 'instant'
        elif buildup_duration < 15:
            buildup_type = 'quick'
        elif buildup_duration > 25:
            buildup_type = 'slow_burn'
        
        # Calculate intensity curve
        intensity_curve = []
        if len(scores) > 1:
            max_score = max(scores)
            intensity_curve = [score / max_score for score in scores[buildup_start_idx:]]
        
        return {
            'buildup_type': buildup_type,
            'buildup_duration': buildup_duration,
            'intensity_curve': intensity_curve,
            'peak_prominence': max(scores) / (baseline + 0.1),
            'consistency': np.std(scores) / (np.mean(scores) + 0.1)  # Lower = more consistent buildup
        }


class MomentumTracker:
    """Tracks momentum and energy flow in streams."""
    
    def __init__(self):
        self.momentum_history = deque(maxlen=600)  # 10 minutes
        self.energy_levels = deque(maxlen=60)      # 1 minute of energy samples
        
    def update_momentum(self, chat_activity, video_activity, social_proof):
        """Update momentum based on current activity levels."""
        current_time = time.time()
        
        # Calculate combined energy
        energy = (chat_activity * 0.5 + video_activity * 0.3 + social_proof * 0.2)
        
        # Store energy level
        self.energy_levels.append({
            'timestamp': current_time,
            'energy': energy,
            'chat': chat_activity,
            'video': video_activity,
            'social': social_proof
        })
        
        # Calculate momentum (rate of change)
        if len(self.energy_levels) >= 5:
            recent_energies = [e['energy'] for e in list(self.energy_levels)[-5:]]
            momentum = np.diff(recent_energies).mean()  # Average rate of change
        else:
            momentum = 0.0
        
        # Store momentum
        self.momentum_history.append({
            'timestamp': current_time,
            'momentum': momentum,
            'energy': energy
        })
        
        return momentum
    
    def get_momentum_score(self):
        """Get current momentum score (0-1)."""
        if not self.momentum_history:
            return 0.0
        
        recent_momentum = [m['momentum'] for m in list(self.momentum_history)[-10:]]
        avg_momentum = np.mean(recent_momentum)
        
        # Normalize momentum to 0-1 scale
        return max(0, min(avg_momentum + 0.5, 1.0))
    
    def is_building_momentum(self):
        """Check if momentum is currently building."""
        if len(self.momentum_history) < 10:
            return False
        
        recent_momentum = [m['momentum'] for m in list(self.momentum_history)[-10:]]
        
        # Check if momentum is consistently positive
        positive_count = sum(1 for m in recent_momentum if m > 0)
        return positive_count >= 7  # 70% of recent momentum is positive
    
    def predict_peak_timing(self):
        """Predict when current momentum might peak."""
        if not self.is_building_momentum():
            return None
        
        recent_energy = [e['energy'] for e in list(self.energy_levels)[-30:]]  # Last 30 seconds
        
        if len(recent_energy) < 10:
            return None
        
        # Simple linear regression to predict peak
        x = np.arange(len(recent_energy))
        y = np.array(recent_energy)
        
        if len(x) > 1 and np.std(x) > 0:
            slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
            
            if slope > 0:  # Still building
                # Predict when it might level off (very rough estimate)
                current_rate = slope
                time_to_peak = max(5, min(20, 10 / (current_rate + 0.1)))  # 5-20 seconds
                return time.time() + time_to_peak
        
        return None