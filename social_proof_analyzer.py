#!/usr/bin/env python3
"""
Social Proof Analyzer for Viral Detection
Analyzes social signals like subscriber activity, moderator reactions, and community engagement
"""

import time
import re
import logging
from collections import Counter, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("BeastClipper")


class SocialProofAnalyzer:
    """Analyzes social proof signals for viral moment detection."""
    
    def __init__(self):
        """Initialize social proof analyzer."""
        # User classification patterns
        self.vip_indicators = [
            'subscriber', 'sub', 'mod', 'moderator', 'vip', 'founder', 'staff'
        ]
        
        # High-value emotes that indicate strong reactions
        self.high_value_emotes = [
            'pogchamp', 'pog', 'poggers', 'omegalul', 'lul', 'kappa', 'pepehands',
            'monkas', 'sadge', 'copium', 'aware', 'ez', 'gg', 'hype', 'fire'
        ]
        
        # Message quality indicators
        self.quality_indicators = {
            'clip_request': ['clip', 'clipper', 'someone clip', 'clip that', 'clip it'],
            'strong_reaction': ['holy', 'insane', 'crazy', 'wtf', 'omg', 'no way', 'yooo'],
            'agreement': ['facts', 'true', 'real', 'fr', 'no cap', 'based'],
            'excitement': ['hype', 'lets go', 'poggers', 'sheesh', 'fire', 'bussin']
        }
        
        # User activity tracking
        self.user_activity = {}  # username -> activity data
        self.message_history = deque(maxlen=200)  # Last 200 messages
        self.activity_windows = deque(maxlen=60)  # Last 60 seconds of activity
        
    def analyze_social_signals(self, chat_messages: List[Dict], timestamp: float) -> Dict:
        """
        Analyze social proof from chat messages.
        
        Args:
            chat_messages: List of chat message dictionaries
            timestamp: Timestamp of the analysis
            
        Returns:
            Dictionary containing social proof metrics
        """
        if not chat_messages:
            return self._empty_social_proof()
        
        try:
            # Update message history
            for msg in chat_messages:
                self.message_history.append(msg)
                self._update_user_activity(msg)
            
            # Calculate social proof metrics
            metrics = {
                'timestamp': timestamp,
                'total_messages': len(chat_messages),
                'unique_chatters': self._count_unique_chatters(chat_messages),
                'message_velocity': self._calculate_message_velocity(chat_messages, timestamp),
                'vip_engagement': self._analyze_vip_engagement(chat_messages),
                'emote_usage': self._analyze_emote_usage(chat_messages),
                'message_quality': self._analyze_message_quality(chat_messages),
                'user_diversity': self._calculate_user_diversity(chat_messages),
                'reaction_intensity': self._calculate_reaction_intensity(chat_messages),
                'social_momentum': self._calculate_social_momentum(chat_messages, timestamp),
                'total_score': 0.0
            }
            
            # Calculate overall social proof score
            metrics['total_score'] = self._calculate_total_score(metrics)
            
            # Store activity window
            self.activity_windows.append({
                'timestamp': timestamp,
                'metrics': metrics
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Social proof analysis error: {e}")
            return self._empty_social_proof()
    
    def _empty_social_proof(self) -> Dict:
        """Return empty social proof metrics."""
        return {
            'timestamp': time.time(),
            'total_messages': 0,
            'unique_chatters': 0,
            'message_velocity': 0.0,
            'vip_engagement': {'count': 0, 'score': 0.0},
            'emote_usage': {},
            'message_quality': {'total_score': 0.0, 'categories': {}},
            'user_diversity': 0.0,
            'reaction_intensity': 0.0,
            'social_momentum': 0.0,
            'total_score': 0.0
        }
    
    def _count_unique_chatters(self, messages: List[Dict]) -> int:
        """Count unique chatters in message list."""
        usernames = set()
        for msg in messages:
            username = msg.get('username', '').lower()
            if username:
                usernames.add(username)
        return len(usernames)
    
    def _calculate_message_velocity(self, messages: List[Dict], current_time: float) -> float:
        """Calculate messages per second over recent time window."""
        if not messages:
            return 0.0
        
        # Count messages in last 10 seconds
        recent_messages = [
            msg for msg in messages
            if current_time - msg.get('timestamp', 0) <= 10
        ]
        
        return len(recent_messages) / 10.0
    
    def _analyze_vip_engagement(self, messages: List[Dict]) -> Dict:
        """Analyze engagement from VIP users (subscribers, mods, etc.)."""
        vip_messages = []
        
        for msg in messages:
            username = msg.get('username', '').lower()
            message_text = msg.get('text', '').lower()
            
            # Check if user appears to be VIP based on username or message content
            is_vip = any(indicator in username for indicator in self.vip_indicators)
            
            # Additional VIP detection (could be enhanced with actual Twitch badges)
            if not is_vip:
                # Check for mod actions or subscriber language
                is_vip = any(word in message_text for word in ['@', 'timeout', 'resub', 'gifted'])
            
            if is_vip:
                vip_messages.append(msg)
        
        vip_count = len(vip_messages)
        total_messages = len(messages)
        
        # VIP engagement score (higher weight for VIP participation)
        if total_messages > 0:
            vip_ratio = vip_count / total_messages
            engagement_score = min(vip_ratio * 3.0, 1.0)  # Max score of 1.0
        else:
            engagement_score = 0.0
        
        return {
            'count': vip_count,
            'ratio': vip_ratio if total_messages > 0 else 0.0,
            'score': engagement_score
        }
    
    def _analyze_emote_usage(self, messages: List[Dict]) -> Dict:
        """Analyze emote usage patterns."""
        emote_counts = Counter()
        total_emotes = 0
        
        for msg in messages:
            text = msg.get('text', '').lower()
            
            # Count high-value emotes
            for emote in self.high_value_emotes:
                count = text.count(emote)
                if count > 0:
                    emote_counts[emote] += count
                    total_emotes += count
            
            # Also count generic emote patterns (e.g., :emote: format)
            emote_pattern_matches = re.findall(r':\w+:', text)
            for match in emote_pattern_matches:
                emote_name = match.strip(':').lower()
                if emote_name not in self.high_value_emotes:  # Don't double count
                    emote_counts[emote_name] += 1
                    total_emotes += 1
        
        return {
            'total_emotes': total_emotes,
            'unique_emotes': len(emote_counts),
            'top_emotes': dict(emote_counts.most_common(5)),
            'emote_density': total_emotes / max(len(messages), 1)
        }
    
    def _analyze_message_quality(self, messages: List[Dict]) -> Dict:
        """Analyze message quality and reaction types."""
        quality_scores = {category: 0 for category in self.quality_indicators.keys()}
        total_quality_score = 0.0
        
        for msg in messages:
            text = msg.get('text', '').lower()
            
            # Check each quality category
            for category, indicators in self.quality_indicators.items():
                for indicator in indicators:
                    if indicator in text:
                        quality_scores[category] += 1
                        
                        # Weight different categories
                        if category == 'clip_request':
                            total_quality_score += 2.0  # High value
                        elif category == 'strong_reaction':
                            total_quality_score += 1.5
                        else:
                            total_quality_score += 1.0
                        break  # Only count once per message per category
        
        return {
            'total_score': total_quality_score,
            'categories': quality_scores,
            'avg_quality_per_message': total_quality_score / max(len(messages), 1)
        }
    
    def _calculate_user_diversity(self, messages: List[Dict]) -> float:
        """Calculate user diversity (how many different users are participating)."""
        if not messages:
            return 0.0
        
        unique_users = self._count_unique_chatters(messages)
        total_messages = len(messages)
        
        # Higher diversity score when more users participate
        # vs. one user spamming many messages
        if total_messages <= unique_users:
            return 1.0  # Perfect diversity
        
        diversity_ratio = unique_users / total_messages
        
        # Apply curve to favor higher diversity
        return min(diversity_ratio * 2.0, 1.0)
    
    def _calculate_reaction_intensity(self, messages: List[Dict]) -> float:
        """Calculate overall reaction intensity based on multiple factors."""
        if not messages:
            return 0.0
        
        intensity_score = 0.0
        
        for msg in messages:
            text = msg.get('text', '')
            
            # All caps indicates intensity
            if text.isupper() and len(text) > 3:
                intensity_score += 0.5
            
            # Multiple exclamation/question marks
            intensity_score += min(text.count('!') * 0.1, 0.5)
            intensity_score += min(text.count('?') * 0.1, 0.3)
            
            # Repeated characters (e.g., "yooooo", "nooo")
            repeated_chars = re.findall(r'([a-zA-Z])\1{2,}', text.lower())
            intensity_score += min(len(repeated_chars) * 0.2, 0.4)
            
            # Length suggests effort/engagement
            if len(text) > 20:
                intensity_score += 0.1
        
        # Normalize by message count
        avg_intensity = intensity_score / len(messages)
        
        return min(avg_intensity, 1.0)
    
    def _calculate_social_momentum(self, messages: List[Dict], current_time: float) -> float:
        """Calculate social momentum based on activity trend."""
        if not self.activity_windows or len(self.activity_windows) < 3:
            return 0.5  # Neutral momentum when insufficient data
        
        # Get recent activity scores
        recent_scores = []
        for window in list(self.activity_windows)[-5:]:  # Last 5 windows
            window_score = (
                window['metrics']['unique_chatters'] * 0.3 +
                window['metrics']['message_velocity'] * 0.4 +
                window['metrics']['reaction_intensity'] * 0.3
            )
            recent_scores.append(window_score)
        
        if len(recent_scores) < 2:
            return 0.5
        
        # Calculate trend (increasing = positive momentum)
        early_avg = sum(recent_scores[:len(recent_scores)//2]) / (len(recent_scores)//2)
        late_avg = sum(recent_scores[len(recent_scores)//2:]) / (len(recent_scores) - len(recent_scores)//2)
        
        if early_avg == 0:
            return 0.8 if late_avg > 0 else 0.5
        
        momentum = (late_avg - early_avg) / early_avg
        
        # Normalize to 0-1 range
        normalized_momentum = max(0, min((momentum + 1) / 2, 1))
        
        return normalized_momentum
    
    def _calculate_total_score(self, metrics: Dict) -> float:
        """Calculate overall social proof score."""
        # Weighted combination of all metrics
        score = 0.0
        
        # Unique chatters (20% weight)
        chatter_score = min(metrics['unique_chatters'] / 10.0, 1.0)
        score += chatter_score * 0.20
        
        # Message velocity (25% weight)
        velocity_score = min(metrics['message_velocity'] / 5.0, 1.0)
        score += velocity_score * 0.25
        
        # VIP engagement (15% weight)
        score += metrics['vip_engagement']['score'] * 0.15
        
        # Message quality (20% weight)
        quality_score = min(metrics['message_quality']['total_score'] / 10.0, 1.0)
        score += quality_score * 0.20
        
        # User diversity (10% weight)
        score += metrics['user_diversity'] * 0.10
        
        # Reaction intensity (10% weight)
        score += metrics['reaction_intensity'] * 0.10
        
        return min(score, 1.0)
    
    def _update_user_activity(self, message: Dict):
        """Update user activity tracking."""
        username = message.get('username', '')
        if not username:
            return
        
        current_time = time.time()
        
        if username not in self.user_activity:
            self.user_activity[username] = {
                'first_seen': current_time,
                'last_message': current_time,
                'message_count': 0,
                'total_chars': 0,
                'emote_usage': Counter()
            }
        
        user_data = self.user_activity[username]
        user_data['last_message'] = current_time
        user_data['message_count'] += 1
        user_data['total_chars'] += len(message.get('text', ''))
        
        # Track emote usage
        text = message.get('text', '').lower()
        for emote in self.high_value_emotes:
            if emote in text:
                user_data['emote_usage'][emote] += 1
    
    def get_user_engagement_score(self, username: str) -> float:
        """Get engagement score for a specific user."""
        if username not in self.user_activity:
            return 0.0
        
        user_data = self.user_activity[username]
        current_time = time.time()
        
        # Factors for engagement score
        recency = max(0, 1 - (current_time - user_data['last_message']) / 300)  # 5 min decay
        activity = min(user_data['message_count'] / 10.0, 1.0)  # Max at 10 messages
        effort = min(user_data['total_chars'] / 500.0, 1.0)  # Max at 500 chars
        emote_usage = min(sum(user_data['emote_usage'].values()) / 5.0, 1.0)  # Max at 5 emotes
        
        engagement_score = (recency * 0.3 + activity * 0.3 + effort * 0.2 + emote_usage * 0.2)
        
        return engagement_score
    
    def get_trending_topics(self, time_window: int = 300) -> List[str]:
        """Get trending topics/keywords from recent chat."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Get recent messages
        recent_messages = [
            msg for msg in self.message_history
            if msg.get('timestamp', 0) >= cutoff_time
        ]
        
        # Extract keywords (simple approach)
        word_counts = Counter()
        
        for msg in recent_messages:
            text = msg.get('text', '').lower()
            words = re.findall(r'\b\w{3,}\b', text)  # Words 3+ characters
            
            for word in words:
                # Skip common words
                if word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'end', 'let', 'put', 'say', 'she', 'too', 'use']:
                    word_counts[word] += 1
        
        # Return top trending words
        return [word for word, count in word_counts.most_common(10) if count >= 2]
    
    def reset_activity_tracking(self):
        """Reset activity tracking (useful for new streams)."""
        self.user_activity.clear()
        self.message_history.clear()
        self.activity_windows.clear()
        logger.info("Social proof analyzer reset")
