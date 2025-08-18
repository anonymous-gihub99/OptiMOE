# models/conversation_analyzer.py
"""
Conversation Analysis Module
Analyzes conversation context, tracks topics, and maintains conversation state
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import Counter, deque
from datetime import datetime, timedelta
import numpy as np
from loguru import logger


class ConversationAnalyzer:
    """
    Advanced conversation analysis for contextual understanding
    Tracks topics, sentiment flow, and conversation dynamics
    """
    
    def __init__(self, 
                 max_history: int = 10,
                 topic_threshold: float = 0.3):
        """
        Initialize conversation analyzer
        
        Args:
            max_history: Maximum conversation turns to maintain
            topic_threshold: Minimum score for topic relevance
        """
        self.max_history = max_history
        self.topic_threshold = topic_threshold
        
        # Conversation state
        self.history = deque(maxlen=max_history)
        self.topics = Counter()
        self.current_topic = None
        self.conversation_start = None
        self.turn_count = 0
        
        # Topic keywords for classification
        self.topic_keywords = self._initialize_topic_keywords()
        
        logger.info("ConversationAnalyzer initialized")
    
    def _initialize_topic_keywords(self) -> Dict[str, List[str]]:
        """Initialize topic classification keywords"""
        return {
            'work': [
                'job', 'work', 'office', 'meeting', 'deadline', 'project',
                'boss', 'colleague', 'promotion', 'salary', 'career', 'task'
            ],
            'personal': [
                'family', 'friend', 'home', 'life', 'personal', 'private',
                'relationship', 'love', 'dating', 'marriage', 'kids'
            ],
            'health': [
                'health', 'sick', 'doctor', 'medicine', 'hospital', 'pain',
                'exercise', 'gym', 'workout', 'diet', 'sleep', 'tired'
            ],
            'entertainment': [
                'movie', 'music', 'game', 'show', 'concert', 'party',
                'fun', 'weekend', 'vacation', 'hobby', 'sport', 'play'
            ],
            'education': [
                'study', 'exam', 'school', 'university', 'learn', 'course',
                'homework', 'assignment', 'grade', 'teacher', 'student'
            ],
            'technology': [
                'computer', 'phone', 'app', 'internet', 'software', 'code',
                'program', 'website', 'social media', 'tech', 'digital'
            ],
            'daily': [
                'morning', 'evening', 'night', 'today', 'tomorrow', 'yesterday',
                'breakfast', 'lunch', 'dinner', 'coffee', 'commute', 'routine'
            ],
            'emotional': [
                'happy', 'sad', 'angry', 'stressed', 'excited', 'worried',
                'anxious', 'calm', 'peaceful', 'frustrated', 'depressed'
            ]
        }
    
    def analyze(self, 
                text: str,
                external_history: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze conversation turn and update context
        
        Args:
            text: Current conversation text
            external_history: Optional external conversation history
            
        Returns:
            Comprehensive conversation context
        """
        # Initialize conversation if first turn
        if self.conversation_start is None:
            self.conversation_start = datetime.now()
        
        self.turn_count += 1
        
        # Extract features from text
        keywords = self._extract_keywords(text)
        topics = self._detect_topics(text)
        sentiment_indicators = self._detect_sentiment_indicators(text)
        
        # Create turn data
        turn_data = {
            'text': text,
            'keywords': keywords,
            'topics': topics,
            'sentiment_indicators': sentiment_indicators,
            'timestamp': datetime.now().isoformat(),
            'turn_number': self.turn_count
        }
        
        # Add to history
        self.history.append(turn_data)
        
        # Update global topics
        for topic, score in topics.items():
            self.topics[topic] += score
        
        # Determine current topic
        if self.topics:
            self.current_topic = self.topics.most_common(1)[0][0]
        
        # Build comprehensive context
        context = self._build_context(external_history)
        
        return context
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
            'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which'
        }
        
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Get unique keywords maintaining order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:15]  # Limit to top 15
    
    def _detect_topics(self, text: str) -> Dict[str, float]:
        """Detect topics in text"""
        text_lower = text.lower()
        detected_topics = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = 0
            matches = 0
            
            for keyword in keywords:
                if keyword in text_lower:
                    matches += 1
                    # Weight by keyword position in list (earlier = more important)
                    position_weight = 1.0 - (keywords.index(keyword) / len(keywords)) * 0.5
                    score += position_weight
            
            if matches > 0:
                # Normalize score
                normalized_score = score / len(keywords)
                if normalized_score >= self.topic_threshold:
                    detected_topics[topic] = normalized_score
        
        return detected_topics
    
    def _detect_sentiment_indicators(self, text: str) -> Dict[str, List[str]]:
        """Detect sentiment indicators in text"""
        text_lower = text.lower()
        indicators = {
            'positive': [],
            'negative': [],
            'question': [],
            'exclamation': []
        }
        
        # Positive indicators
        positive_words = [
            'good', 'great', 'awesome', 'amazing', 'wonderful', 'excellent',
            'happy', 'love', 'perfect', 'beautiful', 'fantastic', 'best'
        ]
        
        # Negative indicators
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'sad', 'angry', 'frustrated', 'disappointed', 'annoyed'
        ]
        
        # Check for indicators
        for word in positive_words:
            if word in text_lower:
                indicators['positive'].append(word)
        
        for word in negative_words:
            if word in text_lower:
                indicators['negative'].append(word)
        
        # Check for questions
        if '?' in text:
            indicators['question'].append('present')
        
        # Check for exclamations
        if '!' in text:
            indicators['exclamation'].append('present')
        
        return indicators
    
    def _build_context(self, external_history: Optional[List[Dict]] = None) -> Dict:
        """Build comprehensive conversation context"""
        
        # Combine internal and external history
        combined_history = list(self.history)
        if external_history:
            combined_history = external_history[-5:] + combined_history
        
        # Extract all keywords from recent history
        all_keywords = []
        for turn in combined_history[-5:]:
            if isinstance(turn, dict):
                all_keywords.extend(turn.get('keywords', []))
        
        # Get keyword frequency
        keyword_freq = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_freq.most_common(10)]
        
        # Calculate conversation metrics
        conversation_duration = None
        if self.conversation_start:
            conversation_duration = (datetime.now() - self.conversation_start).seconds
        
        # Determine conversation phase
        phase = self._determine_conversation_phase()
        
        # Get recent topics
        recent_topics = []
        for turn in combined_history[-3:]:
            if isinstance(turn, dict) and 'topics' in turn:
                recent_topics.extend(turn['topics'].keys())
        
        # Build summary
        summary = self._generate_summary(combined_history)
        
        context = {
            'keywords': top_keywords,
            'current_topic': self.current_topic,
            'all_topics': dict(self.topics.most_common(5)),
            'recent_topics': list(set(recent_topics)),
            'conversation_phase': phase,
            'turn_count': self.turn_count,
            'duration_seconds': conversation_duration,
            'history': combined_history,
            'summary': summary,
            'sentiment_flow': self._analyze_sentiment_flow(combined_history)
        }
        
        return context
    
    def _determine_conversation_phase(self) -> str:
        """Determine current phase of conversation"""
        if self.turn_count <= 2:
            return 'opening'
        elif self.turn_count <= 5:
            return 'developing'
        elif self.turn_count <= 10:
            return 'established'
        else:
            return 'extended'
    
    def _generate_summary(self, history: List[Dict]) -> str:
        """Generate conversation summary"""
        if not history:
            return "No conversation history"
        
        # Get main topics
        main_topics = list(self.topics.most_common(3))
        
        if not main_topics:
            return "General conversation"
        
        # Create summary
        topic_str = ", ".join([t[0] for t in main_topics[:2]])
        
        if self.turn_count == 1:
            return f"Just started discussing {topic_str}"
        elif self.turn_count < 5:
            return f"Brief conversation about {topic_str}"
        else:
            return f"Extended discussion about {topic_str}"
    
    def _analyze_sentiment_flow(self, history: List[Dict]) -> Dict:
        """Analyze sentiment flow through conversation"""
        if not history:
            return {'trend': 'neutral', 'stability': 1.0}
        
        positive_count = 0
        negative_count = 0
        
        for turn in history[-5:]:
            if isinstance(turn, dict) and 'sentiment_indicators' in turn:
                indicators = turn['sentiment_indicators']
                positive_count += len(indicators.get('positive', []))
                negative_count += len(indicators.get('negative', []))
        
        if positive_count > negative_count * 2:
            trend = 'positive'
        elif negative_count > positive_count * 2:
            trend = 'negative'
        else:
            trend = 'neutral'
        
        # Calculate stability (how consistent the sentiment is)
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            stability = 1.0
        else:
            stability = abs(positive_count - negative_count) / total_indicators
        
        return {
            'trend': trend,
            'stability': stability,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count
        }
    
    def get_conversation_stats(self) -> Dict:
        """Get conversation statistics"""
        return {
            'total_turns': self.turn_count,
            'unique_topics': len(self.topics),
            'current_topic': self.current_topic,
            'top_topics': dict(self.topics.most_common(5)),
            'conversation_start': self.conversation_start.isoformat() if self.conversation_start else None,
            'history_size': len(self.history)
        }
    
    def reset(self):
        """Reset conversation state"""
        self.history.clear()
        self.topics.clear()
        self.current_topic = None
        self.conversation_start = None
        self.turn_count = 0
        logger.info("Conversation analyzer reset")
    
    def should_trigger_music(self, emotion_confidence: float) -> bool:
        """Determine if music should be triggered based on conversation state"""
        
        # Don't trigger in opening phase
        if self.turn_count <= 1:
            return False
        
        # Check emotion confidence
        if emotion_confidence < 0.6:
            return False
        
        # Check if conversation is emotional
        if 'emotional' not in self.topics and self.topics['emotional'] < 0.5:
            # Check sentiment flow
            sentiment = self._analyze_sentiment_flow(list(self.history))
            if sentiment['trend'] == 'neutral' and sentiment['stability'] > 0.7:
                return False
        
        return True


# Testing function
def test_conversation_analyzer():
    """Test conversation analyzer functionality"""
    logger.info("Testing Conversation Analyzer...")
    
    analyzer = ConversationAnalyzer()
    
    # Test conversation
    conversation = [
        "Good morning! Just woke up feeling great.",
        "Time to get ready for work, big presentation today.",
        "I'm a bit nervous about the meeting with the clients.",
        "But I've prepared well, so it should go smoothly.",
        "Wish me luck! This could lead to a promotion."
    ]
    
    for i, text in enumerate(conversation, 1):
        logger.info(f"\nTurn {i}: {text}")
        context = analyzer.analyze(text)
        
        logger.info(f"Current Topic: {context['current_topic']}")
        logger.info(f"Keywords: {', '.join(context['keywords'][:5])}")
        logger.info(f"Phase: {context['conversation_phase']}")
        logger.info(f"Summary: {context['summary']}")
    
    # Get final stats
    stats = analyzer.get_conversation_stats()
    logger.info(f"\nFinal Statistics:")
    logger.info(f"Total Turns: {stats['total_turns']}")
    logger.info(f"Topics: {stats['top_topics']}")
    
    logger.success("Conversation analyzer test completed!")


if __name__ == "__main__":
    test_conversation_analyzer()