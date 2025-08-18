# core/xai_explainer.py
"""
Explainable AI Module for Antashiri System
Provides transparent explanations for music selection decisions
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class ExplanationComponent:
    """Component of an explanation"""
    category: str  # 'emotion', 'context', 'similarity', 'popularity'
    description: str
    importance: float  # 0-1 scale
    evidence: List[str]


class XAIExplainer:
    """
    Explainable AI system for transparent music selection
    Generates human-readable explanations for all decisions
    """
    
    def __init__(self):
        """Initialize XAI explainer"""
        self.explanation_templates = self._load_templates()
        self.importance_weights = {
            'emotion_match': 0.35,
            'contextual_relevance': 0.30,
            'semantic_similarity': 0.20,
            'user_preference': 0.10,
            'popularity': 0.05
        }
        logger.info("XAIExplainer initialized")
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load explanation templates"""
        return {
            'emotion_match': [
                "The song's {song_emotion} mood perfectly matches your {user_emotion} emotional state",
                "Selected for its {song_emotion} tone to complement your {user_emotion} feeling",
                "This {song_emotion} track aligns with your current {user_emotion} mood"
            ],
            'contextual_relevance': [
                "Based on keywords: {keywords}, this song was highly relevant",
                "Your conversation about {topic} triggered this selection",
                "The context of {context} made this song appropriate"
            ],
            'semantic_similarity': [
                "The song's themes closely match your conversation",
                "Lyrical content aligns with discussed topics",
                "Strong semantic connection to your words"
            ],
            'confidence': [
                "Selected with {confidence}% confidence",
                "High certainty ({confidence}%) in this choice",
                "Confidence level: {confidence}%"
            ],
            'alternative': [
                "Also considered: {alternatives}",
                "Other options included: {alternatives}",
                "Runner-ups were: {alternatives}"
            ]
        }
    
    def explain(self,
                context: Dict,
                emotion: Dict,
                song: Dict,
                alternatives: List[Dict]) -> str:
        """
        Generate comprehensive explanation for music selection
        
        Args:
            context: Conversation context
            emotion: Emotion detection result
            song: Selected song
            alternatives: Alternative song options
            
        Returns:
            Human-readable explanation
        """
        # Gather explanation components
        components = self._gather_components(context, emotion, song)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(
            context, emotion, song
        )
        
        # Generate natural language explanation
        explanation = self._generate_natural_explanation(
            components, feature_importance, alternatives
        )
        
        return explanation
    
    def _gather_components(self,
                          context: Dict,
                          emotion: Dict,
                          song: Dict) -> List[ExplanationComponent]:
        """Gather all explanation components"""
        components = []
        
        # Emotion matching component
        if song.get('emotion') == emotion.get('emotion'):
            components.append(ExplanationComponent(
                category='emotion_match',
                description=f"Perfect emotional alignment: {emotion['emotion']}",
                importance=0.9,
                evidence=[
                    f"Your emotion: {emotion['emotion']}",
                    f"Song emotion: {song.get('emotion', 'unknown')}",
                    f"Match confidence: {emotion.get('confidence', 0):.1%}"
                ]
            ))
        else:
            components.append(ExplanationComponent(
                category='emotion_match',
                description=f"Compatible emotional tones",
                importance=0.6,
                evidence=[
                    f"Your emotion: {emotion['emotion']}",
                    f"Song emotion: {song.get('emotion', 'unknown')}"
                ]
            ))
        
        # Context relevance component
        if context.get('keywords'):
            matching_keywords = self._find_matching_keywords(
                context['keywords'], song
            )
            if matching_keywords:
                components.append(ExplanationComponent(
                    category='contextual_relevance',
                    description=f"Strong contextual match",
                    importance=0.8,
                    evidence=[
                        f"Matched keywords: {', '.join(matching_keywords[:3])}",
                        f"Conversation topic: {context.get('current_topic', 'general')}"
                    ]
                ))
        
        # Semantic similarity component
        if song.get('similarity'):
            similarity_score = song['similarity']
            components.append(ExplanationComponent(
                category='semantic_similarity',
                description=f"High semantic similarity",
                importance=float(similarity_score),
                evidence=[
                    f"Similarity score: {similarity_score:.2f}",
                    "Content aligns with conversation"
                ]
            ))
        
        # Energy level component
        if song.get('energy_level'):
            energy = song['energy_level']
            energy_desc = self._describe_energy_level(energy)
            components.append(ExplanationComponent(
                category='energy',
                description=energy_desc,
                importance=0.5,
                evidence=[f"Energy level: {energy:.1%}"]
            ))
        
        return components
    
    def _find_matching_keywords(self,
                                keywords: List[str],
                                song: Dict) -> List[str]:
        """Find keywords that match song attributes"""
        matching = []
        song_text = f"{song.get('title', '')} {song.get('artist', '')} {song.get('genre', '')}".lower()
        
        for keyword in keywords:
            if keyword.lower() in song_text:
                matching.append(keyword)
        
        # Check trigger words
        if 'trigger_words' in song:
            for trigger in song['trigger_words']:
                trigger_word = trigger.get('word', '').lower()
                for keyword in keywords:
                    if keyword.lower() in trigger_word or trigger_word in keyword.lower():
                        if keyword not in matching:
                            matching.append(keyword)
        
        return matching
    
    def _describe_energy_level(self, energy: float) -> str:
        """Describe energy level in human terms"""
        if energy > 0.8:
            return "High energy for motivation"
        elif energy > 0.6:
            return "Upbeat and energizing"
        elif energy > 0.4:
            return "Moderate energy"
        elif energy > 0.2:
            return "Calm and relaxing"
        else:
            return "Very peaceful and soothing"
    
    def _calculate_feature_importance(self,
                                     context: Dict,
                                     emotion: Dict,
                                     song: Dict) -> Dict[str, float]:
        """Calculate importance of each feature in decision"""
        importance = {}
        
        # Emotion match importance
        if song.get('emotion') == emotion.get('emotion'):
            importance['emotion_match'] = self.importance_weights['emotion_match']
        else:
            importance['emotion_match'] = self.importance_weights['emotion_match'] * 0.5
        
        # Context relevance
        if context.get('keywords'):
            matching_keywords = self._find_matching_keywords(
                context['keywords'], song
            )
            relevance = min(1.0, len(matching_keywords) / 5)
            importance['contextual_relevance'] = (
                self.importance_weights['contextual_relevance'] * relevance
            )
        else:
            importance['contextual_relevance'] = 0.1
        
        # Semantic similarity
        if song.get('similarity'):
            importance['semantic_similarity'] = (
                self.importance_weights['semantic_similarity'] * 
                song['similarity']
            )
        else:
            importance['semantic_similarity'] = 0.1
        
        # Normalize importance scores
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def _generate_natural_explanation(self,
                                     components: List[ExplanationComponent],
                                     feature_importance: Dict[str, float],
                                     alternatives: List[Dict]) -> str:
        """Generate natural language explanation"""
        
        # Sort components by importance
        components.sort(key=lambda x: x.importance, reverse=True)
        
        # Build explanation parts
        explanation_parts = []
        
        # Main reasoning
        explanation_parts.append("🎵 Music Selection Reasoning:\n")
        
        # Add top 3 components
        for component in components[:3]:
            explanation_parts.append(f"• {component.description}")
            if component.evidence and component.importance > 0.7:
                for evidence in component.evidence[:2]:
                    explanation_parts.append(f"  - {evidence}")
        
        # Add feature importance if significant
        explanation_parts.append("\n📊 Decision Factors:")
        sorted_importance = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for feature, importance in sorted_importance[:3]:
            if importance > 0.1:
                feature_name = feature.replace('_', ' ').title()
                explanation_parts.append(f"• {feature_name}: {importance:.1%}")
        
        # Add alternatives if available
        if alternatives:
            explanation_parts.append("\n🎸 Also Considered:")
            for alt in alternatives[:2]:
                explanation_parts.append(
                    f"• {alt.get('title', 'Unknown')} - {alt.get('artist', 'Unknown')}"
                )
        
        return "\n".join(explanation_parts)
    
    def explain_decision_path(self,
                             conversation_history: List[Dict],
                             final_decision: Dict) -> str:
        """Explain the decision path through conversation"""
        
        explanation = "🛤️ Decision Path:\n\n"
        
        # Track emotional journey
        emotions = [turn.get('emotion', 'neutral') for turn in conversation_history]
        unique_emotions = list(dict.fromkeys(emotions))  # Preserve order
        
        if len(unique_emotions) > 1:
            explanation += f"Emotional Journey: {' → '.join(unique_emotions)}\n"
        
        # Track topic evolution
        topics = []
        for turn in conversation_history:
            if 'topics' in turn:
                topics.extend(turn['topics'])
        
        if topics:
            unique_topics = list(dict.fromkeys(topics))[:3]
            explanation += f"Topics Discussed: {', '.join(unique_topics)}\n"
        
        # Final decision reasoning
        explanation += f"\nFinal Selection: {final_decision.get('title', 'Unknown')}\n"
        explanation += f"Reason: Best match for current emotional and contextual state"
        
        return explanation
    
    def generate_simple_explanation(self,
                                   emotion: str,
                                   song: Dict,
                                   confidence: float) -> str:
        """Generate simple explanation for quick display"""
        
        explanation = (
            f"Selected '{song['title']}' by {song['artist']} "
            f"because it matches your {emotion} mood "
            f"(confidence: {confidence:.1%})"
        )
        
        if song.get('reason'):
            explanation += f". {song['reason']}"
        
        return explanation
    
    def explain_no_trigger(self, emotion: str, confidence: float) -> str:
        """Explain why music wasn't triggered"""
        
        reasons = []
        
        if confidence < 0.6:
            reasons.append(f"Emotion detection confidence too low ({confidence:.1%})")
        
        if emotion == 'neutral':
            reasons.append("Neutral emotional state detected")
        
        if not reasons:
            reasons.append("Waiting for stronger emotional cues")
        
        return f"🎵 Music not triggered: {', '.join(reasons)}"
    
    def get_transparency_metrics(self, explanation: str) -> Dict[str, float]:
        """Calculate transparency metrics for explanation"""
        
        # Simple metrics for explanation quality
        metrics = {
            'completeness': 0.0,
            'clarity': 0.0,
            'evidence_based': 0.0
        }
        
        # Check completeness
        required_elements = ['emotion', 'confidence', 'reason']
        present_elements = sum(1 for elem in required_elements if elem in explanation.lower())
        metrics['completeness'] = present_elements / len(required_elements)
        
        # Check clarity (based on length and structure)
        if 50 < len(explanation) < 500:
            metrics['clarity'] = 0.8
        elif len(explanation) <= 50:
            metrics['clarity'] = 0.5
        else:
            metrics['clarity'] = 0.6
        
        # Check evidence
        evidence_markers = ['because', 'based on', 'due to', 'confidence:', 'score:']
        evidence_count = sum(1 for marker in evidence_markers if marker in explanation.lower())
        metrics['evidence_based'] = min(1.0, evidence_count / 3)
        
        return metrics


# Testing function
def test_xai_explainer():
    """Test XAI explainer functionality"""
    logger.info("Testing XAI Explainer...")
    
    explainer = XAIExplainer()
    
    # Test data
    context = {
        'keywords': ['happy', 'celebrate', 'promotion'],
        'current_topic': 'work',
        'conversation_phase': 'established'
    }
    
    emotion = {
        'emotion': 'happy',
        'confidence': 0.85,
        'intensity': 0.8
    }
    
    song = {
        'title': 'Celebration',
        'artist': 'Test Artist',
        'emotion': 'happy',
        'energy_level': 0.9,
        'similarity': 0.75,
        'genre': 'pop'
    }
    
    alternatives = [
        {'title': 'Happy Day', 'artist': 'Another Artist'},
        {'title': 'Good Vibes', 'artist': 'Third Artist'}
    ]
    
    # Generate explanation
    explanation = explainer.explain(context, emotion, song, alternatives)
    logger.info("Full Explanation:")
    print(explanation)
    
    # Generate simple explanation
    simple = explainer.generate_simple_explanation(
        emotion['emotion'], song, emotion['confidence']
    )
    logger.info(f"\nSimple Explanation: {simple}")
    
    # Test transparency metrics
    metrics = explainer.get_transparency_metrics(explanation)
    logger.info(f"\nTransparency Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.1%}")
    
    logger.success("XAI Explainer test completed!")


if __name__ == "__main__":
    test_xai_explainer()