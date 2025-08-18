# models/emotion_detector.py
"""
Emotion Detection Module using Mistral-7B-Instruct
Analyzes conversational text to detect emotional states with confidence scores
"""

import os
import json
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from loguru import logger
import numpy as np


class EmotionCategory(Enum):
    """Supported emotion categories"""
    HAPPY = "happy"
    SAD = "sad"
    STRESSED = "stressed"
    CALM = "calm"
    ENERGETIC = "energetic"
    NEUTRAL = "neutral"
    ROMANTIC = "romantic"
    ANGRY = "angry"


@dataclass
class EmotionResult:
    """Emotion detection result"""
    emotion: str
    confidence: float
    intensity: float
    keywords: List[str]
    reasoning: str
    raw_scores: Dict[str, float]


class EmotionDetector:
    """
    Advanced emotion detection using Mistral-7B with pattern matching fallback
    Implements hybrid approach for robust emotion understanding
    """
    
    def __init__(self, 
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 device: Optional[str] = None,
                 use_quantization: bool = True):
        """
        Initialize emotion detector with Mistral-7B
        
        Args:
            model_name: HuggingFace model identifier
            device: Device for computation (cuda/cpu/auto)
            use_quantization: Enable 4-bit quantization for memory efficiency
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Emotion patterns for pattern matching
        self.emotion_patterns = self._load_emotion_patterns()
        
        # Initialize model
        self._initialize_model(use_quantization)
        
        # Cache for recent predictions
        self.cache = {}
        self.cache_size = 100
        
        logger.success(f"EmotionDetector initialized with {model_name}")
    
    def _load_emotion_patterns(self) -> Dict[str, Dict]:
        """Load emotion detection patterns"""
        return {
            EmotionCategory.HAPPY.value: {
                'keywords': [
                    'happy', 'joy', 'excited', 'amazing', 'wonderful', 
                    'fantastic', 'great', 'awesome', 'love', 'blessed',
                    'thrilled', 'delighted', 'cheerful', 'elated'
                ],
                'phrases': [
                    'feeling great', 'best day', 'so happy', 'love this',
                    'over the moon', 'on cloud nine', 'living my best life',
                    'couldn\'t be better', 'absolutely amazing'
                ],
                'intensifiers': ['very', 'super', 'extremely', 'really', 'so']
            },
            EmotionCategory.SAD.value: {
                'keywords': [
                    'sad', 'depressed', 'lonely', 'miss', 'lost', 'cry',
                    'tears', 'broken', 'hurt', 'pain', 'empty', 'hopeless',
                    'miserable', 'devastated', 'heartbroken'
                ],
                'phrases': [
                    'feeling down', 'hard time', 'not okay', 'struggling with',
                    'falling apart', 'can\'t cope', 'want to cry', 'feeling blue',
                    'at my lowest', 'everything hurts'
                ],
                'intensifiers': ['very', 'so', 'extremely', 'really', 'deeply']
            },
            EmotionCategory.STRESSED.value: {
                'keywords': [
                    'stressed', 'anxious', 'worried', 'overwhelmed', 'pressure',
                    'panic', 'tense', 'nervous', 'deadline', 'exam', 'frustrated'
                ],
                'phrases': [
                    'too much', 'can\'t handle', 'falling apart', 'breaking point',
                    'losing my mind', 'going crazy', 'under pressure', 'freaking out'
                ],
                'intensifiers': ['completely', 'totally', 'absolutely', 'extremely']
            },
            EmotionCategory.CALM.value: {
                'keywords': [
                    'peaceful', 'relaxed', 'quiet', 'serene', 'tranquil',
                    'calm', 'zen', 'meditation', 'chill', 'mellow'
                ],
                'phrases': [
                    'taking it easy', 'chilling out', 'winding down',
                    'at peace', 'feeling zen', 'nice and quiet'
                ],
                'intensifiers': ['very', 'quite', 'pretty', 'really']
            },
            EmotionCategory.ENERGETIC.value: {
                'keywords': [
                    'pumped', 'energized', 'ready', 'motivated', 'excited',
                    'workout', 'gym', 'running', 'active', 'power'
                ],
                'phrases': [
                    'let\'s go', 'bring it on', 'ready to', 'pumped up',
                    'full of energy', 'feeling strong', 'time to grind'
                ],
                'intensifiers': ['super', 'totally', 'absolutely', 'really']
            },
            EmotionCategory.ROMANTIC.value: {
                'keywords': [
                    'love', 'romantic', 'heart', 'kiss', 'dating', 'crush',
                    'butterflies', 'soulmate', 'relationship', 'anniversary'
                ],
                'phrases': [
                    'in love', 'falling for', 'special someone', 'heart skips',
                    'thinking about you', 'meant to be'
                ],
                'intensifiers': ['deeply', 'truly', 'madly', 'completely']
            }
        }
    
    def _initialize_model(self, use_quantization: bool):
        """Initialize Mistral-7B model with optimization"""
        try:
            logger.info(f"Loading {self.model_name}...")
            
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model configuration
            if use_quantization and self.device != "cpu":
                # 4-bit quantization for GPU
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                # CPU or full precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    device_map="auto" if self.device != "cpu" else None,
                    low_cpu_mem_usage=True
                )
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            
            logger.success(f"Model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def detect(self, 
               text: str, 
               context: Optional[Dict] = None,
               use_cache: bool = True) -> Dict:
        """
        Detect emotion from text with context
        
        Args:
            text: Input text to analyze
            context: Additional context (conversation history, etc.)
            use_cache: Use cached results for repeated queries
            
        Returns:
            Dictionary containing emotion detection results
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        if use_cache and cache_key in self.cache:
            logger.debug("Using cached emotion result")
            return self.cache[cache_key]
        
        # Pattern-based detection
        pattern_result = self._detect_emotion_patterns(text)
        
        # LLM-based detection
        llm_result = self._detect_emotion_llm(text, context)
        
        # Combine results
        final_result = self._combine_results(pattern_result, llm_result, text)
        
        # Cache result
        if use_cache:
            self._update_cache(cache_key, final_result)
        
        return final_result
    
    def _detect_emotion_patterns(self, text: str) -> Dict:
        """Pattern-based emotion detection"""
        text_lower = text.lower()
        scores = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0
            intensity = 0.5
            matched_keywords = []
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Check phrases (weighted higher)
            for phrase in patterns['phrases']:
                if phrase in text_lower:
                    score += 2
                    matched_keywords.append(phrase)
            
            # Check intensifiers
            for intensifier in patterns.get('intensifiers', []):
                if intensifier in text_lower:
                    intensity = min(1.0, intensity + 0.2)
            
            scores[emotion] = {
                'score': score,
                'intensity': intensity,
                'keywords': matched_keywords
            }
        
        # Get best match
        best_emotion = max(scores.items(), key=lambda x: x[1]['score'])
        
        if best_emotion[1]['score'] > 0:
            confidence = min(0.9, 0.3 + (best_emotion[1]['score'] * 0.1))
            return {
                'emotion': best_emotion[0],
                'confidence': confidence,
                'intensity': best_emotion[1]['intensity'],
                'keywords': best_emotion[1]['keywords'],
                'method': 'pattern'
            }
        
        return {
            'emotion': EmotionCategory.NEUTRAL.value,
            'confidence': 0.5,
            'intensity': 0.5,
            'keywords': [],
            'method': 'pattern'
        }
    
    def _detect_emotion_llm(self, 
                           text: str, 
                           context: Optional[Dict] = None) -> Dict:
        """LLM-based emotion detection using Mistral-7B"""
        try:
            # Build prompt
            prompt = self._build_emotion_prompt(text, context)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse response
            result = self._parse_llm_response(response)
            result['method'] = 'llm'
            
            return result
            
        except Exception as e:
            logger.error(f"LLM emotion detection failed: {e}")
            return {
                'emotion': EmotionCategory.NEUTRAL.value,
                'confidence': 0.3,
                'intensity': 0.5,
                'keywords': [],
                'method': 'llm_fallback'
            }
    
    def _build_emotion_prompt(self, text: str, context: Optional[Dict]) -> str:
        """Build prompt for emotion detection"""
        emotions_list = ", ".join([e.value for e in EmotionCategory])
        
        prompt = f"""<s>[INST] You are an emotion detection expert. Analyze the following text and identify the primary emotion.

Text: "{text}"

{f"Context: {context.get('summary', '')}" if context else ""}

Classify the emotion as one of: {emotions_list}

Respond in this exact format:
EMOTION: [emotion]
CONFIDENCE: [0.0-1.0]
INTENSITY: [0.0-1.0]
KEYWORDS: [comma-separated relevant words]

Be precise and consider the overall tone. [/INST]

Let me analyze this text for emotional content.

"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response to extract emotion data"""
        result = {
            'emotion': EmotionCategory.NEUTRAL.value,
            'confidence': 0.5,
            'intensity': 0.5,
            'keywords': []
        }
        
        try:
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('EMOTION:'):
                    emotion = line.split(':', 1)[1].strip().lower()
                    # Validate emotion
                    valid_emotions = [e.value for e in EmotionCategory]
                    if emotion in valid_emotions:
                        result['emotion'] = emotion
                
                elif line.startswith('CONFIDENCE:'):
                    try:
                        conf = float(line.split(':', 1)[1].strip())
                        result['confidence'] = max(0.0, min(1.0, conf))
                    except:
                        pass
                
                elif line.startswith('INTENSITY:'):
                    try:
                        intensity = float(line.split(':', 1)[1].strip())
                        result['intensity'] = max(0.0, min(1.0, intensity))
                    except:
                        pass
                
                elif line.startswith('KEYWORDS:'):
                    keywords = line.split(':', 1)[1].strip()
                    result['keywords'] = [k.strip() for k in keywords.split(',')]
            
        except Exception as e:
            logger.debug(f"Error parsing LLM response: {e}")
        
        return result
    
    def _combine_results(self, 
                        pattern_result: Dict, 
                        llm_result: Dict,
                        text: str) -> Dict:
        """Combine pattern and LLM results intelligently"""
        # If both agree on emotion, boost confidence
        if pattern_result['emotion'] == llm_result['emotion']:
            confidence = min(0.95, (pattern_result['confidence'] + llm_result['confidence']) / 1.8)
            emotion = pattern_result['emotion']
            intensity = max(pattern_result['intensity'], llm_result['intensity'])
        else:
            # Trust LLM more if pattern confidence is low
            if pattern_result['confidence'] < 0.5:
                emotion = llm_result['emotion']
                confidence = llm_result['confidence'] * 0.9
                intensity = llm_result['intensity']
            # Trust pattern more if it has high confidence
            elif pattern_result['confidence'] > 0.7:
                emotion = pattern_result['emotion']
                confidence = pattern_result['confidence']
                intensity = pattern_result['intensity']
            else:
                # Average when uncertain
                emotion = llm_result['emotion']
                confidence = (pattern_result['confidence'] + llm_result['confidence']) / 2
                intensity = (pattern_result['intensity'] + llm_result['intensity']) / 2
        
        # Combine keywords
        all_keywords = list(set(
            pattern_result.get('keywords', []) + 
            llm_result.get('keywords', [])
        ))[:10]  # Limit to 10 keywords
        
        # Generate reasoning
        reasoning = self._generate_reasoning(emotion, confidence, all_keywords, text)
        
        return {
            'emotion': emotion,
            'confidence': float(confidence),
            'intensity': float(intensity),
            'keywords': all_keywords,
            'reasoning': reasoning,
            'pattern_confidence': pattern_result['confidence'],
            'llm_confidence': llm_result['confidence']
        }
    
    def _generate_reasoning(self, 
                           emotion: str, 
                           confidence: float,
                           keywords: List[str], 
                           text: str) -> str:
        """Generate human-readable reasoning for emotion detection"""
        if confidence > 0.8:
            confidence_desc = "high confidence"
        elif confidence > 0.6:
            confidence_desc = "moderate confidence"
        else:
            confidence_desc = "low confidence"
        
        keywords_str = ", ".join(keywords[:5]) if keywords else "contextual cues"
        
        reasoning = (
            f"Detected {emotion} emotion with {confidence_desc} "
            f"based on: {keywords_str}. "
        )
        
        # Add emotion-specific reasoning
        if emotion == EmotionCategory.HAPPY.value:
            reasoning += "The positive language and enthusiastic tone indicate happiness."
        elif emotion == EmotionCategory.SAD.value:
            reasoning += "The melancholic expressions suggest sadness or loss."
        elif emotion == EmotionCategory.STRESSED.value:
            reasoning += "Multiple stress indicators and pressure-related language detected."
        elif emotion == EmotionCategory.CALM.value:
            reasoning += "The peaceful and relaxed language indicates a calm state."
        elif emotion == EmotionCategory.ENERGETIC.value:
            reasoning += "High energy words and action-oriented language detected."
        
        return reasoning
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return str(hash(text.lower().strip()))
    
    def _update_cache(self, key: str, result: Dict):
        """Update result cache with size limit"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (FIFO)
            oldest = list(self.cache.keys())[0]
            del self.cache[oldest]
        
        self.cache[key] = result
    
    def get_emotion_trajectory(self, 
                              history: List[Dict], 
                              window_size: int = 5) -> Dict:
        """
        Analyze emotion trajectory over conversation history
        
        Args:
            history: List of conversation turns with emotions
            window_size: Size of sliding window for analysis
            
        Returns:
            Trajectory analysis with trends and stability
        """
        if not history:
            return {
                'trend': 'stable',
                'stability': 1.0,
                'dominant_emotion': EmotionCategory.NEUTRAL.value,
                'transitions': []
            }
        
        # Extract emotions and confidences
        emotions = [h.get('emotion', EmotionCategory.NEUTRAL.value) for h in history]
        confidences = [h.get('confidence', 0.5) for h in history]
        
        # Calculate stability (how consistent emotions are)
        if len(set(emotions[-window_size:])) == 1:
            stability = 1.0
        else:
            changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
            stability = max(0.0, 1.0 - (changes / len(emotions)))
        
        # Find dominant emotion
        from collections import Counter
        emotion_counts = Counter(emotions[-window_size:])
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        # Detect transitions
        transitions = []
        for i in range(1, len(emotions)):
            if emotions[i] != emotions[i-1]:
                transitions.append({
                    'from': emotions[i-1],
                    'to': emotions[i],
                    'confidence': confidences[i]
                })
        
        # Determine trend
        if len(transitions) == 0:
            trend = 'stable'
        elif len(transitions) > len(emotions) / 2:
            trend = 'volatile'
        else:
            trend = 'transitioning'
        
        return {
            'trend': trend,
            'stability': float(stability),
            'dominant_emotion': dominant_emotion,
            'transitions': transitions[-3:],  # Last 3 transitions
            'confidence_avg': float(np.mean(confidences))
        }
    
    def batch_detect(self, texts: List[str]) -> List[Dict]:
        """
        Batch emotion detection for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of emotion detection results
        """
        results = []
        
        for text in texts:
            result = self.detect(text)
            results.append(result)
        
        return results


# Testing function
def test_emotion_detector():
    """Test emotion detector functionality"""
    logger.info("Testing Antashiri Emotion Detector...")
    
    # Initialize detector
    detector = EmotionDetector()
    
    # Test cases
    test_cases = [
        "I just got promoted! This is the best day of my life!",
        "I'm feeling really stressed about the deadline tomorrow",
        "Missing my family so much, feeling quite lonely",
        "Time for my morning workout, let's crush it!",
        "Just relaxing with a good book and some tea",
        "I'm so angry about what happened today",
        "Thinking about that special someone makes me smile"
    ]
    
    logger.info("Running emotion detection tests...")
    
    for text in test_cases:
        result = detector.detect(text)
        logger.info(f"\nText: {text}")
        logger.info(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.2%})")
        logger.info(f"Intensity: {result['intensity']:.2%}")
        logger.info(f"Keywords: {', '.join(result['keywords'][:5])}")
        logger.info(f"Reasoning: {result['reasoning']}")
    
    # Test trajectory analysis
    history = [
        {'emotion': 'happy', 'confidence': 0.8},
        {'emotion': 'happy', 'confidence': 0.7},
        {'emotion': 'stressed', 'confidence': 0.9},
        {'emotion': 'stressed', 'confidence': 0.8},
        {'emotion': 'calm', 'confidence': 0.7}
    ]
    
    trajectory = detector.get_emotion_trajectory(history)
    logger.info(f"\nEmotion Trajectory Analysis:")
    logger.info(f"Trend: {trajectory['trend']}")
    logger.info(f"Stability: {trajectory['stability']:.2%}")
    logger.info(f"Dominant: {trajectory['dominant_emotion']}")
    
    logger.success("Emotion detector tests completed!")


if __name__ == "__main__":
    test_emotion_detector()