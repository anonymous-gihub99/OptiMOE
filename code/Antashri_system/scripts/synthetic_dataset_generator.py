# scripts/generate_synthetic_dataset.py
"""
Synthetic Dataset Generator for Antashiri System
Creates emotion-annotated music dataset without copyright issues
"""

import json
import random
import hashlib
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path
from loguru import logger


class SyntheticMusicDatasetGenerator:
    """
    Generates synthetic music dataset with emotional and contextual triggers
    No real lyrics or copyrighted content - fully synthetic generation
    """
    
    def __init__(self, seed: int = 42):
        """Initialize generator with reproducible seed"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Emotional categories and their characteristics
        self.emotions = {
            'happy': {
                'energy': (0.6, 0.9),
                'valence': (0.7, 1.0),
                'genres': ['pop', 'dance', 'indie'],
                'tempo': (120, 140)
            },
            'sad': {
                'energy': (0.2, 0.5),
                'valence': (0.0, 0.3),
                'genres': ['ballad', 'indie', 'acoustic'],
                'tempo': (60, 90)
            },
            'energetic': {
                'energy': (0.8, 1.0),
                'valence': (0.5, 0.9),
                'genres': ['electronic', 'rock', 'hip-hop'],
                'tempo': (130, 180)
            },
            'calm': {
                'energy': (0.1, 0.4),
                'valence': (0.4, 0.7),
                'genres': ['ambient', 'classical', 'jazz'],
                'tempo': (60, 100)
            },
            'romantic': {
                'energy': (0.3, 0.6),
                'valence': (0.6, 0.9),
                'genres': ['r&b', 'soul', 'pop'],
                'tempo': (70, 110)
            },
            'stressed': {
                'energy': (0.5, 0.8),
                'valence': (0.2, 0.5),
                'genres': ['alternative', 'indie', 'electronic'],
                'tempo': (100, 130)
            }
        }
        
        # Artist name components
        self.first_names = [
            'Luna', 'Phoenix', 'Echo', 'Nova', 'Stellar', 'Aurora',
            'Cosmic', 'Mystic', 'Crystal', 'Shadow', 'Dream', 'Ocean',
            'Sky', 'River', 'Storm', 'Ember', 'Frost', 'Zephyr'
        ]
        
        self.last_names = [
            'Waves', 'Hearts', 'Souls', 'Dreams', 'Lights', 'Sounds',
            'Beats', 'Vibes', 'Echoes', 'Harmonics', 'Melodies', 'Rhythms',
            'Tones', 'Notes', 'Chords', 'Frequencies', 'Resonance', 'Symphony'
        ]
        
        # Song title templates
        self.title_patterns = [
            "{emotion} {time}",
            "{action} in {place}",
            "{color} {noun}",
            "The {adjective} {object}",
            "{verb}ing {preposition} {destination}",
            "{number} {items}",
            "{feeling} {moment}"
        ]
        
        # Title components by emotion
        self.title_components = {
            'happy': {
                'emotions': ['Joy', 'Bliss', 'Delight', 'Euphoria', 'Elation'],
                'actions': ['Dancing', 'Celebrating', 'Laughing', 'Singing', 'Jumping'],
                'adjectives': ['Bright', 'Shining', 'Golden', 'Radiant', 'Glorious']
            },
            'sad': {
                'emotions': ['Sorrow', 'Melancholy', 'Longing', 'Grief', 'Solitude'],
                'actions': ['Crying', 'Remembering', 'Missing', 'Losing', 'Fading'],
                'adjectives': ['Empty', 'Broken', 'Lost', 'Distant', 'Forgotten']
            },
            'energetic': {
                'emotions': ['Power', 'Force', 'Energy', 'Strength', 'Fire'],
                'actions': ['Running', 'Fighting', 'Rising', 'Conquering', 'Breaking'],
                'adjectives': ['Electric', 'Explosive', 'Unstoppable', 'Wild', 'Fierce']
            },
            'calm': {
                'emotions': ['Peace', 'Serenity', 'Tranquility', 'Harmony', 'Balance'],
                'actions': ['Floating', 'Breathing', 'Resting', 'Meditating', 'Drifting'],
                'adjectives': ['Gentle', 'Soft', 'Quiet', 'Serene', 'Peaceful']
            },
            'romantic': {
                'emotions': ['Love', 'Passion', 'Desire', 'Devotion', 'Affection'],
                'actions': ['Holding', 'Kissing', 'Embracing', 'Loving', 'Cherishing'],
                'adjectives': ['Tender', 'Sweet', 'Intimate', 'Eternal', 'Beautiful']
            },
            'stressed': {
                'emotions': ['Pressure', 'Tension', 'Anxiety', 'Chaos', 'Urgency'],
                'actions': ['Breaking', 'Escaping', 'Struggling', 'Fighting', 'Surviving'],
                'adjectives': ['Tense', 'Heavy', 'Dark', 'Overwhelming', 'Relentless']
            }
        }
        
        # Trigger word patterns
        self.trigger_patterns = {
            'conversational': [
                "feeling {emotion}",
                "need some {mood} music",
                "time to {action}",
                "perfect for {situation}",
                "makes me feel {feeling}"
            ],
            'contextual': [
                "{time_of_day} vibes",
                "{activity} playlist",
                "{location} soundtrack",
                "{event} music",
                "{weather} day songs"
            ],
            'emotional': [
                "when you're {emotion}",
                "for {mood} moments",
                "{feeling} state of mind",
                "captures the {emotion}",
                "expresses {feeling}"
            ]
        }
        
        logger.info("SyntheticMusicDatasetGenerator initialized")
    
    def generate_artist_name(self) -> str:
        """Generate synthetic artist name"""
        if random.random() < 0.7:
            # Single artist
            return f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
        else:
            # Band name
            return f"The {random.choice(self.first_names)} {random.choice(self.last_names)}"
    
    def generate_song_title(self, emotion: str) -> str:
        """Generate song title based on emotion"""
        components = self.title_components[emotion]
        
        # Select pattern
        pattern = random.choice(self.title_patterns)
        
        # Fill in pattern
        replacements = {
            '{emotion}': random.choice(components['emotions']),
            '{action}': random.choice(components['actions']),
            '{adjective}': random.choice(components['adjectives']),
            '{time}': random.choice(['Morning', 'Night', 'Dawn', 'Sunset', 'Midnight']),
            '{place}': random.choice(['Paradise', 'Heaven', 'Dreams', 'Space', 'Ocean']),
            '{color}': random.choice(['Blue', 'Golden', 'Silver', 'Crimson', 'Violet']),
            '{noun}': random.choice(['Sky', 'Heart', 'Soul', 'Mind', 'Spirit']),
            '{object}': random.choice(['Journey', 'Story', 'Memory', 'Promise', 'Secret']),
            '{verb}': random.choice(['Drift', 'Soar', 'Fall', 'Rise', 'Flow']),
            '{preposition}': random.choice(['Through', 'Beyond', 'Within', 'Above', 'Beneath']),
            '{destination}': random.choice(['Stars', 'Clouds', 'Time', 'Space', 'Dreams']),
            '{number}': random.choice(['Thousand', 'Million', 'Endless', 'Infinite', 'Seven']),
            '{items}': random.choice(['Reasons', 'Ways', 'Moments', 'Dreams', 'Wishes']),
            '{feeling}': random.choice(components['emotions']),
            '{moment}': random.choice(['Moment', 'Hour', 'Day', 'Night', 'Forever'])
        }
        
        title = pattern
        for key, value in replacements.items():
            title = title.replace(key, value)
        
        return title
    
    def generate_trigger_words(self, emotion: str, song_title: str, artist: str) -> List[Dict]:
        """Generate trigger words for a song"""
        triggers = []
        
        # Direct triggers from title and artist
        title_words = [w.lower() for w in song_title.split() if len(w) > 3]
        for word in title_words[:3]:
            triggers.append({
                'word': word,
                'trigger_type': 'direct',
                'category': 'direct_reference',
                'frequency': 1,
                'confidence': 0.9,
                'context': 'title_keyword'
            })
        
        # Artist name trigger
        artist_first = artist.split()[0].lower()
        triggers.append({
            'word': artist_first,
            'trigger_type': 'direct',
            'category': 'direct_reference',
            'frequency': 1,
            'confidence': 0.95,
            'context': 'artist_name'
        })
        
        # Emotional triggers
        emotion_words = {
            'happy': ['joy', 'excited', 'celebration', 'smile', 'cheerful'],
            'sad': ['tears', 'lonely', 'miss', 'heartbreak', 'melancholy'],
            'energetic': ['pump', 'workout', 'energy', 'power', 'motivation'],
            'calm': ['relax', 'peace', 'quiet', 'meditation', 'serene'],
            'romantic': ['love', 'heart', 'together', 'kiss', 'forever'],
            'stressed': ['pressure', 'overwhelmed', 'anxious', 'tense', 'deadline']
        }
        
        for word in random.sample(emotion_words.get(emotion, []), min(3, len(emotion_words.get(emotion, [])))):
            triggers.append({
                'word': word,
                'trigger_type': 'emotion',
                'category': 'emotional_context',
                'frequency': 1,
                'confidence': 0.8,
                'context': f'emotion_{emotion}'
            })
        
        # Contextual triggers
        contexts = {
            'happy': ['party', 'celebration', 'weekend', 'friends'],
            'sad': ['alone', 'night', 'rain', 'memories'],
            'energetic': ['gym', 'morning', 'workout', 'running'],
            'calm': ['evening', 'study', 'sleep', 'reading'],
            'romantic': ['date', 'anniversary', 'valentine', 'wedding'],
            'stressed': ['work', 'deadline', 'exam', 'monday']
        }
        
        for context_word in random.sample(contexts.get(emotion, []), min(2, len(contexts.get(emotion, [])))):
            triggers.append({
                'word': context_word,
                'trigger_type': 'contextual',
                'category': 'cultural_reference',
                'frequency': 1,
                'confidence': 0.7,
                'context': f'situation_{context_word}'
            })
        
        # Conversational phrases
        phrases = [
            f"feeling {emotion}",
            f"need {emotion} music",
            f"perfect for {random.choice(contexts.get(emotion, ['now']))}"
        ]
        
        for phrase in random.sample(phrases, min(2, len(phrases))):
            triggers.append({
                'word': phrase,
                'trigger_type': 'conversational',
                'category': 'lyric_fragment',
                'frequency': 2,
                'confidence': 0.85,
                'context': 'common_phrase'
            })
        
        return triggers
    
    def generate_song(self, emotion: str, index: int) -> Dict:
        """Generate a single synthetic song"""
        
        # Generate basic metadata
        title = self.generate_song_title(emotion)
        artist = self.generate_artist_name()
        
        # Get emotion characteristics
        emotion_chars = self.emotions[emotion]
        
        # Generate song properties
        energy_level = np.random.uniform(*emotion_chars['energy'])
        valence = np.random.uniform(*emotion_chars['valence'])
        tempo = np.random.uniform(*emotion_chars['tempo'])
        genre = random.choice(emotion_chars['genres'])
        
        # Generate unique ID
        song_id = hashlib.md5(f"{title}_{artist}_{index}".encode()).hexdigest()[:12]
        
        # Generate trigger words
        trigger_words = self.generate_trigger_words(emotion, title, artist)
        
        # Generate additional metadata
        year = random.randint(2018, 2024)
        popularity = np.random.beta(2, 5)  # Skewed towards lower values
        duration = random.randint(150, 300)  # 2.5 to 5 minutes in seconds
        
        return {
            'song_id': song_id,
            'title': title,
            'artist': artist,
            'genre': genre,
            'emotion': emotion,
            'energy_level': float(energy_level),
            'valence': float(valence),
            'tempo': float(tempo),
            'year': year,
            'popularity': float(popularity),
            'duration': duration,
            'trigger_words': trigger_words
        }
    
    def generate_dataset(self, num_songs: int = 30000) -> Dict:
        """Generate complete synthetic dataset"""
        
        logger.info(f"Generating {num_songs} synthetic songs...")
        
        songs = []
        emotion_list = list(self.emotions.keys())
        
        # Calculate songs per emotion for balanced distribution
        songs_per_emotion = num_songs // len(emotion_list)
        remainder = num_songs % len(emotion_list)
        
        # Generate songs for each emotion
        for emotion_idx, emotion in enumerate(emotion_list):
            # Add extra song to first emotions if remainder exists
            count = songs_per_emotion + (1 if emotion_idx < remainder else 0)
            
            logger.info(f"Generating {count} {emotion} songs...")
            
            for i in range(count):
                song = self.generate_song(emotion, len(songs))
                songs.append(song)
        
        # Shuffle to mix emotions
        random.shuffle(songs)
        
        # Calculate statistics
        trigger_stats = Counter()
        emotion_distribution = Counter()
        genre_distribution = Counter()
        
        for song in songs:
            emotion_distribution[song['emotion']] += 1
            genre_distribution[song['genre']] += 1
            for trigger in song['trigger_words']:
                trigger_stats[trigger['category']] += 1
        
        # Create final dataset structure
        dataset = {
            'metadata': {
                'total_songs': len(songs),
                'version': '1.0',
                'generator': 'Antashiri Synthetic Generator',
                'emotions': list(emotion_list),
                'trigger_categories': {
                    'direct_reference': 0.30,
                    'lyric_fragment': 0.25,
                    'emotional_context': 0.25,
                    'cultural_reference': 0.20
                },
                'emotion_distribution': dict(emotion_distribution),
                'genre_distribution': dict(genre_distribution),
                'trigger_stats': dict(trigger_stats),
                'total_triggers': sum(trigger_stats.values())
            },
            'songs': songs
        }
        
        logger.success(f"Generated {len(songs)} songs with {sum(trigger_stats.values())} triggers")
        
        # Print statistics
        logger.info("\n📊 Dataset Statistics:")
        logger.info(f"Total songs: {len(songs)}")
        logger.info(f"Total triggers: {sum(trigger_stats.values())}")
        logger.info("\n📈 Emotion Distribution:")
        for emotion, count in emotion_distribution.items():
            logger.info(f"  {emotion}: {count} songs ({count/len(songs)*100:.1f}%)")
        logger.info("\n🎵 Genre Distribution:")
        for genre, count in genre_distribution.most_common(10):
            logger.info(f"  {genre}: {count} songs")
        
        return dataset
    
    def save_dataset(self, dataset: Dict, filepath: str = 'data/synthetic_music_dataset.json'):
        """Save dataset to JSON file"""
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.success(f"Dataset saved to {filepath}")
        
        # Save a smaller sample for testing
        sample_dataset = {
            'metadata': dataset['metadata'],
            'songs': dataset['songs'][:100]
        }
        
        sample_path = filepath.replace('.json', '_sample.json')
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Sample dataset (100 songs) saved to {sample_path}")
    
    def validate_dataset(self, dataset: Dict) -> bool:
        """Validate dataset structure and content"""
        
        required_fields = ['metadata', 'songs']
        if not all(field in dataset for field in required_fields):
            logger.error("Missing required fields in dataset")
            return False
        
        # Check metadata
        metadata = dataset['metadata']
        if metadata['total_songs'] != len(dataset['songs']):
            logger.error("Song count mismatch")
            return False
        
        # Check songs
        for song in dataset['songs'][:10]:  # Check first 10 songs
            required_song_fields = ['song_id', 'title', 'artist', 'emotion', 'trigger_words']
            if not all(field in song for field in required_song_fields):
                logger.error(f"Missing fields in song: {song.get('title', 'Unknown')}")
                return False
        
        logger.success("Dataset validation passed")
        return True


# Main execution
def main():
    """Generate synthetic dataset for Antashiri"""
    
    logger.info("="*60)
    logger.info("🎵 ANTASHIRI SYNTHETIC DATASET GENERATOR")
    logger.info("="*60)
    
    # Initialize generator
    generator = SyntheticMusicDatasetGenerator(seed=42)
    
    # Generate dataset
    dataset = generator.generate_dataset(num_songs=20000)
    
    # Validate dataset
    if generator.validate_dataset(dataset):
        # Save dataset
        generator.save_dataset(dataset)
        
        logger.info("\n✅ Dataset generation complete!")
        logger.info(f"Total songs: {dataset['metadata']['total_songs']}")
        logger.info(f"Total triggers: {dataset['metadata']['total_triggers']}")
    else:
        logger.error("Dataset validation failed!")
    
    logger.info("="*60)


if __name__ == "__main__":
    main()