# models/music_retriever.py
"""
Music Retrieval Module using FAISS and Sentence Embeddings
Implements semantic search for contextual music selection
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger
import torch


@dataclass
class Song:
    """Song data structure"""
    song_id: str
    title: str
    artist: str
    genre: str
    emotion: str
    energy_level: float
    trigger_words: List[Dict]
    popularity: float = 0.5
    year: int = 2020


class MusicRetriever:
    """
    Advanced music retrieval system using semantic search
    Combines FAISS indexing with contextual ranking
    """
    
    def __init__(self,
                 dataset_path: str = "data/synthetic_music_dataset.json",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: Optional[str] = None,
                 use_gpu: bool = False):
        """
        Initialize music retriever
        
        Args:
            dataset_path: Path to music dataset JSON
            embedding_model: Sentence transformer model for embeddings
            index_path: Path to save/load FAISS index
            use_gpu: Use GPU for FAISS operations if available
        """
        self.dataset_path = dataset_path
        self.index_path = index_path or "data/faiss_index.bin"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Load dataset
        self.songs = self._load_dataset()
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        if self.use_gpu:
            self.encoder = self.encoder.cuda()
        logger.success("Embedding model loaded")
        
        # Initialize or load FAISS index
        self.index = None
        self.song_embeddings = None
        self._initialize_index()
        
        # Emotion compatibility matrix
        self.emotion_compatibility = self._build_emotion_compatibility()
        
        logger.success(f"MusicRetriever initialized with {len(self.songs)} songs")
    
    def _load_dataset(self) -> List[Song]:
        """Load music dataset from JSON"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            songs = []
            for song_data in data.get('songs', []):
                song = Song(
                    song_id=song_data['song_id'],
                    title=song_data['title'],
                    artist=song_data['artist'],
                    genre=song_data.get('genre', 'unknown'),
                    emotion=song_data.get('emotion', 'neutral'),
                    energy_level=song_data.get('energy_level', 0.5),
                    trigger_words=song_data.get('trigger_words', []),
                    popularity=song_data.get('popularity', 0.5),
                    year=song_data.get('year', 2020)
                )
                songs.append(song)
            
            logger.info(f"Loaded {len(songs)} songs from dataset")
            return songs
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []
    
    def _initialize_index(self):
        """Initialize or load FAISS index"""
        index_file = Path(self.index_path)
        embeddings_file = Path(self.index_path.replace('.bin', '_embeddings.npy'))
        
        # Try to load existing index
        if index_file.exists() and embeddings_file.exists():
            try:
                logger.info("Loading existing FAISS index...")
                self.index = faiss.read_index(str(index_file))
                self.song_embeddings = np.load(str(embeddings_file))
                
                if self.use_gpu:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                
                logger.success("FAISS index loaded from disk")
                return
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
        
        # Build new index
        logger.info("Building new FAISS index...")
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index from songs"""
        # Generate embeddings for all songs
        song_texts = []
        
        for song in self.songs:
            # Combine song metadata and triggers for embedding
            trigger_text = ' '.join([
                t.get('word', '') for t in song.trigger_words
            ])
            
            song_text = (
                f"{song.title} {song.artist} {song.genre} "
                f"{song.emotion} {trigger_text}"
            )
            song_texts.append(song_text)
        
        # Create embeddings
        logger.info(f"Creating embeddings for {len(song_texts)} songs...")
        self.song_embeddings = self.encoder.encode(
            song_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings
        faiss.normalize_L2(self.song_embeddings)
        
        # Create FAISS index
        dimension = self.song_embeddings.shape[1]
        
        if self.use_gpu:
            # GPU index
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatIP(dimension)
            self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        else:
            # CPU index with optimization
            self.index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        self.index.add(self.song_embeddings)
        
        # Save index
        self._save_index()
        
        logger.success(f"FAISS index built with {len(self.songs)} songs")
    
    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            index_file = Path(self.index_path)
            index_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert GPU index to CPU for saving
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_file))
            else:
                faiss.write_index(self.index, str(index_file))
            
            # Save embeddings
            embeddings_file = str(index_file).replace('.bin', '_embeddings.npy')
            np.save(embeddings_file, self.song_embeddings)
            
            logger.info(f"Index saved to {index_file}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _build_emotion_compatibility(self) -> Dict[str, Dict[str, float]]:
        """Build emotion compatibility matrix"""
        return {
            'happy': {
                'happy': 1.0, 'energetic': 0.8, 'romantic': 0.6,
                'calm': 0.4, 'neutral': 0.5, 'sad': 0.2, 'stressed': 0.3
            },
            'sad': {
                'sad': 1.0, 'calm': 0.7, 'romantic': 0.5,
                'neutral': 0.5, 'happy': 0.2, 'energetic': 0.1, 'stressed': 0.4
            },
            'stressed': {
                'stressed': 1.0, 'energetic': 0.6, 'angry': 0.7,
                'calm': 0.8, 'neutral': 0.5, 'happy': 0.3, 'sad': 0.4
            },
            'calm': {
                'calm': 1.0, 'romantic': 0.7, 'neutral': 0.6,
                'sad': 0.5, 'happy': 0.4, 'stressed': 0.2, 'energetic': 0.3
            },
            'energetic': {
                'energetic': 1.0, 'happy': 0.8, 'angry': 0.6,
                'neutral': 0.4, 'calm': 0.3, 'sad': 0.1, 'stressed': 0.5
            },
            'romantic': {
                'romantic': 1.0, 'calm': 0.7, 'happy': 0.6,
                'sad': 0.5, 'neutral': 0.4, 'energetic': 0.3, 'stressed': 0.2
            },
            'neutral': {
                'neutral': 1.0, 'calm': 0.6, 'happy': 0.5,
                'sad': 0.5, 'energetic': 0.4, 'romantic': 0.4, 'stressed': 0.4
            }
        }
    
    def retrieve(self,
                context: Dict,
                emotion: str,
                top_k: int = 10,
                energy_threshold: Optional[float] = None) -> List[Dict]:
        """
        Retrieve songs based on context and emotion
        
        Args:
            context: Conversation context with keywords
            emotion: Detected emotion
            top_k: Number of songs to retrieve
            energy_threshold: Minimum energy level required
            
        Returns:
            List of matching songs with scores
        """
        # Build query embedding
        query_text = self._build_query_text(context, emotion)
        query_embedding = self.encoder.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        k_search = min(top_k * 3, len(self.songs))  # Search more for filtering
        distances, indices = self.index.search(query_embedding, k_search)
        
        # Score and rank candidates
        candidates = []
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.songs):
                song = self.songs[idx]
                
                # Calculate comprehensive score
                score = self._calculate_song_score(
                    song, emotion, distance, context, energy_threshold
                )
                
                if score > 0:
                    candidates.append({
                        'song_id': song.song_id,
                        'title': song.title,
                        'artist': song.artist,
                        'genre': song.genre,
                        'emotion': song.emotion,
                        'energy_level': song.energy_level,
                        'score': score,
                        'similarity': float(distance),
                        'reason': self._get_selection_reason(song, emotion, score)
                    })
        
        # Sort by score and return top K
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:top_k]
    
    def _build_query_text(self, context: Dict, emotion: str) -> str:
        """Build query text from context"""
        parts = []
        
        # Add emotion
        parts.append(emotion)
        
        # Add keywords from context
        if 'keywords' in context:
            parts.extend(context['keywords'][:10])
        
        # Add recent conversation snippets
        if 'history' in context:
            for turn in context['history'][-3:]:
                if 'text' in turn:
                    # Extract key words from text
                    words = turn['text'].lower().split()
                    important_words = [w for w in words if len(w) > 4]
                    parts.extend(important_words[:3])
        
        return ' '.join(parts)
    
    def _calculate_song_score(self,
                             song: Song,
                             emotion: str,
                             similarity: float,
                             context: Dict,
                             energy_threshold: Optional[float]) -> float:
        """Calculate comprehensive score for song"""
        # Base score from similarity
        score = float(similarity)
        
        # Emotion compatibility bonus
        compatibility = self.emotion_compatibility.get(emotion, {}).get(song.emotion, 0.5)
        score *= (0.7 + 0.3 * compatibility)
        
        # Energy level matching
        if energy_threshold is not None:
            if song.energy_level < energy_threshold:
                score *= 0.5
        
        # Trigger word matching bonus
        if 'keywords' in context:
            context_keywords = set(k.lower() for k in context['keywords'])
            trigger_words = set()
            
            for trigger in song.trigger_words:
                if 'word' in trigger:
                    trigger_words.add(trigger['word'].lower())
            
            overlap = len(context_keywords & trigger_words)
            if overlap > 0:
                score *= (1.0 + 0.1 * min(overlap, 5))
        
        # Popularity adjustment (slight boost for popular songs)
        score *= (0.9 + 0.1 * song.popularity)
        
        # Genre preference (if specified in context)
        if 'preferred_genres' in context:
            if song.genre in context['preferred_genres']:
                score *= 1.2
        
        return score
    
    def _get_selection_reason(self, song: Song, emotion: str, score: float) -> str:
        """Generate reason for song selection"""
        reasons = []
        
        # Emotion match
        if song.emotion == emotion:
            reasons.append(f"Perfect {emotion} mood match")
        else:
            compatibility = self.emotion_compatibility.get(emotion, {}).get(song.emotion, 0.5)
            if compatibility > 0.7:
                reasons.append(f"Compatible with {emotion} mood")
        
        # Energy level
        if song.energy_level > 0.7:
            reasons.append("High energy")
        elif song.energy_level < 0.3:
            reasons.append("Relaxing")
        
        # Score-based reason
        if score > 0.8:
            reasons.append("Strong contextual match")
        elif score > 0.6:
            reasons.append("Good contextual fit")
        
        return " | ".join(reasons) if reasons else "Semantic similarity"
    
    def get_song_by_id(self, song_id: str) -> Optional[Dict]:
        """Get song details by ID"""
        for song in self.songs:
            if song.song_id == song_id:
                return asdict(song)
        return None
    
    def search_by_artist(self, artist: str, limit: int = 10) -> List[Dict]:
        """Search songs by artist name"""
        results = []
        artist_lower = artist.lower()
        
        for song in self.songs:
            if artist_lower in song.artist.lower():
                results.append(asdict(song))
                if len(results) >= limit:
                    break
        
        return results
    
    def get_songs_by_emotion(self, emotion: str, limit: int = 20) -> List[Dict]:
        """Get songs for specific emotion"""
        results = []
        
        for song in self.songs:
            if song.emotion == emotion:
                results.append(asdict(song))
                if len(results) >= limit:
                    break
        
        return results
    
    def update_index(self, new_songs: List[Dict]):
        """Update index with new songs"""
        for song_data in new_songs:
            song = Song(**song_data)
            self.songs.append(song)
        
        # Rebuild index
        self._build_index()
        logger.info(f"Index updated with {len(new_songs)} new songs")


# Testing function
def test_music_retriever():
    """Test music retriever functionality"""
    logger.info("Testing Antashiri Music Retriever...")
    
    # Initialize retriever
    retriever = MusicRetriever()
    
    # Test contexts
    test_contexts = [
        {
            'keywords': ['happy', 'celebrate', 'party'],
            'history': [{'text': 'Just got promoted!'}]
        },
        {
            'keywords': ['sad', 'lonely', 'missing'],
            'history': [{'text': 'Missing my family'}]
        },
        {
            'keywords': ['workout', 'gym', 'energy'],
            'history': [{'text': 'Time for morning workout'}]
        }
    ]
    
    emotions = ['happy', 'sad', 'energetic']
    
    for context, emotion in zip(test_contexts, emotions):
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing emotion: {emotion}")
        logger.info(f"Context: {context['keywords']}")
        
        # Retrieve songs
        songs = retriever.retrieve(context, emotion, top_k=3)
        
        for i, song in enumerate(songs, 1):
            logger.info(f"\n{i}. {song['title']} - {song['artist']}")
            logger.info(f"   Genre: {song['genre']}, Emotion: {song['emotion']}")
            logger.info(f"   Score: {song['score']:.3f}")
            logger.info(f"   Reason: {song['reason']}")
    
    logger.success("Music retriever tests completed!")


if __name__ == "__main__":
    test_music_retriever()