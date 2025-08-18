# core/youtube_player.py
"""
YouTube Integration Module for Antashiri System
Handles YouTube search and playback control
"""

import os
import re
import time
from typing import Optional, List, Dict, Tuple
from urllib.parse import quote_plus
import threading
import queue

from pytube import YouTube, Search
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger


class YouTubePlayer:
    """
    YouTube integration for music playback
    Supports both YouTube API and pytube fallback
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize YouTube player
        
        Args:
            api_key: YouTube Data API key (optional, uses pytube if not provided)
        """
        self.api_key = api_key
        self.youtube_service = None
        
        # Initialize YouTube API if key provided
        if self.api_key:
            try:
                self.youtube_service = build('youtube', 'v3', developerKey=self.api_key)
                logger.info("YouTube API initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize YouTube API: {e}")
                logger.info("Falling back to pytube")
        
        # Playback state
        self.current_video = None
        self.is_playing = False
        self.playback_queue = queue.Queue()
        self.volume = 0.7  # Default volume (0-1)
        
        # Cache for search results
        self.search_cache = {}
        self.cache_size = 100
        
        logger.info("YouTubePlayer initialized")
    
    def search_and_get_url(self, query: str, duration_preference: Optional[str] = None) -> Optional[str]:
        """
        Search YouTube and get video URL
        
        Args:
            query: Search query (e.g., "song title artist")
            duration_preference: Preferred duration ('short', 'medium', 'long')
            
        Returns:
            YouTube video URL or None if not found
        """
        try:
            # Check cache first
            cache_key = f"{query}_{duration_preference}"
            if cache_key in self.search_cache:
                logger.debug(f"Using cached result for: {query}")
                return self.search_cache[cache_key]
            
            # Try YouTube API first
            if self.youtube_service:
                url = self._search_with_api(query, duration_preference)
            else:
                # Fallback to pytube
                url = self._search_with_pytube(query)
            
            # Cache result
            if url and len(self.search_cache) < self.cache_size:
                self.search_cache[cache_key] = url
            
            return url
            
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return None
    
    def _search_with_api(self, query: str, duration_preference: Optional[str] = None) -> Optional[str]:
        """Search using YouTube Data API"""
        try:
            # Build search request
            search_request = self.youtube_service.search().list(
                part='id,snippet',
                q=query,
                type='video',
                maxResults=10,
                videoCategoryId='10',  # Music category
                order='relevance'
            )
            
            # Add duration filter if specified
            if duration_preference:
                duration_map = {
                    'short': 'short',      # < 4 minutes
                    'medium': 'medium',    # 4-20 minutes
                    'long': 'long'        # > 20 minutes
                }
                if duration_preference in duration_map:
                    search_request = search_request.list(
                        videoDuration=duration_map[duration_preference]
                    )
            
            # Execute search
            search_response = search_request.execute()
            
            # Get first valid result
            for item in search_response.get('items', []):
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                
                # Filter out non-music content
                if self._is_music_video(title):
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    logger.info(f"Found: {title}")
                    return url
            
            logger.warning(f"No suitable music video found for: {query}")
            return None
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return None
        except Exception as e:
            logger.error(f"API search error: {e}")
            return None
    
    def _search_with_pytube(self, query: str) -> Optional[str]:
        """Search using pytube (no API key required)"""
        try:
            logger.debug(f"Searching with pytube: {query}")
            
            # Perform search
            search = Search(query)
            results = search.results[:10]  # Get first 10 results
            
            # Find first music video
            for video in results:
                if self._is_music_video(video.title):
                    logger.info(f"Found: {video.title}")
                    return video.watch_url
            
            # If no music video found, return first result
            if results:
                logger.info(f"Using first result: {results[0].title}")
                return results[0].watch_url
            
            logger.warning(f"No results found for: {query}")
            return None
            
        except Exception as e:
            logger.error(f"Pytube search error: {e}")
            return None
    
    def _is_music_video(self, title: str) -> bool:
        """Check if video title suggests it's a music video"""
        title_lower = title.lower()
        
        # Exclude non-music content
        exclude_keywords = [
            'reaction', 'review', 'tutorial', 'cover', 'karaoke',
            'lyrics', 'hour', 'hours', 'mix', 'compilation',
            'playlist', 'full album', 'interview', 'behind the scenes'
        ]
        
        for keyword in exclude_keywords:
            if keyword in title_lower:
                return False
        
        # Prefer official content
        prefer_keywords = [
            'official', 'music video', 'audio', 'vevo', 'records'
        ]
        
        for keyword in prefer_keywords:
            if keyword in title_lower:
                return True
        
        # Default to accepting
        return True
    
    def get_video_info(self, url: str) -> Optional[Dict]:
        """
        Get video information
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video information dictionary or None
        """
        try:
            yt = YouTube(url)
            
            info = {
                'title': yt.title,
                'author': yt.author,
                'length': yt.length,  # Duration in seconds
                'views': yt.views,
                'rating': yt.rating,
                'thumbnail': yt.thumbnail_url,
                'url': url
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None
    
    def get_audio_stream_url(self, video_url: str) -> Optional[str]:
        """
        Get audio stream URL for direct playback
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Audio stream URL or None
        """
        try:
            yt = YouTube(video_url)
            
            # Get audio stream (prefer high quality)
            audio_streams = yt.streams.filter(only_audio=True).order_by('abr').desc()
            
            if audio_streams:
                stream = audio_streams.first()
                stream_url = stream.url
                logger.info(f"Got audio stream: {stream.abr}")
                return stream_url
            
            logger.warning("No audio stream found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get audio stream: {e}")
            return None
    
    def search_multiple(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for multiple videos
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of video information dictionaries
        """
        results = []
        
        try:
            if self.youtube_service:
                # Use API
                search_response = self.youtube_service.search().list(
                    part='id,snippet',
                    q=query,
                    type='video',
                    maxResults=limit,
                    videoCategoryId='10'
                ).execute()
                
                for item in search_response.get('items', []):
                    results.append({
                        'title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle'],
                        'video_id': item['id']['videoId'],
                        'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        'thumbnail': item['snippet']['thumbnails']['default']['url']
                    })
            else:
                # Use pytube
                search = Search(query)
                for video in search.results[:limit]:
                    results.append({
                        'title': video.title,
                        'channel': video.author,
                        'video_id': video.video_id,
                        'url': video.watch_url,
                        'thumbnail': video.thumbnail_url
                    })
            
        except Exception as e:
            logger.error(f"Multiple search failed: {e}")
        
        return results
    
    def create_playlist_from_emotion(self, emotion: str, count: int = 10) -> List[str]:
        """
        Create a playlist based on emotion
        
        Args:
            emotion: Target emotion
            count: Number of songs
            
        Returns:
            List of YouTube URLs
        """
        # Emotion-based search queries
        emotion_queries = {
            'happy': [
                'happy upbeat songs', 'feel good music', 'positive vibes playlist',
                'cheerful songs', 'celebration music'
            ],
            'sad': [
                'sad emotional songs', 'melancholy music', 'heartbreak songs',
                'emotional ballads', 'crying songs'
            ],
            'energetic': [
                'workout motivation music', 'high energy songs', 'pump up music',
                'gym playlist', 'running songs'
            ],
            'calm': [
                'relaxing music', 'calm meditation', 'peaceful songs',
                'ambient relaxation', 'stress relief music'
            ],
            'romantic': [
                'love songs', 'romantic music', 'couple songs',
                'wedding music', 'valentine playlist'
            ],
            'stressed': [
                'stress relief music', 'anxiety calming songs', 'relaxation music',
                'peaceful meditation', 'calm down playlist'
            ]
        }
        
        playlist = []
        queries = emotion_queries.get(emotion, ['music'])
        
        for query in queries[:count//2]:
            results = self.search_multiple(query, limit=2)
            for result in results:
                if result['url'] not in playlist:
                    playlist.append(result['url'])
                    if len(playlist) >= count:
                        break
            if len(playlist) >= count:
                break
        
        logger.info(f"Created {emotion} playlist with {len(playlist)} songs")
        return playlist
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if URL is a valid YouTube video
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid YouTube URL
        """
        youtube_regex = re.compile(
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
            r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        
        return bool(youtube_regex.match(url))
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None
        """
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def get_embed_url(self, video_url: str) -> Optional[str]:
        """
        Get embed URL for iframe embedding
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Embed URL or None
        """
        video_id = self.extract_video_id(video_url)
        if video_id:
            return f"https://www.youtube.com/embed/{video_id}?autoplay=1"
        return None


class YouTubePlaybackController:
    """
    Advanced playback controller with queue management
    """
    
    def __init__(self, player: YouTubePlayer):
        """
        Initialize playback controller
        
        Args:
            player: YouTubePlayer instance
        """
        self.player = player
        self.play_queue = []
        self.current_index = 0
        self.is_playing = False
        self.play_duration = 45  # Default play duration in seconds
        self.fade_duration = 3  # Fade in/out duration
        
        # Playback thread
        self.playback_thread = None
        self.stop_event = threading.Event()
        
        logger.info("YouTubePlaybackController initialized")
    
    def add_to_queue(self, url: str, duration: Optional[int] = None):
        """Add song to playback queue"""
        self.play_queue.append({
            'url': url,
            'duration': duration or self.play_duration
        })
        logger.info(f"Added to queue: {url}")
    
    def play_next(self):
        """Play next song in queue"""
        if self.current_index < len(self.play_queue):
            current = self.play_queue[self.current_index]
            self.play(current['url'], current['duration'])
            self.current_index += 1
        else:
            logger.info("Queue finished")
            self.is_playing = False
    
    def play(self, url: str, duration: int):
        """
        Play a song for specified duration
        
        Args:
            url: YouTube video URL
            duration: Playback duration in seconds
        """
        logger.info(f"Playing: {url} for {duration} seconds")
        self.is_playing = True
        
        # In a real implementation, this would control actual audio playback
        # For web interface, we send the URL to frontend for iframe embedding
        
        # Simulate playback
        time.sleep(duration)
        
        logger.info("Playback finished")
        self.is_playing = False
    
    def stop(self):
        """Stop current playback"""
        self.stop_event.set()
        self.is_playing = False
        logger.info("Playback stopped")
    
    def clear_queue(self):
        """Clear playback queue"""
        self.play_queue.clear()
        self.current_index = 0
        logger.info("Queue cleared")


# Testing function
def test_youtube_player():
    """Test YouTube player functionality"""
    logger.info("Testing YouTube Player...")
    
    # Initialize player (without API key for testing)
    player = YouTubePlayer()
    
    # Test searches
    test_queries = [
        "Shape of You Ed Sheeran",
        "Bohemian Rhapsody Queen",
        "relaxing piano music"
    ]
    
    for query in test_queries:
        logger.info(f"\nSearching: {query}")
        url = player.search_and_get_url(query)
        
        if url:
            logger.success(f"Found: {url}")
            
            # Get video info
            info = player.get_video_info(url)
            if info:
                logger.info(f"Title: {info['title']}")
                logger.info(f"Author: {info['author']}")
                logger.info(f"Duration: {info['length']} seconds")
        else:
            logger.warning(f"No result for: {query}")
    
    # Test playlist creation
    logger.info("\nCreating happy playlist...")
    playlist = player.create_playlist_from_emotion('happy', count=5)
    for i, url in enumerate(playlist, 1):
        logger.info(f"{i}. {url}")
    
    logger.success("YouTube Player test completed!")


if __name__ == "__main__":
    test_youtube_player()