import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# YouTube Data API Configuration
YOUTUBE_API_KEY = "AIzaSyCJfH4i7N0XXUDmz8ojz0zr0eWc1LOzRMs"
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"

class YouTubeAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = YOUTUBE_API_BASE_URL
    
    def extract_channel_id(self, input_str):
        """Extract channel ID from various YouTube URL formats or handle"""
        # Remove any whitespace
        input_str = input_str.strip()
        
        # If it's already a channel ID (starts with UC)
        if input_str.startswith('UC') and len(input_str) == 24:
            return input_str
        
        # Extract from various URL formats
        patterns = [
            r'youtube\.com/channel/([^/?&]+)',
            r'youtube\.com/c/([^/?&]+)',
            r'youtube\.com/user/([^/?&]+)',
            r'youtube\.com/@([^/?&]+)',
            r'youtube\.com/([^/?&]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, input_str)
            if match:
                username_or_id = match.group(1)
                if username_or_id.startswith('UC'):
                    return username_or_id
                else:
                    # Convert username to channel ID
                    return self.get_channel_id_by_username(username_or_id)
        
        # If no URL pattern matches, treat as username
        return self.get_channel_id_by_username(input_str)
    
    def get_channel_id_by_username(self, username):
        """Convert username to channel ID"""
        url = f"{self.base_url}/channels"
        params = {
            'part': 'id',
            'forUsername': username,
            'key': self.api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'items' in data and len(data['items']) > 0:
            return data['items'][0]['id']
        
        # Try search if direct username lookup fails
        search_url = f"{self.base_url}/search"
        search_params = {
            'part': 'snippet',
            'q': username,
            'type': 'channel',
            'maxResults': 1,
            'key': self.api_key
        }
        
        search_response = requests.get(search_url, params=search_params)
        search_data = search_response.json()
        
        if 'items' in search_data and len(search_data['items']) > 0:
            return search_data['items'][0]['snippet']['channelId']
        
        return None
    
    def get_channel_info(self, channel_id):
        """Get basic channel information"""
        url = f"{self.base_url}/channels"
        params = {
            'part': 'snippet,statistics,contentDetails,brandingSettings',
            'id': channel_id,
            'key': self.api_key
        }
        
        response = requests.get(url, params=params)
        return response.json()
    
    def get_recent_videos(self, channel_id, max_results=50):
        """Get recent videos from a channel"""
        # First get the uploads playlist ID
        channel_info = self.get_channel_info(channel_id)
        if 'items' not in channel_info or len(channel_info['items']) == 0:
            return []
        
        uploads_playlist_id = channel_info['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Get videos from uploads playlist
        url = f"{self.base_url}/playlistItems"
        params = {
            'part': 'snippet,contentDetails',
            'playlistId': uploads_playlist_id,
            'maxResults': max_results,
            'key': self.api_key
        }
        
        response = requests.get(url, params=params)
        return response.json()
    
    def get_video_statistics(self, video_ids):
        """Get detailed statistics for videos"""
        # YouTube API allows up to 50 video IDs per request
        video_stats = []
        
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            url = f"{self.base_url}/videos"
            params = {
                'part': 'statistics,contentDetails,snippet',
                'id': ','.join(batch_ids),
                'key': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            if 'items' in data:
                video_stats.extend(data['items'])
        
        return video_stats
    
    def get_video_comments(self, video_id, max_results=100):
        """Get comments for sentiment analysis"""
        url = f"{self.base_url}/commentThreads"
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'maxResults': max_results,
            'order': 'relevance',
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            return data.get('items', [])
        except:
            return []
    
    def analyze_content_niche(self, videos_data, channel_info):
        """Analyze and determine the creator's niche based on video titles, descriptions, and tags"""
        # Collect all text data
        all_text = []
        video_tags = []
        
        # Add channel description
        channel_desc = channel_info.get('snippet', {}).get('description', '')
        if channel_desc:
            all_text.append(channel_desc)
        
        # Collect video titles and descriptions
        for video in videos_data[:20]:  # Analyze recent 20 videos
            title = video.get('snippet', {}).get('title', '')
            description = video.get('snippet', {}).get('description', '')
            tags = video.get('snippet', {}).get('tags', [])
            
            all_text.extend([title, description])
            video_tags.extend(tags)
        
        # Common niche keywords
        niche_keywords = {
            'Gaming': ['game', 'gaming', 'gameplay', 'streamer', 'twitch', 'xbox', 'playstation', 'pc', 'mobile', 'esports'],
            'Technology': ['tech', 'technology', 'review', 'unboxing', 'smartphone', 'laptop', 'gadget', 'software', 'app'],
            'Entertainment': ['funny', 'comedy', 'entertainment', 'fun', 'laugh', 'joke', 'prank', 'viral', 'trending'],
            'Education': ['tutorial', 'learn', 'education', 'teach', 'how to', 'guide', 'tips', 'study', 'school'],
            'Lifestyle': ['vlog', 'daily', 'lifestyle', 'routine', 'life', 'personal', 'family', 'travel', 'fashion'],  
            'Music': ['music', 'song', 'cover', 'singing', 'instrument', 'band', 'album', 'concert', 'lyrics'],
            'Fitness': ['workout', 'fitness', 'gym', 'exercise', 'health', 'diet', 'nutrition', 'bodybuilding'],
            'Cooking': ['cooking', 'recipe', 'food', 'kitchen', 'chef', 'baking', 'meal', 'restaurant'],
            'Business': ['business', 'entrepreneur', 'startup', 'marketing', 'money', 'finance', 'investment'],
            'Science': ['science', 'physics', 'chemistry', 'biology', 'research', 'experiment', 'discovery']
        }
        
        # Count keyword occurrences
        niche_scores = {}
        combined_text = ' '.join(all_text + video_tags).lower()
        
        for niche, keywords in niche_keywords.items():
            score = sum(combined_text.count(keyword) for keyword in keywords)
            niche_scores[niche] = score
        
        # Determine primary niche
        if niche_scores:
            primary_niche = max(niche_scores, key=niche_scores.get)
            confidence = niche_scores[primary_niche] / sum(niche_scores.values()) if sum(niche_scores.values()) > 0 else 0
            
            # Get top 3 niches
            sorted_niches = sorted(niche_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_niches = [niche for niche, score in sorted_niches if score > 0]
            
            return {
                'primary_niche': primary_niche if confidence > 0.1 else 'General Content',
                'confidence': confidence,
                'top_niches': top_niches,
                'niche_scores': niche_scores
            }
        
        return {
            'primary_niche': 'General Content',
            'confidence': 0,
            'top_niches': [],
            'niche_scores': {}
        }
    
    def generate_creator_description(self, channel_info, videos_data, niche_analysis):
        """Generate a comprehensive description of the creator"""
        channel_data = channel_info['snippet']
        stats = channel_info['statistics']
        
        # Basic info
        title = channel_data['title']
        subscriber_count = int(stats.get('subscriberCount', 0))
        video_count = int(stats.get('videoCount', 0))
        
        # Format subscriber count
        if subscriber_count >= 1000000:
            sub_text = f"{subscriber_count/1000000:.1f}M subscribers"
        elif subscriber_count >= 1000:
            sub_text = f"{subscriber_count/1000:.1f}K subscribers"
        else:
            sub_text = f"{subscriber_count} subscribers"
        
        # Analyze upload frequency
        upload_frequency = self.analyze_upload_frequency(videos_data)
        
        # Generate description
        description = f"**{title}** is a {niche_analysis['primary_niche']} content creator with {sub_text} and {video_count:,} videos. "
        
        if niche_analysis['confidence'] > 0.3:
            description += f"They primarily focus on {niche_analysis['primary_niche'].lower()} content"
            if len(niche_analysis['top_niches']) > 1:
                other_niches = [n for n in niche_analysis['top_niches'][1:3] if n != niche_analysis['primary_niche']]
                if other_niches:
                    description += f", with additional content in {', '.join(other_niches).lower()}"
            description += ". "
        
        description += f"The channel uploads {upload_frequency}. "
        
        # Add channel description if available
        channel_desc = channel_data.get('description', '')
        if channel_desc and len(channel_desc) > 50:
            # Extract first meaningful sentence
            sentences = channel_desc.split('.')
            if sentences and len(sentences[0]) > 20:
                description += f"According to their channel description: \"{sentences[0].strip()}.\""
        
        return description
    
    def analyze_upload_frequency(self, videos_data):
        """Analyze how frequently the creator uploads"""
        if len(videos_data) < 2:
            return "irregularly"
        
        upload_dates = []
        for video in videos_data[:10]:  # Check last 10 videos
            try:
                pub_date = datetime.strptime(
                    video['snippet']['publishedAt'], 
                    '%Y-%m-%dT%H:%M:%SZ'
                )
                upload_dates.append(pub_date)
            except:
                continue
        
        if len(upload_dates) < 2:
            return "irregularly"
        
        # Calculate average days between uploads
        upload_dates.sort(reverse=True)
        intervals = []
        for i in range(len(upload_dates) - 1):
            diff = (upload_dates[i] - upload_dates[i + 1]).days
            intervals.append(diff)
        
        if intervals:
            avg_interval = np.mean(intervals)
            if avg_interval <= 1:
                return "daily"
            elif avg_interval <= 3:
                return "multiple times per week"
            elif avg_interval <= 7:
                return "weekly"
            elif avg_interval <= 14:
                return "bi-weekly"
            elif avg_interval <= 30:
                return "monthly"
            else:
                return "irregularly"
        
        return "irregularly"

class YCCVCalculator:
    def __init__(self):
        self.weights = {
            'content_engagement': 0.35,
            'audience_growth': 0.25,
            'community_health': 0.20,
            'content_strategy': 0.20,
        }
    
    def normalize_metric(self, current_value, min_value, max_value):
        """Normalize metric to 0-100 scale"""
        if max_value == min_value:
            return 50  # Default middle value if no variation
        return ((current_value - min_value) / (max_value - min_value)) * 100
    
    def analyze_sentiment(self, comments):
        """Analyze sentiment of comments using TextBlob"""
        if not comments:
            return 0, {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
        
        sentiments = []
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for comment in comments:
            try:
                text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                sentiments.append(polarity)
                
                if polarity > 0.1:
                    sentiment_counts['positive'] += 1
                elif polarity < -0.1:
                    sentiment_counts['negative'] += 1
                else:
                    sentiment_counts['neutral'] += 1
            except:
                continue
        
        if not sentiments:
            return 0, {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
        
        # Calculate Net Sentiment Score
        total = len(sentiments)
        net_sentiment = ((sentiment_counts['positive'] - sentiment_counts['negative']) / total) * 100
        
        sentiment_counts['total'] = total
        return net_sentiment, sentiment_counts
        
    
    def calculate_engagement_score(self, videos_data):
        """Calculate Content Engagement & Quality score with detailed explanation"""
        if not videos_data:
            return 0, {}
        
        engagement_metrics = {
            'total_videos': len(videos_data),
            'total_views': 0,
            'total_likes': 0,
            'total_comments': 0,
            'engagement_rates': [],
            'avg_engagement_rate': 0
        }
        
        engagement_scores = []
        
        for video in videos_data:
            stats = video.get('statistics', {})
            
            # Get basic metrics
            views = int(stats.get('viewCount', 0))
            likes = int(stats.get('likeCount', 0))
            comments = int(stats.get('commentCount', 0))
            
            engagement_metrics['total_views'] += views
            engagement_metrics['total_likes'] += likes
            engagement_metrics['total_comments'] += comments
            
            if views == 0:
                continue
            
            # Calculate engagement actions per 1000 views
            # Using weights: comments x2 (more valuable interaction)
            engagement_rate = ((likes + (comments * 2)) / views) * 1000
            engagement_scores.append(engagement_rate)
            engagement_metrics['engagement_rates'].append(engagement_rate)
        
        avg_engagement = np.mean(engagement_scores) if engagement_scores else 0
        engagement_metrics['avg_engagement_rate'] = avg_engagement
        
        # Calculate percentile score (normalize to 0-100)
        # Industry benchmark: 10-30 per 1000 views is good
        normalized_score = min((avg_engagement  * 2.5), 100)
        
        return normalized_score, engagement_metrics
    
    def calculate_growth_metrics(self, channel_stats, recent_videos):
        """Calculate audience growth metrics with detailed explanation"""
        subscriber_count = int(channel_stats.get('subscriberCount', 0))
        view_count = int(channel_stats.get('viewCount', 0))
        video_count = int(channel_stats.get('videoCount', 0))
        
        growth_metrics = {
            'subscriber_count': subscriber_count,
            'total_views': view_count,
            'video_count': video_count,
            'avg_views_per_video': view_count / max(video_count, 1),
            'recent_performance': {}
        }
        
        # Estimate growth based on recent video performance
        if recent_videos:
            recent_views = []
            for v in recent_videos:
                views = int(v.get('statistics', {}).get('viewCount', 0))
                recent_views.append(views)
            
            avg_recent_views = np.mean(recent_views) if recent_views else 0
            growth_metrics['recent_performance'] = {
                'avg_recent_views': avg_recent_views,
                'recent_videos_count': len(recent_views)
            }
            
            # Estimate CTR and growth based on performance relative to subscriber base
            estimated_ctr = min((avg_recent_views / max(subscriber_count, 1)) * 100, 100)
            growth_indicator = min(avg_recent_views / max(growth_metrics['avg_views_per_video'], 1) * 50, 100)
            
            final_score = (estimated_ctr + growth_indicator) / 2
            growth_metrics['estimated_ctr'] = estimated_ctr
            growth_metrics['growth_indicator'] = growth_indicator
            
            return final_score, growth_metrics
        
        return 0, growth_metrics
    
    def calculate_community_health(self, videos_data, analyzer):
        """Calculate community health score with detailed explanation"""
        if not videos_data:
            return 0, {}
        
        community_metrics = {
            'videos_analyzed': 0,
            'total_comments_analyzed': 0,
            'sentiment_breakdown': {},
            'avg_sentiment': 0,
            'sentiment_scores': []
        }
        
        sentiment_scores = []
        all_sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
        
        # Analyze sentiment for recent videos
        for video in videos_data[:5]:  # Check top 5 recent videos
            video_id = video['id']
            comments = analyzer.get_video_comments(video_id, 50)
            
            if comments:
                sentiment, sentiment_counts = self.analyze_sentiment(comments)
                sentiment_scores.append(sentiment)
                community_metrics['videos_analyzed'] += 1
                community_metrics['total_comments_analyzed'] += sentiment_counts['total']
                
                # Aggregate sentiment counts
                for key in ['positive', 'neutral', 'negative', 'total']:
                    all_sentiment_counts[key] += sentiment_counts[key]
        
        community_metrics['sentiment_breakdown'] = all_sentiment_counts
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        community_metrics['avg_sentiment'] = avg_sentiment
        community_metrics['sentiment_scores'] = sentiment_scores
        
        # Normalize sentiment score to 0-100 (shift to positive range)
        normalized_score = max(0, avg_sentiment + 50)
        
        return normalized_score, community_metrics
    
    def calculate_content_strategy(self, videos_data):
        """Calculate content strategy and consistency score with detailed explanation"""
        if not videos_data:
            return 0, {}
        
        strategy_metrics = {
            'total_videos': len(videos_data),
            'upload_intervals': [],
            'avg_interval_days': 0,
            'consistency_rating': '',
            'upload_pattern': {}
        }
        
        # Analyze upload frequency
        upload_dates = []
        for video in videos_data:
            try:
                pub_date = datetime.strptime(
                    video['snippet']['publishedAt'], 
                    '%Y-%m-%dT%H:%M:%SZ'
                )
                upload_dates.append(pub_date)
            except:
                continue
        
        if len(upload_dates) < 2:
            return 0, strategy_metrics
        
        # Calculate average days between uploads
        upload_dates.sort(reverse=True)
        intervals = []
        for i in range(len(upload_dates) - 1):
            diff = (upload_dates[i] - upload_dates[i + 1]).days
            intervals.append(diff)
        
        strategy_metrics['upload_intervals'] = intervals
        
        if intervals:
            avg_interval = np.mean(intervals)
            strategy_metrics['avg_interval_days'] = avg_interval
            
            # Determine consistency rating and score
            if avg_interval <= 1:
                consistency_rating = "Daily"
                consistency_score = 100
            elif avg_interval <= 3:
                consistency_rating = "Multiple times per week"
                consistency_score = 90
            elif avg_interval <= 7:
                consistency_rating = "Weekly"
                consistency_score = 85
            elif avg_interval <= 14:
                consistency_rating = "Bi-weekly"
                consistency_score = 70
            elif avg_interval <= 30:
                consistency_rating = "Monthly"
                consistency_score = 50
            else:
                consistency_rating = "Irregular"
                consistency_score = max(0, 100 - (avg_interval - 30) * 2)
            
            strategy_metrics['consistency_rating'] = consistency_rating
            
            # Adjust for consistency (penalize high variation)
            interval_std = np.std(intervals) if len(intervals) > 1 else 0
            consistency_penalty = min(interval_std / avg_interval * 20, 30) if avg_interval > 0 else 0
            final_score = max(0, consistency_score - consistency_penalty)
            
            strategy_metrics['consistency_penalty'] = consistency_penalty
            
            return final_score, strategy_metrics
        
        return 0, strategy_metrics
    
    def calculate_risk_factors(self, videos_data, channel_stats,analyzer):
        # """Calculate risk factors using the new methodology from the images"""
        # risk_metrics = {
        #     'gvi': 0,  # Growth Volatility Index
        #     'svs': 0,  # Sentiment Variability Score
        #     'edi': 0,  # Engagement Decay Index
        #     'ccr': 0,  # Content Consistency Risk
        #     'vdi': 0,  # Virality Dependence Index
        #     'details': {}
        # }
        
        # # 1. Growth Volatility Index (GVI)
        # if len(videos_data) > 1:
        #     view_counts = [int(v.get('statistics', {}).get('viewCount', 0)) for v in videos_data]
        #     daily_pct_changes = np.diff(view_counts) / view_counts[:-1] * 100
        #     if len(daily_pct_changes) > 0:
        #         gvi = np.std(daily_pct_changes)
        #         risk_metrics['gvi'] = gvi
        #         risk_metrics['details']['gvi'] = {
        #             'daily_pct_changes': daily_pct_changes,
        #             'std_dev': gvi
        #         }
        
        # # 2. Sentiment Variability Score (SVS)
        # if len(videos_data) > 0:
        #     sentiment_scores = []
        #     for video in videos_data[:5]:  # Check top 5 recent videos
        #         video_id = video['id']
        #         comments = analyzer.get_video_comments(video_id, 50)
        #         if comments:
        #             sentiment, _ = self.analyze_sentiment(comments)
        #             sentiment_scores.append(sentiment)
            
        #     if len(sentiment_scores) > 1:
        #         svs = np.std(sentiment_scores)
        #         risk_metrics['svs'] = svs
        #         risk_metrics['details']['svs'] = {
        #             'sentiment_scores': sentiment_scores,
        #             'std_dev': svs
        #         }
        
        # # 3. Engagement Decay Index (EDI)
        # if len(videos_data) > 1:
        #     recent_engagement = []
        #     past_engagement = []
            
        #     # Split videos into recent and past (50/50)
        #     split_point = len(videos_data) // 2
        #     recent_videos = videos_data[:split_point]
        #     past_videos = videos_data[split_point:]
            
        #     # Calculate average engagement for each period
        #     for video in recent_videos:
        #         stats = video.get('statistics', {})
        #         views = int(stats.get('viewCount', 0))
        #         likes = int(stats.get('likeCount', 0))
        #         comments = int(stats.get('commentCount', 0))
        #         if views > 0:
        #             recent_engagement.append((likes + comments) / views)
            
        #     for video in past_videos:
        #         stats = video.get('statistics', {})
        #         views = int(stats.get('viewCount', 0))
        #         likes = int(stats.get('likeCount', 0))
        #         comments = int(stats.get('commentCount', 0))
        #         if views > 0:
        #             past_engagement.append((likes + comments) / views)
            
        #     if recent_engagement and past_engagement:
        #         avg_recent = np.mean(recent_engagement)
        #         avg_past = np.mean(past_engagement)
        #         if avg_past > 0:
        #             edi = np.log(avg_recent / avg_past)
        #             risk_metrics['edi'] = edi
        #             risk_metrics['details']['edi'] = {
        #                 'recent_engagement': avg_recent,
        #                 'past_engagement': avg_past,
        #                 'log_ratio': edi
        #             }
        
        # # 4. Content Consistency Risk (CCR)
        # if len(videos_data) > 1:
        #     upload_dates = []
        #     for video in videos_data:
        #         try:
        #             pub_date = datetime.strptime(
        #                 video['snippet']['publishedAt'], 
        #                 '%Y-%m-%dT%H:%M:%SZ'
        #             )
        #             upload_dates.append(pub_date)
        #         except:
        #             continue
            
        #     if len(upload_dates) > 1:
        #         upload_dates.sort()
        #         intervals = np.diff([d.timestamp() for d in upload_dates]) / (60*60*24)  # Convert to days
        #         interval_std = np.std(intervals) if len(intervals) > 1 else 0
                
        #         # Simple topic drift penalty (count unique words in titles)
        #         titles = [v['snippet']['title'] for v in videos_data[:10]]
        #         unique_words = len(set(" ".join(titles).lower().split()))
        #         total_words = len(" ".join(titles).split())
        #         topic_drift_penalty = (unique_words / max(total_words, 1)) * 10
                
        #         ccr = interval_std + topic_drift_penalty
        #         risk_metrics['ccr'] = ccr
        #         risk_metrics['details']['ccr'] = {
        #             'interval_std': interval_std,
        #             'topic_drift_penalty': topic_drift_penalty,
        #             'unique_words': unique_words,
        #             'total_words': total_words
        #         }
        
        # # 5. Virality Dependence Index (VDI)
        # if len(videos_data) > 2:
        #     view_counts = [int(v.get('statistics', {}).get('viewCount', 0)) for v in videos_data]
        #     max_3 = np.mean(sorted(view_counts, reverse=True)[:3])
        #     median = np.median(view_counts)
        #     if median > 0:
        #         vdi = max_3 / median
        #         risk_metrics['vdi'] = vdi
        #         risk_metrics['details']['vdi'] = {
        #             'max_3_videos': max_3,
        #             'median_views': median,
        #             'ratio': vdi
        #         }
        
        # # Normalize all risk metrics to [0,1] range
        # normalized_metrics = {}
        # for key in ['gvi', 'svs', 'edi', 'ccr', 'vdi']:
        #     normalized_metrics[key] = min(max(risk_metrics[key], 0), 1)
        
        # # Calculate final risk score with equal weights
        # weights = {
        #     'gvi': 0.2,
        #     'svs': 0.2,
        #     'edi': 0.2,
        #     'ccr': 0.2,
        #     'vdi': 0.2
        # }
        
        # risk_score = (
        #     normalized_metrics['gvi'] * weights['gvi'] +
        #     normalized_metrics['svs'] * weights['svs'] +
        #     normalized_metrics['edi'] * weights['edi'] +
        #     normalized_metrics['ccr'] * weights['ccr'] +
        #     normalized_metrics['vdi'] * weights['vdi']
        # )
        
        # # Convert risk score to multiplier (higher risk = lower multiplier)
        # risk_multiplier = max(0.5, 1 - risk_score)  # Minimum 50% of original score
        
        # return risk_multiplier, {
        #     'risk_score': risk_score,
        #     'normalized_metrics': normalized_metrics,
        #     'weights': weights,
        #     'details': risk_metrics['details']
        # }

        """Calculate risk factors with improved metrics and normalization"""
        # Initialize with default values
        risk_metrics = {
            'gvi': {'value': 0, 'normalized': 0.5},  # Default to medium risk
            'svs': {'value': 0, 'normalized': 0.5},
            'edi': {'value': 0, 'normalized': 0.5},
            'ccr': {'value': 0, 'normalized': 0.5},
            'vdi': {'value': 0, 'normalized': 0.5},
            'details': {}
        }

        try:
             # 1. Advanced Growth Volatility Index (GVI)
            if len(videos_data) > 1:
                try:
                    # Get clean view counts data
                    view_counts = []
                    for v in videos_data:
                        count = int(v.get('statistics', {}).get('viewCount', 0))
                        if count > 0:  # Only positive view counts
                            view_counts.append(count)
                    
                    if len(view_counts) >= 5:  # Minimum 5 videos for volatility calculation
                        # Calculate robust percentage changes
                        pct_changes = []
                        for i in range(1, len(view_counts)):
                            prev = view_counts[i-1]
                            curr = view_counts[i]
                            if prev > 0:
                                change = (curr - prev) / prev
                                pct_changes.append(change)
                        
                        if len(pct_changes) >= 3:  # Need at least 3 changes
                            # Calculate IQR-based volatility (more robust than std dev)
                            q75, q25 = np.percentile(pct_changes, [75, 25])
                            iqr = q75 - q25
                            
                            # Empirical scaling based on actual YouTube data:
                            # - IQR < 0.2 (20%): Low volatility → 0-0.3
                            # - IQR 0.2-0.5: Medium → 0.3-0.7
                            # - IQR > 0.5: High → 0.7-1.0
                            if iqr < 0.2:
                                normalized_gvi = 0.3 * (iqr / 0.2)
                            elif iqr < 0.5:
                                normalized_gvi = 0.3 + 0.4 * ((iqr - 0.2) / 0.3)
                            else:
                                normalized_gvi = 0.7 + 0.3 * min((iqr - 0.5) / 0.5, 1)
                            
                            # Adjust for channel size (smaller channels can be more volatile)
                            avg_views = np.mean(view_counts)
                            size_factor = min(1.0, max(0.7, 1 - (avg_views / 100000)))
                            normalized_gvi *= size_factor
                            
                            # Final clamping
                            normalized_gvi = min(max(normalized_gvi, 0), 1)
                            
                            risk_metrics['gvi'].update({
                                'value': iqr * 100,  # Store as percentage
                                'normalized': normalized_gvi,
                                'valid': True
                            })
                            
                            risk_metrics['details']['gvi'] = {
                                'view_counts': view_counts,
                                'pct_changes': pct_changes,
                                'iqr': iqr,
                                'size_factor': size_factor,
                                'normalization_curve': [
                                    (0.0, 0.0),
                                    (0.2, 0.3),
                                    (0.5, 0.7),
                                    (1.0, 1.0)
                                ]
                            }
                            
                except Exception as e:
                    print(f"GVI calculation error: {str(e)}")

            # Fallback to medium risk if calculation failed
            if not risk_metrics['gvi']['valid']:
                risk_metrics['gvi']['normalized'] = 0.5
                risk_metrics['details']['gvi'] = {
                    'info': 'Using default medium risk (0.5)',
                    'reason': 'Insufficient valid data points' if len(videos_data) < 5 else 'Calculation error'
                }

            # 2. Sentiment Variability Score (SVS)
            if len(videos_data) > 0:
                sentiment_scores = []
                for video in videos_data[:5]:
                    try:
                        video_id = video['id']
                        comments = analyzer.get_video_comments(video_id, 50)
                        if comments:
                            sentiment, _ = self.analyze_sentiment(comments)
                            sentiment_scores.append(sentiment)
                    except:
                        continue
                
                if len(sentiment_scores) > 1:
                    svs = np.std(sentiment_scores) * 2
                    risk_metrics['svs']['value'] = svs
                    risk_metrics['svs']['normalized'] = min(svs / 50, 1)
                    risk_metrics['details']['svs'] = {
                        'sentiment_scores': sentiment_scores,
                        'std_dev': svs
                    }

                # 3. Robust Engagement Decay Index (EDI)
                if len(videos_data) >= 4:  # Need at least 4 videos for meaningful comparison
                    try:
                        # Split into quartiles (newest 25% vs previous 25%)
                        split_point = len(videos_data) // 4
                        recent_videos = videos_data[:split_point]
                        past_videos = videos_data[split_point:2*split_point]
                        
                        def safe_engagement(video):
                            stats = video.get('statistics', {})
                            views = max(int(stats.get('viewCount', 1)), 1)  # Ensure ≥1
                            likes = int(stats.get('likeCount', 0))
                            comments = int(stats.get('commentCount', 0))
                            return (likes + comments) / views
                        
                        recent_engagements = [safe_engagement(v) for v in recent_videos]
                        past_engagements = [safe_engagement(v) for v in past_videos]
                        
                        # Calculate median engagement (more robust than mean)
                        recent_median = np.median(recent_engagements) if recent_engagements else 0
                        past_median = np.median(past_engagements) if past_engagements else 0
                        
                        if past_median > 0.001:  # Minimum engagement threshold (0.1%)
                            pct_change = (recent_median - past_median) / past_median
                            
                            # Improved normalization curve
                            if pct_change <= -0.5:  # >50% improvement
                                normalized = 0
                            elif pct_change >= 1.0:  # >100% decline
                                normalized = 1
                            else:  # Linear in between
                                normalized = min(max((pct_change + 0.5) / 1.5, 0), 1)
                            
                            risk_metrics['edi']['value'] = pct_change * 100  # Store as percentage
                            risk_metrics['edi']['normalized'] = normalized
                            risk_metrics['edi']['valid'] = True
                            risk_metrics['details']['edi'] = {
                                'recent_median': recent_median,
                                'past_median': past_median,
                                'pct_change': pct_change * 100,
                                'normalization': f"{pct_change:.2f} → {normalized:.2f}"
                            }
                            
                    except Exception as e:
                        print(f"EDI calculation error: {str(e)}")

                # Fallback if EDI calculation failed
                if not risk_metrics['edi']['valid']:
                    risk_metrics['edi']['normalized'] = 0.5  # Neutral score
                    risk_metrics['details']['edi'] = {
                        'reason': 'Insufficient valid data' if len(videos_data) < 4 else 'Calculation error',
                        'videos_analyzed': len(videos_data)
                    }

            # 4. Improved Content Consistency Risk (CCR)
            if len(videos_data) > 1:
                upload_dates = []
                for video in videos_data:
                    try:
                        pub_date = datetime.strptime(
                            video['snippet']['publishedAt'], 
                            '%Y-%m-%dT%H:%M:%SZ'
                        )
                        upload_dates.append(pub_date)
                    except:
                        continue
                
                if len(upload_dates) > 1:
                    upload_dates.sort()
                    intervals = np.diff([d.timestamp() for d in upload_dates]) / (60*60*24)
                    
                    # Calculate consistency score (0-1 where 1 is perfectly consistent)
                    if len(intervals) > 1:
                        cv = np.std(intervals) / np.mean(intervals)  # Coefficient of variation
                        ccr = 1 - min(cv, 1)  # Invert so higher = more risk
                        
                        # Topic consistency analysis
                        titles = [v['snippet']['title'] for v in videos_data[:10]]
                        unique_words = len(set(" ".join(titles).lower().split()))
                        total_words = len(" ".join(titles).split())
                        topic_drift = unique_words / max(total_words, 1)
                        
                        # Combine timing and topic consistency
                        risk_metrics['ccr']['value'] = (ccr + topic_drift) / 2
                        risk_metrics['ccr']['normalized'] = risk_metrics['ccr']['value']
                        risk_metrics['details']['ccr'] = {
                            'upload_intervals': intervals.tolist(),
                            'consistency_score': ccr,
                            'topic_drift': topic_drift
                        }

            # 5. Improved Virality Dependence Index (VDI)
            if len(videos_data) > 2:
                view_counts = [int(v.get('statistics', {}).get('viewCount', 0)) for v in videos_data]
                top_percentile = np.percentile(view_counts, 90)
                median_views = np.median(view_counts)
                
                if median_views > 0:
                    # Calculate ratio of top 10% to median
                    vdi = top_percentile / median_views
                    
                    # Improved normalization (1-10 scale → 0-1)
                    risk_metrics['vdi']['value'] = vdi
                    risk_metrics['vdi']['normalized'] = min((vdi - 1) / 9, 1)  # 1x=0, 10x=1
                    risk_metrics['details']['vdi'] = {
                        'top_percentile': top_percentile,
                        'median_views': median_views,
                        'ratio': vdi
                    }

        except Exception as e:
                    st.warning(f"Risk calculation partially failed: {str(e)}")

        # Calculate final risk score with weights
        weights = {
            'gvi': 0.25,
            'svs': 0.20,
            'edi': 0.20,
            'ccr': 0.15,
            'vdi': 0.20
        }
        
        # Safely calculate risk score
        risk_score = sum(
            risk_metrics[metric].get('normalized', 0.5) * weight 
            for metric, weight in weights.items()
        )
        
        risk_multiplier = max(0.5, 1 - (risk_score ** 1.5))
        
        return risk_multiplier, {
            'risk_score': risk_score,
            'normalized_metrics': {k: v.get('normalized', 0.5) for k, v in risk_metrics.items()},
            'raw_metrics': {k: v.get('value', 0) for k, v in risk_metrics.items()},
            'weights': weights,
            'details': risk_metrics['details']
        }
    
    def calculate_yccv_score(self, channel_id, analyzer):
        """Calculate the complete YCCV score with detailed explanations"""
        # Get channel information
        channel_info = analyzer.get_channel_info(channel_id)
        if 'items' not in channel_info or len(channel_info['items']) == 0:
            return None, "Channel not found"
        
        channel_data = channel_info['items'][0]
        channel_stats = channel_data['statistics']
        
        # Get recent videos
        recent_videos_response = analyzer.get_recent_videos(channel_id, 20)
        if 'items' not in recent_videos_response:
            return None, "Could not fetch videos"
        
        video_ids = [item['contentDetails']['videoId'] for item in recent_videos_response['items']]
        videos_data = analyzer.get_video_statistics(video_ids)
        
        # Analyze niche and generate description
        niche_analysis = analyzer.analyze_content_niche(videos_data, channel_data)
        creator_description = analyzer.generate_creator_description(channel_data, videos_data, niche_analysis)
        
        # Calculate pillar scores with detailed metrics
        engagement_score, engagement_details = self.calculate_engagement_score(videos_data)
        growth_score, growth_details = self.calculate_growth_metrics(channel_stats, videos_data)
        community_score, community_details = self.calculate_community_health(videos_data, analyzer)
        strategy_score, strategy_details = self.calculate_content_strategy(videos_data)
        
        
        # Compile pillar scores
        pillar_scores = {
            'engagement': engagement_score,
            'growth': growth_score,
            'community': community_score,
            'strategy': strategy_score,
        }
        
        # Calculate weighted overall score
        overall_score = (
            pillar_scores['engagement'] * self.weights['content_engagement'] +
            pillar_scores['growth'] * self.weights['audience_growth'] +
            pillar_scores['community'] * self.weights['community_health'] +
            pillar_scores['strategy'] * self.weights['content_strategy'] 
        )
        
        # Apply risk factors
        risk_multiplier, risk_analysis = self.calculate_risk_factors(videos_data, channel_stats,analyzer)
        final_score = overall_score * risk_multiplier
        
        return {
            'final_score': final_score,
            'pillar_scores': pillar_scores,
            'risk_multiplier': risk_multiplier,
            'risk_analysis': risk_analysis,
            'channel_info': channel_data,
            'raw_metrics': {
                'subscribers': int(channel_stats.get('subscriberCount', 0)),
                'total_views': int(channel_stats.get('viewCount', 0)),
                'video_count': int(channel_stats.get('videoCount', 0))
            },
            'detailed_metrics': {
                'engagement': engagement_details,
                'growth': growth_details,
                'community': community_details,
                'strategy': strategy_details,
                
            },
            'niche_analysis': niche_analysis,
            'creator_description': creator_description
        }, "Success"

def main():
    st.set_page_config(
        page_title="YouTube Creator Comprehensive Value (YCCV) Calculator",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 YouTube Creator Comprehensive Value (YCCV) Calculator")
    st.markdown("Analyze any YouTube creator's comprehensive value using advanced metrics across 5 core pillars.")
    
    # Initialize components
    analyzer = YouTubeAnalyzer(YOUTUBE_API_KEY)
    calculator = YCCVCalculator()
    
    # Input section
    st.header("🎯 Channel Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        channel_input = st.text_input(
            "Enter YouTube Channel URL, Handle, or Username:",
            placeholder="e.g., @MrBeast, https://youtube.com/@MrBeast, or MrBeast",
            help="You can enter various formats: @username, channel URL, or just the username"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze Channel", type="primary")
    
    if analyze_button and channel_input:
        with st.spinner("Analyzing channel... This may take a moment."):
            # Extract channel ID
            channel_id = analyzer.extract_channel_id(channel_input)
            
            if not channel_id:
                st.error("❌ Could not find the specified channel. Please check the input and try again.")
                return
            
            # Calculate YCCV score
            result, message = calculator.calculate_yccv_score(channel_id, analyzer)
            
            if result is None:
                st.error(f"❌ Error: {message}")
                return
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Creator Description Section
            st.header("👤 Creator Profile")
            st.markdown(result['creator_description'])
            
            # Niche Analysis
            niche_info = result['niche_analysis']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Primary Niche", niche_info['primary_niche'])
            
            with col2:
                confidence_pct = niche_info['confidence'] * 100
                st.metric("Niche Confidence", f"{confidence_pct:.1f}%")
            
            with col3:
                if len(niche_info['top_niches']) > 1:
                    other_niches = ', '.join(niche_info['top_niches'][1:3])
                    st.metric("Secondary Niches", other_niches if other_niches else "None")
                else:
                    st.metric("Secondary Niches", "None")
            
            # Channel Info Header
            channel_info = result['channel_info']
            st.header(f"📺 {channel_info['snippet']['title']} - Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "YCCV Score",
                    f"{result['final_score']:.1f}/100",
                    help="Overall YouTube Creator Comprehensive Value score"
                )
            
            with col2:
                st.metric(
                    "Subscribers",
                    f"{result['raw_metrics']['subscribers']:,}",
                )
            
            with col3:
                st.metric(
                    "Total Views",
                    f"{result['raw_metrics']['total_views']:,}",
                )
            
            with col4:
                st.metric(
                    "Videos",
                    f"{result['raw_metrics']['video_count']:,}",
                )
            
            # Pillar Scores Visualization
            st.header("📊 Pillar Scores Breakdown")
            
            pillar_names = [
                "Content Engagement\n& Quality (35%)",
                "Audience Growth\n& Reach (25%)",
                "Community Health\n& Sentiment (20%)",
                "Content Strategy\n& Consistency (20)",
            ]
            
            pillar_values = [
                result['pillar_scores']['engagement'],
                result['pillar_scores']['growth'],
                result['pillar_scores']['community'],
                result['pillar_scores']['strategy'],
            ]
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=pillar_values,
                theta=pillar_names,
                fill='toself',
                name='YCCV Scores',
                line_color='rgb(0, 123, 255)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="YCCV Pillar Scores",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Score Explanations
            st.header("📋 Detailed Score Analysis")
            
            # Create tabs for each pillar
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎥 Content Engagement", 
                "📈 Audience Growth", 
                "💬 Community Health", 
                "📅 Content Strategy", 
            ])
            
            with tab1:
                st.subheader("Content Engagement & Quality Analysis")
                st.metric("Score", f"{result['pillar_scores']['engagement']:.1f}/100")
                
                engagement_data = result['detailed_metrics']['engagement']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Key Metrics:**")
                    st.write(f"• **Videos Analyzed:** {engagement_data['total_videos']}")
                    st.write(f"• **Total Views:** {engagement_data['total_views']:,}")
                    st.write(f"• **Total Likes:** {engagement_data['total_likes']:,}")
                    st.write(f"• **Total Comments:** {engagement_data['total_comments']:,}")
                    st.write(f"• **Avg Engagement Rate:** {engagement_data['avg_engagement_rate']:.2f} per 1000 views")
                
                with col2:
                    st.markdown("**🔍 How This Score is Calculated:**")
                    st.write("1. **Engagement Rate = (Likes + Comments×2) ÷ Views × 1000**")
                    st.write("2. Comments are weighted 2x as they represent deeper engagement")
                    st.write("3. Industry benchmark: 10-30 per 1000 views is good")
                    st.write("4. Score normalized to 0-100 scale")
                    st.write("5. Higher engagement = better content quality")
                
                # Engagement distribution chart
                if engagement_data['engagement_rates']:
                    fig_eng = px.histogram(
                        x=engagement_data['engagement_rates'],
                        title="Engagement Rate Distribution Across Videos",
                        labels={'x': 'Engagement Rate (per 1000 views)', 'y': 'Number of Videos'}
                    )
                    st.plotly_chart(fig_eng, use_container_width=True)
            
            with tab2:
                st.subheader("Audience Growth & Reach Analysis")
                st.metric("Score", f"{result['pillar_scores']['growth']:.1f}/100")
                
                growth_data = result['detailed_metrics']['growth']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Key Metrics:**")
                    st.write(f"• **Subscribers:** {growth_data['subscriber_count']:,}")
                    st.write(f"• **Avg Views/Video:** {growth_data['avg_views_per_video']:,.0f}")
                    if 'recent_performance' in growth_data:
                        st.write(f"• **Recent Avg Views:** {growth_data['recent_performance']['avg_recent_views']:,.0f}")
                        st.write(f"• **Estimated CTR:** {growth_data.get('estimated_ctr', 0):.2f}%")
                
                with col2:
                    st.markdown("**🔍 How This Score is Calculated:**")
                    st.write("1. **Estimated CTR = Recent Avg Views ÷ Subscribers × 100**")
                    st.write("2. **Growth Indicator = Recent Performance vs Historical Average**")
                    st.write("3. Higher recent performance indicates growing audience interest")
                    st.write("4. CTR shows how well content attracts subscriber attention")
                    st.write("5. Final score combines both metrics")
            
            with tab3:
                st.subheader("Community Health & Sentiment Analysis")
                st.metric("Score", f"{result['pillar_scores']['community']:.1f}/100")
                
                community_data = result['detailed_metrics']['community']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Key Metrics:**")
                    st.write(f"• **Videos Analyzed:** {community_data['videos_analyzed']}")
                    st.write(f"• **Comments Analyzed:** {community_data['total_comments_analyzed']}")
                    st.write(f"• **Average Sentiment:** {community_data['avg_sentiment']:.2f}")
                    
                    if community_data['sentiment_breakdown']['total'] > 0:
                        breakdown = community_data['sentiment_breakdown']
                        pos_pct = (breakdown['positive'] / breakdown['total']) * 100
                        neg_pct = (breakdown['negative'] / breakdown['total']) * 100
                        neu_pct = (breakdown['neutral'] / breakdown['total']) * 100
                        
                        st.write(f"• **Positive Comments:** {pos_pct:.1f}% ({breakdown['positive']})")
                        st.write(f"• **Negative Comments:** {neg_pct:.1f}% ({breakdown['negative']})")
                        st.write(f"• **Neutral Comments:** {neu_pct:.1f}% ({breakdown['neutral']})")
                
                with col2:
                    st.markdown("**🔍 How This Score is Calculated:**")
                    st.write("1. **Sentiment Analysis using TextBlob library**")
                    st.write("2. Comments classified as Positive (>0.1), Negative (<-0.1), Neutral")
                    st.write("3. **Net Sentiment = (Positive - Negative) ÷ Total × 100**")
                    st.write("4. Score shifted to 0-100 range (sentiment + 50)")
                    st.write("5. Higher positive sentiment = healthier community")
                
                # Sentiment chart
                if community_data['sentiment_breakdown']['total'] > 0:
                    breakdown = community_data['sentiment_breakdown']
                    fig_sent = px.pie(
                        values=[breakdown['positive'], breakdown['neutral'], breakdown['negative']],
                        names=['Positive', 'Neutral', 'Negative'],
                        title="Community Sentiment Distribution",
                        color_discrete_sequence=['#2E8B57', '#FFD700', '#DC143C']
                    )
                    st.plotly_chart(fig_sent, use_container_width=True)
            
            with tab4:
                st.subheader("Content Strategy & Consistency Analysis")
                st.metric("Score", f"{result['pillar_scores']['strategy']:.1f}/100")
                
                strategy_data = result['detailed_metrics']['strategy']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Key Metrics:**")
                    st.write(f"• **Videos Analyzed:** {strategy_data['total_videos']}")
                    if strategy_data['avg_interval_days'] > 0:
                        st.write(f"• **Avg Upload Interval:** {strategy_data['avg_interval_days']:.1f} days")
                        st.write(f"• **Upload Pattern:** {strategy_data['consistency_rating']}")
                        if 'consistency_penalty' in strategy_data:
                            st.write(f"• **Consistency Penalty:** {strategy_data['consistency_penalty']:.1f}%")
                
                with col2:
                    st.markdown("**🔍 How This Score is Calculated:**")
                    st.write("1. **Upload intervals calculated between consecutive videos**")
                    st.write("2. **Base scores:** Daily(100), Weekly(85), Bi-weekly(70), Monthly(50)")
                    st.write("3. **Consistency penalty applied for irregular uploads**")
                    st.write("4. Penalty = (Standard Deviation ÷ Mean Interval) × 20")
                    st.write("5. Regular uploads = better audience retention")
                
                # Upload frequency chart
                if strategy_data['upload_intervals']:
                    fig_upload = px.line(
                        y=strategy_data['upload_intervals'],
                        title="Upload Intervals Over Time",
                        labels={'y': 'Days Between Uploads', 'x': 'Video Sequence'}
                    )
                    fig_upload.add_hline(
                        y=np.mean(strategy_data['upload_intervals']), 
                        line_dash="dash", 
                        annotation_text=f"Average: {np.mean(strategy_data['upload_intervals']):.1f} days"
                    )
                    st.plotly_chart(fig_upload, use_container_width=True)

            
            # Risk Analysis Section
            st.header("⚠️ Risk Assessment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Score Breakdown")
                
                risk_metrics = result['risk_analysis']['normalized_metrics']
                weights = result['risk_analysis']['weights']
                
                # Create radar chart for risk factors
                risk_fig = go.Figure()
                
                risk_fig.add_trace(go.Scatterpolar(
                    r=[risk_metrics['gvi'], risk_metrics['svs'], risk_metrics['edi'], risk_metrics['ccr'], risk_metrics['vdi']],
                    theta=['Growth Volatility', 'Sentiment Variability', 'Engagement Decay', 'Content Consistency', 'Virality Dependence'],
                    fill='toself',
                    name='Risk Factors',
                    line_color='rgb(220, 53, 69)'
                ))
                
                risk_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Risk Factor Scores (Normalized 0-1)",
                    height=400
                )
                
                st.plotly_chart(risk_fig, use_container_width=True)
            
            with col2:
                st.subheader("Risk Analysis Summary")
                st.metric(
                    "Overall Risk Score",
                    f"{result['risk_analysis']['risk_score']:.3f}",
                    help="Higher score indicates higher risk (0-1 scale)"
                )
                
                st.metric(
                    "Risk Multiplier Applied",
                    f"{result['risk_multiplier']:.3f}",
                    help="Applied to overall score (1.0 = no risk)"
                )
                
                st.markdown("**Risk Factor Details:**")
                st.write(f"• **Growth Volatility Index (GVI):** {risk_metrics['gvi']:.3f} (Weight: {weights['gvi']*100}%)")
                st.write(f"• **Sentiment Variability Score (SVS):** {risk_metrics['svs']:.3f} (Weight: {weights['svs']*100}%)")
                st.write(f"• **Engagement Decay Index (EDI):** {risk_metrics['edi']:.3f} (Weight: {weights['edi']*100}%)")
                st.write(f"• **Content Consistency Risk (CCR):** {risk_metrics['ccr']:.3f} (Weight: {weights['ccr']*100}%)")
                st.write(f"• **Virality Dependence Index (VDI):** {risk_metrics['vdi']:.3f} (Weight: {weights['vdi']*100}%)")
                
                if result['risk_analysis']['risk_score'] > 0.7:
                    st.error("⚠️ **High Risk Warning:** This channel shows significant risk factors in multiple areas")
                elif result['risk_analysis']['risk_score'] > 0.4:
                    st.warning("⚠️ **Moderate Risk Warning:** This channel shows some risk factors to monitor")
                else:
                    st.success("✅ **Low Risk:** This channel shows stable performance with minimal risk factors")
            
            # Overall Summary
            st.header("🎯 Overall Assessment")
            
            score = result['final_score']
            if score >= 80:
                st.success("🌟 **Excellent** - High-performing creator with strong metrics across all pillars")
                st.balloons()
            elif score >= 60:
                st.info("👍 **Good** - Solid performance with room for improvement in some areas")
            elif score >= 40:
                st.warning("⚡ **Moderate** - Mixed performance, focus on key growth areas")
            else:
                st.error("📉 **Needs Improvement** - Significant opportunities for growth across multiple pillars")
            
            # Detailed Recommendations
            st.header("💡 Detailed Recommendations")
            
            recommendations = []
            
            if result['pillar_scores']['engagement'] < 50:
                recommendations.append({
                    'category': 'Content Engagement',
                    'icon': '🎥',
                    'suggestion': 'Focus on creating more engaging content with better hooks, storytelling, and call-to-actions',
                    'score': result['pillar_scores']['engagement']
                })
            
            if result['pillar_scores']['growth'] < 50:
                recommendations.append({
                    'category': 'Audience Growth',
                    'icon': '📈',
                    'suggestion': 'Optimize thumbnails, titles, and posting times. Consider trending topics in your niche',
                    'score': result['pillar_scores']['growth']
                })
            
            if result['pillar_scores']['community'] < 50:
                recommendations.append({
                    'category': 'Community Health',
                    'icon': '💬',
                    'suggestion': 'Engage more with your audience through comments, community posts, and live streams',
                    'score': result['pillar_scores']['community']
                })
            
            if result['pillar_scores']['strategy'] < 50:
                recommendations.append({
                    'category': 'Content Strategy',
                    'icon': '📅',
                    'suggestion': 'Establish a consistent posting schedule and stick to it. Plan content in advance',
                    'score': result['pillar_scores']['strategy']
                })
            
            
            if recommendations:
                st.markdown("**🎯 Priority Areas for Improvement:**")
                # Sort by lowest score first (highest priority)
                recommendations.sort(key=lambda x: x['score'])
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{rec['icon']} Priority {i}: {rec['category']} (Score: {rec['score']:.1f}/100)"):
                        st.write(rec['suggestion'])
                        
                        # Add specific tips based on category
                        if rec['category'] == 'Content Engagement':
                            st.markdown("""
                            **Specific Tips:**
                            - Use pattern interrupts in first 15 seconds
                            - Ask questions to encourage comments
                            - Create content series to build anticipation
                            - Improve video quality (lighting, audio, editing)
                            """)
                        elif rec['category'] == 'Audience Growth':
                            st.markdown("""
                            **Specific Tips:**
                            - A/B test different thumbnail styles
                            - Use keyword research for titles
                            - Post when your audience is most active
                            - Collaborate with other creators
                            """)
                        elif rec['category'] == 'Community Health':
                            st.markdown("""
                            **Specific Tips:**
                            - Respond to comments within first few hours
                            - Create community polls and discussions
                            - Address negative feedback constructively
                            - Build a positive community culture
                            """)
                        elif rec['category'] == 'Content Strategy':
                            st.markdown("""
                            **Specific Tips:**
                            - Use content calendar tools
                            - Batch record multiple videos
                            - Plan seasonal and trending content
                            - Maintain consistent branding
                            """)
                
            else:
                st.success("🎉 **Excellent Performance!** Your channel is performing well across all metrics. Keep up the great work!")
                st.markdown("""
                **Suggestions for Continued Growth:**
                - Experiment with new content formats
                - Expand into related niches
                - Build strategic partnerships
                - Consider launching your own products/services
                """)
    
    # Information section
    st.header("ℹ️ About YCCV Model")
    
    with st.expander("Learn more about the YCCV calculation methodology"):
        st.markdown("""
        The **YouTube Creator Comprehensive Value (YCCV)** model is a sophisticated algorithm that evaluates creators across five core pillars:
        
        ### 📊 Core Pillars & Weights:
        
        #### 1. Content Engagement & Quality (35%) 🎥
        - **Calculation:** (Likes + Comments×2) ÷ Views × 1000
        - **Benchmark:** 10-30 per 1000 views is considered good
        - **Why it matters:** Higher engagement indicates quality content that resonates with audience
        
        #### 2. Audience Growth & Reach (25%) 📈
        - **Calculation:** Recent performance vs subscriber base + growth indicators
        - **Factors:** Click-through rate estimation, view consistency, subscriber engagement
        - **Why it matters:** Shows channel's ability to attract and retain new viewers
        
        #### 3. Community Health & Sentiment (20%) 💬
        - **Calculation:** Sentiment analysis of recent comments using TextBlob
        - **Scale:** Net sentiment score (positive - negative) normalized to 0-100
        - **Why it matters:** Healthy community indicates sustainable long-term growth
        
        #### 4. Content Strategy & Consistency (20) 📅
        - **Calculation:** Upload frequency analysis with consistency penalties
        - **Scoring:** Daily(100), Weekly(85), Bi-weekly(70), Monthly(50), Irregular(penalty)
        - **Why it matters:** Consistent posting builds audience expectations and retention
        
        ### ⚠️ Risk Assessment Framework:
        
        The risk assessment now includes five key metrics as shown in the images:
        
        #### 1. Growth Volatility Index (GVI)
        - **Captures:** Instability in subscriber/view growth
        - **Calculation:** std_dev(daily_pct_change(subscriber_count))
        
        #### 2. Sentiment Variability Score (SVS)
        - **Captures:** Emotional volatility in audience response
        - **Calculation:** std_dev(sentiment_scores)
        
        #### 3. Engagement Decay Index (EDI)
        - **Captures:** Recent falloff in engagement rates
        - **Calculation:** log(recent_engagement_rate / past_engagement_rate)
        
        #### 4. Content Consistency Risk (CCR)
        - **Captures:** Irregular posting and topic drift
        - **Calculation:** std_dev(days_between_uploads) + topic_drift_penalty
        
        #### 5. Virality Dependence Index (VDI)
        - **Captures:** Fragility due to over-reliance on viral content
        - **Calculation:** max_3_video_views / median_video_views
        
        #### Final Risk Score:
        - **Calculation:** risk_score = w1*GVI + w2*SVS + w3*EDI + w4*CCR + w5*VDI
        - All sub-indices are normalized to [0,1]
        - Weights are equal (20% each) by default
        - Risk multiplier = max(0.5, 1 - risk_score)
        
        ### 🔢 Final Score Interpretation:
        - **80-100:** Excellent - Top-tier creator with strong fundamentals
        - **60-79:** Good - Solid performance with growth opportunities  
        - **40-59:** Moderate - Mixed results, focus on improvement areas
        - **0-39:** Needs Improvement - Significant development required
        
        ### 🎯 Niche Analysis:
        The system automatically analyzes content to determine:
        - **Primary niche** based on keyword frequency in titles, descriptions, and tags
        - **Confidence score** indicating certainty of niche classification
        - **Secondary niches** for diversified content creators
        
        ### 📈 Data Sources:
        - YouTube Data API v3 for all metrics
        - TextBlob library for sentiment analysis
        - Statistical analysis for consistency and volatility calculations
        - Industry benchmarks for score normalization
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*YCCV Calculator - Comprehensive YouTube Creator Analytics*")

if __name__ == "__main__":
    main()