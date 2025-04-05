# Provide your CLIENT_ID and CLIENT_SECRET in .env file
from dotenv import load_dotenv
import os
load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


import praw
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()
class RedditMentalHealthScraper:
    def __init__(self, client_id, client_secret, user_agent="SocialAnalysis<v0>"):
        """
        Initialize Reddit API connection
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        self.keywords = [
            # Mental Health
            "depressed", "anxiety", "mental health", "ptsd", "trauma", 
            "burnout", "emotional support", "mental breakdown", 
            "psychological distress", "intrusive thoughts",
            
            # Substance Use
            "addiction", "substance abuse", "recovery", "relapse", 
            "alcohol addiction", "drug addiction", "sober", 
            "addiction help", "substance use disorder",
            
            # Emotional Distress
            "overwhelmed", "struggling", "feeling hopeless", 
            "emotional pain", "self-harm", "suicidal thoughts", 
            "suicide prevention", "mental health crisis",
            
            # Treatment and Support
            "therapy", "counseling", "medication", "support group", 
            "mental health resources", "coping mechanisms"
        ]
    
    def extract_posts(self, target_subreddits, limit=100):
        """
        Extract posts from subreddits containing target keywords
        """
        extracted_posts = []
        
        # Find all subreddits matching target names
        matching_subreddits = []
        for target in target_subreddits:
            try:
                subreddits = self.reddit.subreddits.search_by_name(target, include_nsfw=True)
                for i in subreddits:
                    matching_subreddits.append(i.display_name)
            except Exception as e:
                print(f"Could not find subreddit matching {target}: {e}")
        
        print(f"Matching subreddits found: {matching_subreddits}")
        for subreddit_name in tqdm(matching_subreddits):
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search hot posts and new posts
                for post_stream in [subreddit.hot(limit=limit), subreddit.new(limit=limit)]:
                    for post in post_stream:
                        # Check if post contains any of the keywords
                        if self._contains_keywords(post.title.lower()) or \
                           self._contains_keywords(post.selftext.lower()):
                            
                            post_data = {
                                'post_id': post.id,
                                'timestamp': datetime.fromtimestamp(post.created_utc).isoformat(),
                                'title': post.title,
                                'content': post.selftext,
                                'subreddit': subreddit_name,
                                'ups': post.ups,
                                'num_comments': post.num_comments,
                                'url': post.url
                            }
                            
                            extracted_posts.append(post_data)
            
            except Exception as e:
                print(f"Error processing subreddit {subreddit_name}: {e}")
        
        return extracted_posts
    
    # Method istested but not used in analysis, as it needs higher api calls
    def extract_posts_with_comments(self, target_subreddits, limit=100, comment_limit=50, replace_more_limit=0):
        """
        Extract posts from subreddits containing target keywords along with their comments
        
        Parameters:
        - target_subreddits: List of subreddit names to search
        - limit: Maximum number of posts to retrieve per subreddit
        - comment_limit: Maximum number of comments to retrieve per post
        
        Returns:
        - List of dictionaries containing post data and comments
        """
        extracted_posts = []
        
        # Find all subreddits matching target names
        matching_subreddits = []
        for target in target_subreddits:
            try:
                subreddits = self.reddit.subreddits.search_by_name(target, include_nsfw=True)
                for i in subreddits:
                    matching_subreddits.append(i.display_name)
            except Exception as e:
                print(f"Could not find subreddit matching {target}: {e}")
        
        print(f"Matching subreddits found: {matching_subreddits}")
        for subreddit_name in tqdm(matching_subreddits):
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search hot posts and new posts
                for post_stream in [subreddit.hot(limit=limit), subreddit.new(limit=limit)]:
                    for post in post_stream:
                        # Check if post contains any of the keywords
                        if self._contains_keywords(post.title.lower()) or \
                        self._contains_keywords(post.selftext.lower()):
                            
                            # Extract comments
                            comments_data = []
                            post.comments.replace_more(limit=replace_more_limit)  # Replace MoreComments objects
                            
                            for comment in post.comments.list()[:comment_limit]:
                                comment_data = {
                                    'comment_id': comment.id,
                                    'author': str(comment.author) if comment.author else '[deleted]',
                                    'body': comment.body,
                                    'score': comment.score,
                                    'timestamp': datetime.fromtimestamp(comment.created_utc).isoformat(),
                                    'parent_id': comment.parent_id
                                }
                                comments_data.append(comment_data)
                            
                            post_data = {
                                'post_id': post.id,
                                'timestamp': datetime.fromtimestamp(post.created_utc).isoformat(),
                                'title': post.title,
                                'content': post.selftext,
                                'subreddit': subreddit_name,
                                'ups': post.ups,
                                'num_comments': post.num_comments,
                                'url': post.url,
                                'comments': comments_data
                            }
                            
                            extracted_posts.append(post_data)
            
            except Exception as e:
                print(f"Error processing subreddit {subreddit_name}: {e}")
        
        return extracted_posts
    
    def _contains_keywords(self, text):
        """
        Check if text contains any of the predefined keywords
        """
        return any(keyword.lower() in text for keyword in self.keywords)

    def save_to_csv(self, posts, filename='mental_health_posts.csv'):
        """
        Save extracted posts to CSV
        """
        df = pd.DataFrame(posts)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved {len(posts)} posts to {filename}")

    def save_to_json(self, posts, filename='mental_health_posts.json'):
        """
        Save extracted posts to JSON
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(posts, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(posts)} posts to {filename}")

def main():
    # Reddit API credentials
    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")
    USER_AGENT = 'SocialAnalysis<v0>(by /u/Sad-Net-4568)'
    
    # Target subreddit partial names to search
    target_subreddits = [
        'mentalhealth', 'depression', 'anxiety', 
        'addiction', 'support', 'mental', 'psychiatry', 'ketamine', 'SuicideWatch',
        'bipolar', 'ptsd', 'trauma', 'burnout', 'emotional', 'intrusivethoughts', 'stopdrinking',
        'leaves', 'stopsmoking', 'stopdrugs', 'stopselfharm', 'opiatesrecovery', 'recovery', 'therapy', 'crisis',
        'selfharm', 'stress', 'panic'
    ]

    scraper = RedditMentalHealthScraper(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    
    # Used in analysis
    extracted_posts = scraper.extract_posts(target_subreddits, limit=1000)

    # Tested with comment extraction but not used in further analysis
    # extracted_posts = scraper.extract_posts_with_comments(target_subreddits, limit=100, comment_limit=50, replace_more_limit=1)
    
    scraper.save_to_csv(posts=extracted_posts, filename='mental_health_postsV1.csv')
    # scraper.save_to_json(posts=extracted_posts, filename='mental_health_postsV1.json') # Not used as CSV is easier to work with in Pandas
    print("Extraction complete.")

if __name__ == "__main__":
    main()