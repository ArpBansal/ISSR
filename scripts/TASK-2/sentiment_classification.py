import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
import nltk
import os
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
nltk.download('vader_lexicon')

# Parallel VADER sentiment analysis
def process_chunk(chunk_df, text_column='content'):
    sid = SentimentIntensityAnalyzer()
    chunk_df['sentiment_scores'] = chunk_df[text_column].apply(
        lambda text: sid.polarity_scores(str(text)) if pd.notna(text) else {'compound': 0}
    )
    chunk_df['sentiment_score'] = chunk_df['sentiment_scores'].apply(lambda x: x['compound'])

    def categorize_sentiment(score):
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'

    chunk_df['sentiment'] = chunk_df['sentiment_score'].apply(categorize_sentiment)
    chunk_df = chunk_df.drop('sentiment_scores', axis=1)
    return chunk_df

def analyze_sentiment_vader_parallel(df, text_column='content', batch_size=1000, n_jobs=None):
    """
    Parallel implementation of VADER sentiment analysis
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    result_df = df.copy()

    # Split dataframe into chunks for parallel processing
    df_chunks = np.array_split(result_df, n_jobs)

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        processed_chunks = list(tqdm(
            executor.map(partial(process_chunk, text_column=text_column), df_chunks),
            total=len(df_chunks),
            desc="Sentiment Analysis"
        ))

    # Combine results
    result_df = pd.concat(processed_chunks)
    return result_df

# Custom PyTorch Dataset for BERT embeddings
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Multi-GPU BERT embeddings function
def get_bert_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32, num_gpus=None):
    """
    Generate embeddings using BERT with multi-GPU support
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        # Fall back to CPU if no GPUs available
        model = SentenceTransformer(model_name)
        return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Create multiple model instances for different GPUs
    models = []
    for gpu_id in range(num_gpus):
        device = f"cuda:{gpu_id}"
        model = SentenceTransformer(model_name)
        model.to(device)
        models.append((model, device))

    # Create DataLoader
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size * num_gpus, shuffle=False)

    # Generate embeddings in batches
    all_embeddings = []

    for batch in tqdm(dataloader, desc="Generating BERT embeddings"):
        # Split batch for each GPU
        batch_size_per_gpu = len(batch) // num_gpus
        batch_splits = []

        for i in range(num_gpus):
            start_idx = i * batch_size_per_gpu
            end_idx = start_idx + batch_size_per_gpu if i < num_gpus - 1 else len(batch)
            batch_splits.append(batch[start_idx:end_idx])

        # Process in parallel across GPUs
        batch_embeddings = []

        def encode_on_gpu(model_device, batch_text):
            model, device = model_device
            return model.encode(batch_text)

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            batch_results = list(executor.map(
                encode_on_gpu, 
                models,
                batch_splits
            ))

        # Combine results
        for result in batch_results:
            batch_embeddings.append(result)

        combined_embeddings = np.vstack(batch_embeddings)
        all_embeddings.append(combined_embeddings)

    return np.vstack(all_embeddings)



def preprocess(text):
    if pd.isna(text) or text == '':
        return []
    return word_tokenize(text.lower())

def calculate_crisis_score(tokens, crisis_terms_set):
    if not tokens:
        return 0, []

    # Find crisis terms in post
    found_terms = [t for t in tokens if t in crisis_terms_set]

    # Calculate score based on number of crisis terms
    score = len(found_terms) / len(tokens) if tokens else 0

    return score, found_terms

def extract_terms_from_text(text_series, min_words=2, workers=None):
    if workers is None:
        workers = multiprocessing.cpu_count()


    with ProcessPoolExecutor(max_workers=workers) as executor:
        all_words_lists = list(executor.map(tokenize_text, text_series))

    # Flatten list of lists
    all_words = [word for sublist in all_words_lists for word in sublist]

    # Count word frequencies
    word_counts = pd.Series(all_words).value_counts()

    # Return top words
    return word_counts.head(20).index.tolist()

# Tokenize in parallel
def tokenize_text(text):
    if pd.isna(text) or text == '':
        return []
    words = word_tokenize(text.lower())
    min_words = 2
    return [w for w in words if len(w) > min_words and w.isalpha()]


def find_crisis_terms(text, terms):
    if pd.isna(text) or text == '' or not terms:
        return []

    text_lower = text.lower()
    found_terms = [term for term in terms if term in text_lower]
    return found_terms


# Detect high-risk terms using parallel BERT
def detect_crisis_terms_bert_parallel(df, text_column='content', title_column='title', batch_size=32, num_gpus=None):
    """
    Detect high-risk crisis terms using BERT with multi-GPU support
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    result_df = df.copy()

    # Combine title and content
    result_df['combined_text'] = result_df.apply(
        lambda row: str(row[title_column] or '') + ' ' + str(row[text_column] or ''), 
        axis=1
    )

    # Get post embeddings using multi-GPU
    posts = result_df['combined_text'].fillna('').tolist()
    if not posts:
        print("Warning: No valid text found for BERT model")
        result_df['crisis_score'] = 0
        result_df['high_risk_terms'] = [[] for _ in range(len(result_df))]
        return result_df

    post_embeddings = get_bert_embeddings(
        texts=posts,
        batch_size=batch_size,
        num_gpus=num_gpus
    )

    print("Clustering posts to identify high-risk content...")

    # Use KMeans to cluster posts
    n_clusters = min(5, len(posts))  # Use fewer clusters for smaller datasets
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(post_embeddings)

    # Identify high-risk cluster by looking for posts labeled as high-risk
    high_risk_posts_idx = result_df['risk_level'] == 'High-Risk'

    if high_risk_posts_idx.any():
        # Count posts in each cluster by risk level
        cluster_risk_counts = pd.crosstab(
            index=cluster_labels, 
            columns=result_df['risk_level']
        )

        # Identify clusters with higher proportion of high-risk posts
        if 'High-Risk' in cluster_risk_counts.columns:
            # Calculate proportion of high-risk posts in each cluster
            cluster_risk_props = cluster_risk_counts.div(
                cluster_risk_counts.sum(axis=1), axis=0
            )

            # Get clusters with high proportion of high-risk posts
            high_risk_clusters = cluster_risk_props[
                cluster_risk_props['High-Risk'] > 0.3
            ].index.tolist()
        else:
            high_risk_clusters = []
    else:
        # If no known high-risk posts, look for outlier clusters (smallest clusters)
        cluster_counts = pd.Series(cluster_labels).value_counts()
        high_risk_clusters = cluster_counts[cluster_counts < cluster_counts.median()].index.tolist()

    # Set crisis score based on cluster membership
    result_df['crisis_score'] = [
        0.9 if label in high_risk_clusters else 0.1 for label in cluster_labels
    ]

    # Extract keywords from high-risk clusters in parallel
    print("Extracting crisis terms from high-risk clusters...")

    # Function to extract most common words

    # Extract terms from high-risk clusters
    if high_risk_clusters:
        high_risk_mask = [label in high_risk_clusters for label in cluster_labels]
        high_risk_texts = result_df.loc[high_risk_mask, 'combined_text']

        workers = multiprocessing.cpu_count()
        crisis_terms = extract_terms_from_text(high_risk_texts, workers=workers)
        print(f"Extracted {len(crisis_terms)} crisis terms")
    else:
        crisis_terms = []

    # If crisis_terms is defined
    if not crisis_terms:
        print("Warning: No crisis terms found. Using fallback terms.")
        crisis_terms = [
            'suicide', 'kill myself', 'end my life', 'die', 'better off dead', 
            'no reason to live', 'can\'t go on', 'take my own life', 'ending it all',
            'want to die', 'don\'t want to be here', 'give up', 'hopeless',
            'self harm', 'cut myself', 'hurt myself', 'overdose', 'pills',
            'worthless', 'burden', 'no purpose', 'no point', 'saying goodbye',
            'last post', 'final note', 'throwaway account', 'final post', 
            'delete this later', 'not going to respond', 'just needed to say this'
        ]
        result_df = result_df.drop(['crisis_score'], axis=1)

    # Saving crisis_terms
    with open('crisis_terms.txt', 'w') as f:
        f.write(str(crisis_terms))
    # Apply to each post in parallel
    find_terms = partial(find_crisis_terms, terms=crisis_terms)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        high_risk_terms = list(tqdm(
            executor.map(find_terms, result_df['combined_text']),
            total=len(result_df),
            desc="Finding crisis terms in posts"
        ))

    result_df['high_risk_terms'] = high_risk_terms

    # Drop temporary columns
    result_df = result_df.drop('combined_text', axis=1)

    return result_df

# Interactive HTML Visualizations with Plotly
def create_interactive_visualizations(df, output_dir='.', method:str='bert'):
    os.makedirs(output_dir, exist_ok=True)

    # Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    fig1 = px.bar(
        sentiment_counts, 
        x='Sentiment', 
        y='Count',
        color='Sentiment',
        title='Distribution of Posts by Sentiment',
        color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
    )

    fig1.update_layout(
        xaxis_title='Sentiment',
        yaxis_title='Number of Posts',
        legend_title='Sentiment'
    )

    fig1.write_html(f"{output_dir}/sentiment_{method}distribution.html")

    # Sentiment by risk level
    fig2 = px.histogram(
        df, 
        x='risk_level',
        color='sentiment',
        barmode='group',
        title='Distribution of Posts by Risk Level and Sentiment',
        color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
    )

    fig2.update_layout(
        xaxis_title='Risk Level',
        yaxis_title='Number of Posts',
        legend_title='Sentiment'
    )

    fig2.write_html(f"{output_dir}/sentiment_{method}_by_risk.html")

    if 'crisis_score' in df.columns:
        # Crisis score by risk level
        fig3 = px.box(
            df,
            x='risk_level',
            y='crisis_score',
            color='risk_level',
            title='Crisis Scores by Risk Level'
        )

        fig3.update_layout(
            xaxis_title='Risk Level',
            yaxis_title='Crisis Score'
        )

        fig3.write_html(f"{output_dir}/crisis_score_{method}_by_risk.html")

    # Heatmap of sentiment vs risk level
    crosstab = pd.crosstab(df['risk_level'], df['sentiment'])

    fig4 = px.imshow(
        crosstab,
        text_auto=True,
        aspect="auto",
        title='Heatmap of Risk Level vs Sentiment',
        color_continuous_scale='Viridis'
    )

    fig4.update_layout(
        xaxis_title='Sentiment',
        yaxis_title='Risk Level'
    )

    fig4.write_html(f"{output_dir}/sentiment_risk_{method}_heatmap.html")

    # Top high-risk terms
    all_terms = []
    for terms in df['high_risk_terms']:
        all_terms.extend(terms)

    if all_terms:
        term_counts = pd.Series(all_terms).value_counts().reset_index()
        term_counts.columns = ['Term', 'Frequency']
        term_counts = term_counts.sort_values('Frequency', ascending=False).head(15)

        fig5 = px.bar(
            term_counts,
            x='Term',
            y='Frequency',
            title='Top 15 High-Risk Terms',
            color='Frequency',
            color_continuous_scale='Reds'
        )

        fig5.update_layout(
            xaxis_title='Term',
            yaxis_title='Frequency',
            xaxis={'categoryorder':'total descending'}
        )
    else:
        # Create empty figure if no terms
        fig5 = go.Figure()
        fig5.update_layout(
            title="No High-Risk Terms Detected",
            xaxis_title="Term",
            yaxis_title="Frequency"
        )

    fig5.write_html(f"{output_dir}/top_crisis_{method}_terms.html")

    # Create dashboard
    from plotly.subplots import make_subplots
    if 'crisis_score'in df.columns:
        fig = make_subplots(
            rows=2, 
            cols=3,
            subplot_titles=(
                "Sentiment Distribution",
                "Risk Level vs Sentiment",
                "Crisis Scores by Risk Level",
                "Risk Level vs Sentiment Heatmap",
                "Top High-Risk Terms"
            )
        )

        # Add traces to subplots
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)

        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=2)

        for trace in fig3.data:
            fig.add_trace(trace, row=1, col=3)

        for trace in fig4.data:
            fig.add_trace(trace, row=2, col=1)

        for trace in fig5.data:
            fig.add_trace(trace, row=2, col=2)

        # Update layout
        fig.update_layout(
            title_text="Reddit Post Analysis Dashboard",
            height=900,
            width=1500,
            showlegend=False
        )

        fig.write_html(f"{output_dir}/dashboard_{method}_.html")

        return fig1, fig2, fig3, fig4, fig5

    else:
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                "Sentiment Distribution",
                "Risk Level vs Sentiment",
                "Risk Level vs Sentiment Heatmap",
                "Top High-Risk Terms"
            )
        )

        # Add traces to subplots
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)

        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=2)

        for trace in fig4.data:
            fig.add_trace(trace, row=2, col=1)

        for trace in fig5.data:
            fig.add_trace(trace, row=2, col=2)

        # Update layout
        fig.update_layout(
            title_text="Reddit Post Analysis Dashboard",
            height=900,
            width=1500,
            showlegend=False
        )

        fig.write_html(f"{output_dir}/dashboard_{method}_.html")
        return fig1, fig2, fig4, fig5

# Main execution function
def process_reddit_data(df, output_dir='.', method='bert', batch_size=32, num_gpus=None):
    """
    Process Reddit data with parallel sentiment analysis and crisis term detection.

    Args:
        df: DataFrame containing Reddit posts
        output_dir: Directory to save HTML plots
        method: Detection method - 'bert' or 'word2vec'
        batch_size: Batch size for processing
        num_gpus: Number of GPUs to use (None = auto-detect)

    Returns:
        Processed DataFrame with sentiment and crisis term data
    """
    # Auto-detect GPU count if not specified
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for processing")

    workers = multiprocessing.cpu_count()
    print(f"Using {workers} CPU cores for parallel processing")

    # Step 1: Parallel Sentiment Analysis
    print("Performing parallel sentiment analysis...")
    df_with_sentiment = analyze_sentiment_vader_parallel(df, n_jobs=workers)

    print("Detecting high-risk crisis terms with multi-GPU BERT...")
    final_df = detect_crisis_terms_bert_parallel(
        df_with_sentiment, 
        batch_size=batch_size,
        num_gpus=num_gpus
    )

    # Step 3: Create and save interactive visualizations
    print("Creating interactive visualizations...")
    create_interactive_visualizations(final_df, output_dir, method)

    # Print summary statistics
    print("\nSentiment Distribution:")
    print(final_df['sentiment'].value_counts())

    print("\nSentiment by Risk Level:")
    print(pd.crosstab(final_df['risk_level'], final_df['sentiment']))


    all_terms = []
    for terms in final_df['high_risk_terms']:
        all_terms.extend(terms)

    if all_terms:
        print("\nTop 20 Detected Crisis Terms:")
        print(pd.Series(all_terms).value_counts().head(20))

    # Sample high-risk posts with negative sentiment
    high_risk_negative = final_df[
        (final_df['risk_level'] == 'High-Risk') & 
        (final_df['sentiment'] == 'Negative')
    ]

    print(f"\nTop 5 High-Risk Negative Posts (by crisis score):")
    if len(high_risk_negative) > 0:
        for idx, row in high_risk_negative.head().iterrows():
            print(f"Post ID: {row['post_id']}")
            print(f"Title: {row['title']}")
            print(f"Risk Terms: {row['high_risk_terms']}")
            print(f"Sentiment Score: {row['sentiment_score']:.4f}")
            print("-" * 50)

    return final_df

import pandas as pd

# Load your data
df = pd.read_csv('mental_health_postsV1_classified.csv')

# this line ran the above process, uncomment to run it
# processed_df = process_reddit_data(df, method='bert', output_dir='results_bert')
# processed_df.to_csv('crisis_terms_processed.csv')

"""You can just refer dashboard_bert.html for plots"""