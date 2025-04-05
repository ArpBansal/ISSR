from tqdm import tqdm
import pandas as pd
from transformers import pipeline
import torch
from datasets import Dataset

def classify_post_risk(df:pd.DataFrame) -> pd.DataFrame:
    """
    Classify social media posts into risk levels using zero-shot classification.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'post_id' and 'content' columns

    Returns:
    pandas.DataFrame: Original DataFrame with added 'risk_level' column
    """
    # Initialize zero-shot classification pipeline
    # Using a robust model for nuanced classification
    classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli"
    )

    # Define risk level categories with descriptive labels
    risk_categories = [
        "High-Risk: Immediate Crisis",
        "Moderate Concern: Seeking Help",
        "Low Concern: General Discussion"
    ]

    # Classification function with detailed criteria
    def determine_risk_level(text):
        try:
            # Perform zero-shot classification
            result = classifier(
                text, 
                candidate_labels=risk_categories, 
                hypothesis_template="This text indicates {}"
            )

            # Get the top predicted category
            top_category = result['labels'][0]

            if "High-Risk" in top_category:
                return "High-Risk"
            elif "Moderate Concern" in top_category:
                return "Moderate Concern"
            else:
                return "Low Concern"

        except Exception as e:
            print(f"Error classifying text: {text}")
            return "Unclassified"

    # Apply risk classification to the DataFrame
    tqdm.pandas(desc="Classifying posts")
    df['risk_level'] = df['content'].progress_apply(determine_risk_level)
    return df

# Example usage
def main(df:pd.DataFrame): 
    # Classify posts
    classified_df = classify_post_risk(df)
    # classified_df.to_csv('mental_health_postsV1_classified.csv', index=False)
    # Display results
    print(classified_df)

# Additional helper functions for advanced analysis
def get_risk_level_summary(df):
    """
    Generate summary statistics of risk levels
    """
    risk_summary = df['risk_level'].value_counts(normalize=True) * 100
    print("\nRisk Level Distribution:")
    print(risk_summary)
    return risk_summary

def identify_high_risk_posts(df):
    """
    Extract and highlight high-risk posts
    """
    high_risk_posts = df[df['risk_level'] == 'High-Risk']
    print("\nHigh-Risk Posts:")
    print(high_risk_posts)
    return high_risk_posts

if __name__ == "__main__":
    data = pd.read_csv('mental_health_postsV1_preprocessed.csv')
    df = classify_post_risk(data)
    df.to_csv('mental_health_postsV1_classified.csv', index=False)