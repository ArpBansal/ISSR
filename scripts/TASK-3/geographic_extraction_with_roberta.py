import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.express as px
import requests
import time
import plotly.graph_objects as go
from tqdm import tqdm
from collections import Counter
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import concurrent.futures
import os

# Create NER pipeline on specified device
def create_ner_pipeline(device_id=None):
    device = f"cuda:{device_id}" if device_id is not None else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

    model.to(device)

    return pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)

def extract_locations_with_transformer(text, ner_pipeline):
    """
    Use a transformer-based NER model to identify location mentions in text.
    Returns a list of location names found in the text.
    """
    if not text or pd.isna(text) or text.strip() == "":
        return []

    try:
        # Extract named entities
        results = ner_pipeline(text)
        # Filter for location entities (LOC or GPE)
        locations = []
        for entity in results:
            if entity['entity_group'] in ['LOC', 'GPE']:
                # Clean up the location string
                location = entity['word'].strip()
                if location:
                    locations.append(location)
        return locations

    except Exception as e:
        print(f"Error in NER extraction: {e}")
        return []

def process_batch(batch_df, device_id=None):
    # Create pipeline on appropriate device
    nlp = create_ner_pipeline(device_id)

    batch_results = []

    for _, row in batch_df.iterrows():
        # Extract locations from the combined text
        locations = extract_locations_with_transformer(row['text_for_location'], nlp)

        # Add each location as a separate entry with all original row data
        for loc in locations:
            # Create a copy of the original row
            result_row = row.copy()
            # Add the location
            result_row['extracted_location'] = loc
            # Add to results
            batch_results.append(result_row)

    return batch_results

# Function to geocode locations using a geocoding API
def geocode_location(location_name):
    """
    Convert location name to latitude and longitude using a geocoding API.
    Returns (lat, lng) tuple or None if geocoding fails.
    """
    try:
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": location_name,
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "CrisisLocationAnalysis/1.0"
        }

        response = requests.get(base_url, params=params, headers=headers)

        if response.status_code == 200:
            results = response.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                return (lat, lon)

        return None

    except Exception as e:
        print(f"Geocoding error for {location_name}: {e}")
        return None

def map_risk_level_to_numeric(risk_text):
    # Define mapping of text values to numeric scores
    risk_mapping = {
        'Low Concern': 1.0,
        'Moderate Concern': 2.5,
        'High-Risk': 5.0,
        'Unclassified': 0.0  # Assign a small value for unclassified
    }

    # Return the mapped value or a default if not found
    return risk_mapping.get(risk_text, 0.5)

def main():
    # Load the dataset
    # file_path = "/kaggle/input/reddit-preprocessed-data/mental_health_postsV1_classified.csv" # For kaggle env
    file_path = "mental_health_postsV1_classified.csv"

    df = pd.read_csv(file_path)

    # Combine title and content for better location extraction
    df['text_for_location'] = df['title'].fillna('') + ' ' + df['content'].fillna('')

    # Check for GPU availability
    gpu_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if gpu_available else 0

    if gpu_available:
        print(f"GPU processing enabled! Found {num_gpus} GPU devices.")
        device_type = "GPU"
    else:
        print("No GPUs found. Using CPU processing.")
        device_type = "CPU"
        num_gpus = 0

    # Determine workers based on available hardware
    if gpu_available:
        # Use GPU processing - create one batch per GPU
        num_workers = num_gpus
    else:
        # Use CPU processing - create batches based on CPU cores
        num_workers = os.cpu_count() or 4

    # Create batches
    batch_size = max(1, len(df) // max(1, num_workers))
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    print(f"Using {num_workers} {device_type} workers with batch size {batch_size}")

    # Extract locations using parallel processing
    print("Extracting locations using parallel processing...")
    all_results = []

    if gpu_available:
        # GPU processing using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, batch in enumerate(batches):
                # Assign each batch to a GPU (cycling if more batches than GPUs)
                gpu_id = i % num_gpus
                future = executor.submit(process_batch, batch, gpu_id)
                futures.append(future)

            # Collect results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch processing: {e}")
    else:
        # CPU processing using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(process_batch, batch)
                futures.append(future)

            # Collect results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch processing: {e}")

    # Create a dataframe from all results
    if all_results:
        locations_df = pd.DataFrame(all_results)
    else:
        print("No locations found!")
        return

    # Save the complete dataset with extracted locations
    locations_df.to_csv("crisis_locations_extracted.csv", index=False)
    print(f"Saved {len(locations_df)} entries with location data to crisis_locations_extracted.csv")

    # # Count occurrences of each location
    location_counts = Counter(locations_df['extracted_location'])

    # Geocode each unique location
    print("Geocoding unique locations...")
    geocoded_locations = {}
    unique_locations = list(location_counts.keys())

    for loc in tqdm(unique_locations):
        if loc not in geocoded_locations:
            coords = geocode_location(loc)
            if coords:
                geocoded_locations[loc] = coords
            time.sleep(1)  # Respect API rate limits

    # Add geocoded coordinates to the locations dataframe
    locations_df['coordinates'] = locations_df['extracted_location'].map(lambda x: geocoded_locations.get(x, None))
    locations_df = locations_df.dropna(subset=['coordinates'])

    # Extract latitude and longitude from coordinates
    locations_df['latitude'] = locations_df['coordinates'].map(lambda x: x[0] if x else None)
    locations_df['longitude'] = locations_df['coordinates'].map(lambda x: x[1] if x else None)

    # Save the geocoded data
    locations_df.to_csv("crisis_locations_geocoded.csv", index=False)
    print(f"Saved {len(locations_df)} geocoded entries to crisis_locations_geocoded.csv")

    # this is the data we saved two lines above
    # locations_df = pd.read_csv("/kaggle/input/issr-reddit-locations-data/crisis_locations_geocoded.csv")

    # Calculate total risk per location (if risk_level exists in the dataset)
    geocoded_locations = {}

    # Iterate through unique locations in the dataframe
    for _, row in locations_df.drop_duplicates(subset=['extracted_location']).iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            # Store the coordinates as a tuple (lat, lng)
            geocoded_locations[row['extracted_location']] = (row['latitude'], row['longitude'])

    print(f"Built dictionary with {len(geocoded_locations)} unique locations")

    if 'risk_level' in locations_df.columns:

        print("Checking location names...")
        if any(loc.count('Unclassified') > 1 for loc in locations_df['extracted_location'].unique() if isinstance(loc, str)):
            print("Detected corrupted location names, fixing...")

            # Fix the corrupted location names
            def clean_location_name(name):
                if isinstance(name, str):
                    if name.count('Unclassified') > 1:
                        return 'Unclassified'
                    if name.count('Moderate Concern') > 1:
                        return 'Moderate Concern'
                    if name.count('High-Risk') > 1:
                        return 'High-Risk'
                    if name.count('Low Concern') > 1:
                        return 'Low Concern'
                return name

            # Apply the cleaning function
            locations_df['clean_location'] = locations_df['extracted_location'].apply(clean_location_name)
            # Use the cleaned column for visualization
            locations_df = locations_df.rename(columns={'extracted_location': 'original_location', 'clean_location': 'extracted_location'})

            # Update the geocoded_locations dictionary with the cleaned names
            new_geocoded_locations = {}
            for loc, coords in geocoded_locations.items():
                clean_loc = clean_location_name(loc)
                new_geocoded_locations[clean_loc] = coords
            geocoded_locations = new_geocoded_locations

        # Prepare data for heatmap
        locations_df['numeric_risk'] = locations_df['risk_level'].apply(map_risk_level_to_numeric)

        location_risk = locations_df.groupby('extracted_location')['numeric_risk'].sum().reset_index()
        location_risk = location_risk.sort_values('numeric_risk', ascending=False)

        # Get top 5 locations with highest crisis discussions
        top_5_locations = location_risk.head(5)
        print("\nTop 5 locations with highest crisis discussions:")
        print(top_5_locations)

        # Create heatmap using Folium
        print("Creating heatmap visualization...")
        m = folium.Map(location=[39.50, -98.35], zoom_start=4)

        # Prepare data for heatmap - ensure all values are numeric
        heat_data = []
        for _, row in locations_df.iterrows():
            try:
                lat = float(row['latitude'])
                lng = float(row['longitude'])
                risk = float(row['numeric_risk'])

                # Only add if all values are valid numbers
                if not (pd.isna(lat) or pd.isna(lng) or pd.isna(risk)):
                    heat_data.append([lat, lng, risk])
            except (ValueError, TypeError):
                continue

        # Add heatmap to the map if we have valid data
        if heat_data:
            import numpy as np
            heat_data_array = np.array(heat_data)
            HeatMap(heat_data_array, radius=15).add_to(m)
        else:
            print("Warning: No valid data for heatmap")

        # Add markers for top 5 locations with cleaner labels
        for _, row in top_5_locations.iterrows():
            if row['extracted_location'] in geocoded_locations:
                coords = geocoded_locations[row['extracted_location']]
                # Get the original risk text for this location (first occurrence)
                original_risk = locations_df[locations_df['extracted_location'] == row['extracted_location']]['risk_level'].iloc[0]

                # Create a cleaner popup
                popup_html = f"""
                <div style="font-family: Arial, sans-serif; padding: 5px;">
                    <h4>{row['extracted_location']}</h4>
                    <p>Risk Level: {original_risk}</p>
                    <p>Risk Score: {row['numeric_risk']:.1f}</p>
                </div>
                """

                folium.Marker(
                    location=coords,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=row['extracted_location'],
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)

        m.save('crisis_heatmap.html')

        # Create a Plotly visualization for top locations
        fig = px.bar(
            top_5_locations,
            x='extracted_location',
            y='numeric_risk',
            title='Top 5 Locations with Highest Crisis Discussions',
            labels={'extracted_location': 'Location', 'numeric_risk': 'Crisis Risk Score'},
            text='numeric_risk'
        )

        fig.update_layout(
            xaxis_title="Location",
            yaxis_title="Crisis Risk Score",
            xaxis={'categoryorder': 'total descending'},  # Sort by highest risk
            font=dict(size=12),
            xaxis_tickangle=-45,
            margin=dict(l=50, r=50, t=80, b=100)
        )

        # Format the bar text
        fig.update_traces(
            texttemplate='%{text:.1f}',
            textposition='outside'
        )

        fig.write_html('top_locations.html')

    print("\nAnalysis complete! Files saved:")
    print("- crisis_locations_extracted.csv (all records with location data)")
    print("- crisis_locations_geocoded.csv (records with geocoded locations)")
    if 'risk_level' in locations_df.columns:
        print("- crisis_heatmap.html (interactive map)")
        print("- top_locations.html (bar chart of top locations)")

if __name__ == "__main__":
    main()