import pandas as pd
import numpy as np
import spacy
import folium
from folium.plugins import HeatMap
import plotly.express as px
from collections import Counter
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
from tqdm import tqdm

# Load spaCy model for NLP-based place recognition
nlp = spacy.load("en_core_web_lg")

def extract_locations(text):
    """Extract location entities from text using spaCy"""
    if not isinstance(text, str):
        return []

    doc = nlp(text)
    locations = []

    # Extract GPE (Geo-Political Entity) and LOC (Location) entities
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            locations.append(ent.text)

    return locations

def geocode_location(location_name, geolocator):
    """Convert location name to coordinates using geocoding"""
    try:
        # Add 'USA' to improve geocoding accuracy for US locations
        location = geolocator.geocode(location_name, exactly_one=True) # removed US bias, let's see results
        if location is None:
            location = geolocator.geocode(location_name, exactly_one=True)

        if location:
            return {
                'location': location_name,
                'lat': location.latitude,
                'lon': location.longitude,
                'address': location.address
            }
        return None
    except (GeocoderTimedOut, GeocoderUnavailable):
        # Handle timeout errors by waiting and retrying once
        time.sleep(1)
        try:
            location = geolocator.geocode(location_name, exactly_one=True)
            if location:
                return {
                    'location': location_name,
                    'lat': location.latitude,
                    'lon': location.longitude,
                    'address': location.address
                }
            return None
        except (GeocoderTimedOut, GeocoderUnavailable):
            return None

def process_dataset(df:pd.DataFrame):
    """Process the dataset to extract and geocode locations"""
    print("Extracting locations from posts...")

    # Initialize a geolocator with a custom user agent
    geolocator = Nominatim(user_agent="crisis_mapping_tool")

    # Combine title and content for better location extraction
    df['full_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')

    # Extract locations from text
    df['extracted_locations'] = df['full_text'].apply(extract_locations)

    df.to_csv('mental_health_postsV1_extracted_unbiased_locations.csv', index=False)
    # Flatten the list of locations and count occurrences
    all_locations = []
    for locations in df['extracted_locations']:
        all_locations.extend(locations)

    location_counts = Counter(all_locations)

    # Get the top 50 locations for geocoding (to avoid excessive API calls)
    top_locations = location_counts.most_common(50)

    print(f"Top 10 most mentioned locations: {top_locations[:10]}")

    # Geocode the top locations
    geocoded_locations = {}
    print("Geocoding top locations...")
    for location_name, count in tqdm(top_locations):
        if location_name not in geocoded_locations:
            geo_info = geocode_location(location_name, geolocator)
            if geo_info:
                geo_info['count'] = count
                geocoded_locations[location_name] = geo_info
            time.sleep(1)  # Be nice to the geocoding service

    return geocoded_locations, df

def create_folium_heatmap(geocoded_locations, output_file="crisis_heatmap.html"):
    """Create a Folium heatmap of crisis locations"""
    # Create a list of [lat, lon, weight] for heatmap
    heatmap_data = []
    for loc in geocoded_locations.values():
        # Use count as weight and add some randomness to avoid perfect overlaps
        for _ in range(loc['count']):
            # Small random offset for better visualization
            random_lat = loc['lat'] + np.random.normal(0, 0.01)
            random_lon = loc['lon'] + np.random.normal(0, 0.01)
            heatmap_data.append([random_lat, random_lon, 1])

    # Create base map centered on the USA
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    # Add heatmap layer
    HeatMap(heatmap_data).add_to(m)

    # Add markers for top 5 locations
    top_5_locations = sorted(geocoded_locations.values(), key=lambda x: x['count'], reverse=True)[:5]

    for loc in top_5_locations:
        folium.Marker(
            location=[loc['lat'], loc['lon']],
            popup=f"{loc['location']}: {loc['count']} mentions",
            tooltip=loc['location']
        ).add_to(m)

    # Save map
    m.save(output_file)
    print(f"Folium heatmap saved to {output_file}")

    return top_5_locations

def create_plotly_heatmap(geocoded_locations, df, output_file="crisis_plotly_map.html"):
    """Create a Plotly choropleth map of crisis locations"""
    # Create a dataframe for plotting
    top_locations_df = pd.DataFrame([
        {
            'location': loc['location'],
            'lat': loc['lat'],
            'lon': loc['lon'],
            'count': loc['count']
        } for loc in geocoded_locations.values()
    ])

    # Create scatter mapbox
    fig = px.scatter_mapbox(
        top_locations_df,
        lat="lat",
        lon="lon",
        size="count",
        color="count",
        hover_name="location",
        size_max=25,
        zoom=3,
        mapbox_style="carto-positron",
        title="Crisis Mentions by Location",
        color_continuous_scale=px.colors.sequential.Inferno,
    )

    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Post Count")
    )

    # Save to HTML
    fig.write_html(output_file)
    print(f"Plotly visualization saved to {output_file}")

    return top_locations_df

def analyze_crisis_by_location(df, geocoded_locations):
    """Analyze crisis patterns by location"""
    # Create a mapping of each post to its detected locations
    post_locations = {}
    for idx, row in df.iterrows():
        for loc in row['extracted_locations']:
            if loc in geocoded_locations:
                if idx not in post_locations:
                    post_locations[idx] = []
                post_locations[idx].append(loc)

    # For posts with locations, analyze risk level patterns
    location_risk_levels = {}
    for idx, locations in post_locations.items():
        risk_level = df.loc[idx, 'risk_level']
        for loc in locations:
            if loc not in location_risk_levels:
                location_risk_levels[loc] = []
            location_risk_levels[loc].append(risk_level)

    # Calculate average risk level by location
    location_avg_risk = {}
    for loc, risks in location_risk_levels.items():
        # Convert any risk levels to numeric if they're not already
        numeric_risks = []
        for risk in risks:
            if isinstance(risk, (int, float)):
                numeric_risks.append(risk)
            elif isinstance(risk, str) and risk.replace('.', '', 1).isdigit():
                numeric_risks.append(float(risk))

        if numeric_risks:  # Only calculate if we have valid numeric risk values
            location_avg_risk[loc] = sum(numeric_risks) / len(numeric_risks)

    # Sort and return top locations by average risk
    sorted_locations = sorted(location_avg_risk.items(), key=lambda x: x[1], reverse=True)
    return sorted_locations

def main(data_file_path):
    """Main function to process data and generate visualizations"""
    # Load the dataset
    print(f"Loading data from {data_file_path}")
    df = pd.read_csv(data_file_path)

    print(f"Dataset loaded with {len(df)} posts")
    print(f"Columns: {df.columns.tolist()}")

    # Process dataset to extract and geocode locations
    geocoded_locations, processed_df = process_dataset(df)

    # Create heatmap visualizations
    top_5_locations = create_folium_heatmap(geocoded_locations, output_file="crisis_heatmap_unbiased.html")
    top_locations_df = create_plotly_heatmap(geocoded_locations, processed_df, output_file="crisis_plotly_map_unbiased.html")

    # Analyze crisis patterns by location
    location_risk_analysis = analyze_crisis_by_location(processed_df, geocoded_locations)

    # Print top 5 locations with highest crisis discussions
    print("\nTop 5 locations with highest crisis discussions:")
    for i, (loc_name, loc_data) in enumerate(sorted(
            geocoded_locations.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:5]):
        print(f"{i+1}. {loc_name}: {loc_data['count']} mentions")

    # Print top 5 locations with highest average risk level
    if location_risk_analysis:
        print("\nTop 5 locations with highest average risk level:")
        for i, (loc, avg_risk) in enumerate(location_risk_analysis[:5]):
            print(f"{i+1}. {loc}: {avg_risk:.2f} average risk")

    print("\nAnalysis complete!")
    print("- Crisis heatmap saved as 'crisis_heatmap_unbiased.html'")
    print("- Plotly visualization saved as 'crisis_plotly_map_unbiased.html'")

if __name__=="__main__":
    main(data_file_path='mental_health_postsV1_classified.csv')
    