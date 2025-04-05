
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


"""This script generates various visualizations for the crisis terms dataset: crisis_terms_bert.csv.
We are analyzing the risk levels and sentiment scores over time, as well as the distribution of these metrics.

Consclusion infered via plots:
 - Their is more data for the recent year, so we can't infer that high-risk posts are increasing over year.
 - The sentiment score is not a good indicator of risk level. Needs to study their methods relevance.
 - There is a clear seasonal pattern in the risk levels, with slightly more high-risk-levels in later second-half of year."""

df = pd.read_csv("crisis_terms_bert.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract year and month for analysis
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['year_month'] = df['timestamp'].dt.strftime('%Y-%m')

# 1. Monthly Average Risk Level Analysis
# Convert risk_level to numeric for averaging
risk_map = {'Low Concern': 1, 'Moderate Concern': 2, 'High-Risk': 3, 'Unclassified': 0}
df['risk_numeric'] = df['risk_level'].map(risk_map)

# Group by year and month
monthly_risk = df.groupby('year_month').agg({
    'risk_numeric': 'mean',
    'sentiment_score': 'mean',
    'timestamp': 'first'  # Keep first timestamp for proper ordering
}).reset_index()

# Sort by timestamp
monthly_risk = monthly_risk.sort_values('timestamp')

# 2. Risk Level Distribution by Month and Year
risk_counts_monthly = df.groupby(['year_month', 'risk_level']).size().reset_index(name='count')
risk_counts_yearly = df.groupby(['year', 'risk_level']).size().reset_index(name='count')

# 3. Sentiment Analysis Over Time
monthly_sentiment = df.groupby('year_month').agg({
    'sentiment_score': 'mean',
    'timestamp': 'first'
}).reset_index().sort_values('timestamp')

sentiment_counts_monthly = df.groupby(['year_month', 'sentiment']).size().reset_index(name='count')
sentiment_counts_yearly = df.groupby(['year', 'sentiment']).size().reset_index(name='count')

# 4. Combined Risk and Sentiment Analysis
combined_monthly = df.groupby('year_month').agg({
    'risk_numeric': 'mean',
    'sentiment_score': 'mean',
    'timestamp': 'first'
}).reset_index().sort_values('timestamp')

# 5. High Risk Terms Analysis
df['has_high_risk_terms'] = df['high_risk_terms'].apply(lambda x: 1 if len(x) > 0 else 0)
high_risk_monthly = df.groupby('year_month').agg({
    'has_high_risk_terms': 'sum',
    'timestamp': 'first'
}).reset_index().sort_values('timestamp')

# Create visualizations

# 1. Monthly Average Risk Level Trend
fig1 = px.line(monthly_risk, x='year_month', y='risk_numeric', 
              title='Average Risk Level Over Time',
              labels={'risk_numeric': 'Risk Level (1=Low, 4=Critical)', 'year_month': 'Month'})

fig1.update_layout(xaxis_tickangle=-45)

# 2. Risk Level Distribution Stacked Area Chart
fig2 = px.area(risk_counts_monthly, x='year_month', y='count', color='risk_level',
              title='Risk Level Distribution by Month',
              labels={'count': 'Number of Posts', 'year_month': 'Month', 'risk_level': 'Risk Level'},
              color_discrete_map={'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'})

fig2.update_layout(xaxis_tickangle=-45)

# 3. Sentiment Score Trend
fig3 = px.line(monthly_sentiment, x='year_month', y='sentiment_score',
              title='Average Sentiment Score Over Time',
              labels={'sentiment_score': 'Sentiment Score (-1 to 1)', 'year_month': 'Month'})

fig3.update_layout(xaxis_tickangle=-45)

# 4. Sentiment Distribution
fig4 = px.area(sentiment_counts_monthly, x='year_month', y='count', color='sentiment',
              title='Sentiment Distribution by Month',
              labels={'count': 'Number of Posts', 'year_month': 'Month', 'sentiment': 'Sentiment'},
              color_discrete_map={'negative': 'red', 'neutral': 'gray', 'positive': 'green'})

fig4.update_layout(xaxis_tickangle=-45)

# 5. Combined Risk and Sentiment by Month
fig5 = make_subplots(specs=[[{"secondary_y": True}]])

fig5.add_trace(
    go.Scatter(x=combined_monthly['year_month'], y=combined_monthly['risk_numeric'], name="Risk Level"),
    secondary_y=False,
)

fig5.add_trace(
    go.Scatter(x=combined_monthly['year_month'], y=combined_monthly['sentiment_score'], name="Sentiment Score"),
    secondary_y=True,
)

fig5.update_layout(
    title_text="Risk Level vs Sentiment Score Over Time",
    xaxis_tickangle=-45
)

fig5.update_yaxes(title_text="Risk Level (1-4)", secondary_y=False)
fig5.update_yaxes(title_text="Sentiment Score (-1 to 1)", secondary_y=True)

# 6. Yearly Risk Level Distribution
fig6 = px.bar(risk_counts_yearly, x='year', y='count', color='risk_level', barmode='group',
             title='Risk Level Distribution by Year',
             labels={'count': 'Number of Posts', 'year': 'Year', 'risk_level': 'Risk Level'},
             color_discrete_map={'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'})

# 7. Yearly Sentiment Distribution
fig7 = px.bar(sentiment_counts_yearly, x='year', y='count', color='sentiment', barmode='group',
             title='Sentiment Distribution by Year',
             labels={'count': 'Number of Posts', 'year': 'Year', 'sentiment': 'Sentiment'},
             color_discrete_map={'negative': 'red', 'neutral': 'gray', 'positive': 'green'})

# 8. High Risk Terms Occurrence Over Time
fig8 = px.bar(high_risk_monthly, x='year_month', y='has_high_risk_terms',
             title='Posts with High Risk Terms Over Time',
             labels={'has_high_risk_terms': 'Number of Posts', 'year_month': 'Month'})

fig8.update_layout(xaxis_tickangle=-45)

# Display the figures
# fig1.show()
# fig2.show()
# fig3.show()
# fig4.show()
# fig5.show()
# fig6.show()
# fig7.show()
# fig8.show()

# Create a dashboard layout
dashboard_fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        "Average Risk Level Over Time",
        "Risk Level Distribution by Month",
        "Average Sentiment Score Over Time",
        "Sentiment Distribution by Month",
        "Risk Level vs Sentiment Score Over Time",
        "Risk Level Distribution by Year",
        "Sentiment Distribution by Year",
        "Posts with High Risk Terms Over Time"
    ),
    vertical_spacing=0.1,
    horizontal_spacing=0.1
)

# Add each figure as a subplot
dashboard_fig.add_traces(fig1.data, rows=1, cols=1)
dashboard_fig.add_traces(fig2.data, rows=1, cols=2)
dashboard_fig.add_traces(fig3.data, rows=2, cols=1)
dashboard_fig.add_traces(fig4.data, rows=2, cols=2)
dashboard_fig.add_traces(fig5.data, rows=3, cols=1)
dashboard_fig.add_traces(fig6.data, rows=3, cols=2)
dashboard_fig.add_traces(fig7.data, rows=4, cols=1)
dashboard_fig.add_traces(fig8.data, rows=4, cols=2)

# Update layout for better visualization
dashboard_fig.update_layout(
    height=1200, width=1800,
    title_text="Dashboard: Risk and Sentiment Analysis",
    showlegend=False
)

# Display the dashboard
dashboard_fig.write_image("dashboard.png", scale=2)
dashboard_fig.write_html("dashboard.html", include_plotlyjs='cdn')

def create_risk_heatmap(df):
    
    # Ensure risk_level is properly converted to numeric
    if df['risk_level'].dtype == 'object':
        risk_map = {'Low Concern': 1, 'Moderate Concern': 2, 'High-Risk': 3, 'Unclassified': 0}
        df['risk_numeric'] = df['risk_level'].map(risk_map)

    # Extract numeric month for proper ordering
    df['month_num'] = df['timestamp'].dt.month

    # Create pivot table with proper ordering
    heatmap_data = df.pivot_table(
        values='risk_numeric', 
        index='month_num',  # Use numeric month for correct ordering
        columns='year', 
        aggfunc='mean'
    ).round(2)

    # Create month labels for y-axis
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    # Sort by month number to ensure correct order
    heatmap_data = heatmap_data.sort_index()

    # Create the heatmap
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Month", color="Avg Risk Level"),
        x=heatmap_data.columns,
        y=[month_names[i] for i in heatmap_data.index],
        title="Monthly Risk Level Heatmap",
        color_continuous_scale="RdYlGn_r"
    )

    # Add custom layout
    fig_heatmap.update_layout(
        height=600,
        width=800,
        xaxis={'side': 'bottom'},
        xaxis_title="Year",
        yaxis_title="Month"
    )

    # Fix axis display issues
    fig_heatmap.update_xaxes(tickangle=0)

    fig_heatmap.write_image("monthly_risk_heatmap.png", scale=2)

    return fig_heatmap

def seasonal_analysis(df):
    if df['risk_level'].dtype == 'object':
        risk_map = {'Low Concern': 1, 'Moderate Concern': 2, 'High-Risk': 3, 'Unclassified': 0}
        df['risk_numeric'] = df['risk_level'].map(risk_map)

    # Add season column with better ordering
    season_order = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
    df['season'] = df['timestamp'].dt.month.map({
        1: 'Winter', 2: 'Winter', 3: 'Spring', 
        4: 'Spring', 5: 'Spring', 6: 'Summer',
        7: 'Summer', 8: 'Summer', 9: 'Fall',
        10: 'Fall', 11: 'Fall', 12: 'Winter'
    })

    seasonal_risk = df.groupby(['year', 'season']).agg({
        'risk_numeric': 'mean'
    }).reset_index()

    fig_seasonal = px.line(
        seasonal_risk, 
        x='season', 
        y='risk_numeric', 
        color='year',
        title='Seasonal Risk Level Analysis',
        labels={'risk_numeric': 'Average Risk Level', 'season': 'Season'},
        category_orders={"season": ["Winter", "Spring", "Summer", "Fall"]}
    )

    fig_seasonal.update_layout(
        height=600,
        width=900,
        legend_title="Year",
        yaxis_range=[1, 4]
    )

    fig_seasonal.write_image("seasonal_risk_analysis.png", scale=2)

    return fig_seasonal

fig_seasonal = seasonal_analysis(df)
fig_heatmap = create_risk_heatmap(df)
