"""
Simplified visualizer that works without wordcloud dependency
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import re
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleGFTADsVisualizer:
    """Simplified visualization suite for GF-TADs extracted data"""
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """Initialize with either a data file path or DataFrame"""
        if df is not None:
            self.df = df
        elif data_path:
            self.load_data(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.setup_style()
        self.prepare_data()
        
    def load_data(self, data_path: str):
        """Load data from file"""
        path = Path(data_path)
        
        if path.suffix.lower() == '.xlsx':
            self.df = pd.read_excel(data_path)
        elif path.suffix.lower() == '.csv':
            self.df = pd.read_csv(data_path)
        elif path.suffix.lower() == '.json':
            self.df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Use Excel, CSV, or JSON")
    
    def setup_style(self):
        """Set up visualization styles"""
        # Matplotlib/Seaborn style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Plotly theme
        self.plotly_theme = "plotly_white"
        
        # Color schemes
        # Lighter, friendlier palette: light blue, light red-orange, and a pie palette
        self.colors = {
            'blue': '#2471A3',        # Darker blue
            'red': '#E76F51',         # Darker red-orange
            'light_blue': '#7FB3D5',  # Lighter but still visible blue
            'light_red': '#F4A261',   # Lighter but more orange-red
            'gray': '#E5E5E5',
            'dark_gray': '#333333',
            'pie_palette': [
                '#2471A3', '#E76F51', '#F4A261', '#7FB3D5', '#F4A261',
                '#A3D9A5', '#F4978E', '#B5B2C2', '#F6C28B', '#B8E0D2',
                '#F9F871', '#B2A4FF', '#FFB5E8', '#B5EAD7', '#FFDAC1'
            ]
        }
    
    def prepare_data(self):
        """Prepare data for visualization"""
        # Clean and process data
        self.df['meeting_number'] = pd.to_numeric(self.df['meeting_number'], errors='coerce')

        # Ensure all fields are string before concatenation
        for col in ['what', 'when', 'who', 'where', 'impact']:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(lambda x: str(x) if not pd.isna(x) else '')

        # Create combined text fields for analysis
        self.df['all_text'] = (
            self.df['what'].fillna('') + ' '
            + self.df['when'].fillna('') + ' '
            + self.df['who'].fillna('') + ' '
            + self.df['where'].fillna('') + ' '
            + self.df['impact'].fillna('')
        )

        # Extract years from 'when' field
        self.df['year'] = self.df['when'].astype(str).str.extract(r'(\d{4})').astype(float)

        # Process objectives (convert string representation of list to actual list)
        self.df['objectives_processed'] = self.df['objectives'].apply(self.process_objectives)

        # Create activity categories
        self.df['activity_category'] = self.df['what'].apply(self.categorize_activity)
        
    def process_objectives(self, obj_str):
        """Process objectives string into list"""
        if pd.isna(obj_str) or obj_str == '[]':
            return []
        try:
            obj_str = str(obj_str)
            # Try to evaluate as Python list
            if obj_str.startswith('[') and obj_str.endswith(']'):
                return eval(obj_str)
            else:
                # Split by common delimiters
                return [obj.strip() for obj in obj_str.split(';') if obj.strip()]
        except:
            return [str(obj_str)] if obj_str else []
    
    def categorize_activity(self, activity_text):
        """Categorize activities based on keywords"""
        if pd.isna(activity_text):
            return 'Unknown'
        activity_lower = str(activity_text).lower()
        categories = {
            'Capacity Building': ['develop', 'training', 'capacity', 'strengthen', 'enhance'],
            'Surveillance': ['monitor', 'surveillance', 'track', 'observe', 'watch'],
            'Coordination': ['coordinate', 'collaborate', 'partnership', 'alliance'],
            'Prevention': ['prevent', 'preparedness', 'early warning', 'risk'],
            'Response': ['response', 'emergency', 'outbreak', 'crisis'],
            'Research': ['research', 'study', 'investigate', 'analyze'],
            'Policy': ['policy', 'strategy', 'framework', 'guidelines'],
            'Communication': ['communicate', 'inform', 'share', 'disseminate']
        }
        for category, keywords in categories.items():
            if any(keyword in activity_lower for keyword in keywords):
                return category
        return 'Other'
    
    def create_overview_dashboard(self):
        """Create comprehensive overview dashboard"""

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Organizations Involvement',
                'Location Distribution',
                'Disease Distribution',
                'Temporal Distribution',
                'Activities Category',
                ' '
            ],
            specs=[
                [{"type": "bar"}, {"type": "pie"}, None],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]
            ]
        )


        # 1. Organizations Involvement (lighter blue)
        if 'who' in self.df.columns:
            org_expanded = self.df['who'].dropna().astype(str)
            orgs = []
            for val in org_expanded:
                for org in val.split(','):
                    org = org.strip()
                    if org:
                        orgs.append(org)
            org_counts = pd.Series(orgs).value_counts().head(15)
            fig.add_trace(
                go.Bar(
                    x=org_counts.values,
                    y=org_counts.index,
                    orientation='h',
                    name='Organizations Involvement',
                    marker_color=self.colors['blue']
                ),
                row=1, col=1
            )
            fig.update_yaxes(autorange='reversed', row=1, col=1)
        else:
            fig.add_trace(
                go.Bar(x=[], y=[], name='Organizations Involvement', marker_color=self.colors['blue']),
                row=1, col=1
            )

        # 2. Where (Location) Distribution Pie Chart - Top center (distinct palette)
        if 'where' in self.df.columns:
            where_expanded = self.df['where'].dropna().astype(str)
            locations = []
            for val in where_expanded:
                for part in val.split(';'):
                    for loc in part.split(','):
                        loc = loc.strip()
                        if loc:
                            locations.append(loc)
            where_counts = pd.Series(locations).value_counts().head(15)
            pie_colors = self.colors['pie_palette'][:len(where_counts)]
            fig.add_trace(
                go.Pie(
                    labels=where_counts.index,
                    values=where_counts.values,
                    name="Where (Location)",
                    marker_colors=pie_colors,
                    showlegend=True
                ),
                row=1, col=2
            )
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5
                )
            )
        else:
            fig.add_trace(
                go.Pie(labels=[], values=[], name="Where (Location)", marker_colors=self.colors['pie_palette']),
                row=1, col=2
            )

        # 3. Activities by Work Area (even lighter blue) - now in second row, third column
        if 'area_of_work' in self.df.columns:
            area_expanded = self.df['area_of_work'].dropna().astype(str)
            areas = []
            for val in area_expanded:
                for area in val.split(','):
                    area = area.strip()
                    if area:
                        areas.append(area)
            area_counts = pd.Series(areas).value_counts().head(15)
            area_counts = area_counts.sort_values(ascending=False)
            fig.add_trace(
                go.Bar(
                    x=area_counts.values,
                    y=area_counts.index,
                    orientation='h',
                    name='Activities Category',
                    marker_color=self.colors['light_blue']
                ),
                row=2, col=3
            )
            fig.update_yaxes(autorange='reversed', row=2, col=3)
        else:
            fig.add_trace(
                go.Bar(x=[], y=[], name='Activities Category', marker_color=self.colors['light_blue']),
                row=2, col=3
            )

        # 4. Disease Distribution (light red-orange)
        if 'disease' in self.df.columns:
            disease_expanded = self.df['disease'].dropna().astype(str)
            diseases = []
            for val in disease_expanded:
                for disease in val.split(','):
                    disease = disease.strip()
                    if disease:
                        diseases.append(disease)
            disease_counts = pd.Series(diseases).value_counts().head(15)
            fig.add_trace(
                go.Bar(
                    x=disease_counts.values,
                    y=disease_counts.index,
                    orientation='h',
                    marker_color=self.colors['red'],
                    name='Disease Distribution'
                ),
                row=2, col=1
            )
            fig.update_yaxes(autorange='reversed', row=2, col=1)
        else:
            fig.add_trace(
                go.Bar(x=[], y=[], name='Disease Distribution', marker_color=self.colors['red']),
                row=2, col=1
            )

        # 5. Temporal Distribution (lighter blue line, very light orange markers)
        year_data = self.df['year'].dropna()
        if not year_data.empty:
            year_counts = year_data.value_counts().sort_index()
            fig.add_trace(
                go.Scatter(
                    x=year_counts.index,
                    y=year_counts.values,
                    mode='lines+markers',
                    name='Activities by Year',
                    line=dict(color=self.colors['blue'], width=3),
                    marker=dict(size=10, color=self.colors['light_red'])
                ),
                row=2, col=2
            )
        else:
            fig.add_trace(
                go.Scatter(x=[], y=[], name='Activities by Year', line=dict(color=self.colors['blue']), marker=dict(color=self.colors['light_red'])),
                row=2, col=2
            )

        # Add summary metrics at the top (including Meetings held)
        # Count all activities where the word 'meeting' or 'meetings' appears in the 'what' column (case-insensitive)
        if 'what' in self.df.columns:
            meeting_mask = self.df['what'].astype(str).str.contains(r'\bmeetings?\b', case=False, na=False)
            num_meetings_held = int(meeting_mask.sum())
        else:
            num_meetings_held = 0
        self.overview_metrics = {
            'Total Activities': len(self.df),
            'Meetings held': num_meetings_held,
            'Avg Confidence': round(self.df['confidence_score'].mean(), 2),
            'Documents Loaded': self.df['document_type'].nunique()
        }
        print(self.df['what'].unique())
        print('Meetings held:', num_meetings_held)

        # Update layout
        fig.update_layout(
            height=1200,
            title_text="GF-TADs Data Analysis Dashboard",
            title_x=0.5,
            showlegend=False,
            template=self.plotly_theme
        )

        return fig
    
    def create_activity_timeline(self):
        """Create timeline visualization: year (x), number of activities (y), colored by area_of_work"""
        # Filter data with valid years (exclude NaN)
        timeline_data = self.df[self.df['year'].notna()].copy()
        # Remove any rows where 'year' is still NaN (shouldn't be, but extra safety)
        timeline_data = timeline_data[~timeline_data['year'].isna()]
        if timeline_data.empty:
            print("No temporal data available for timeline")
            return None

        # Ensure area_of_work is string for grouping
        if 'area_of_work' in timeline_data.columns:
            timeline_data['area_of_work'] = timeline_data['area_of_work'].astype(str)
        else:
            timeline_data['area_of_work'] = 'Unknown'

        # Group by year and area_of_work, count activities
        grouped = timeline_data.groupby(['year', 'area_of_work']).size().reset_index(name='activity_count')

        # Remove any rows where year is NaN (shouldn't be needed, but double check)
        grouped = grouped[~grouped['year'].isna()]

        fig = px.bar(
            grouped,
            x='year',
            y='activity_count',
            color='area_of_work',
            barmode='group',
            title='Activities per Year by Area of Work',
            labels={'year': 'Year', 'activity_count': 'Number of Activities', 'area_of_work': 'Area of Work'},
            template=self.plotly_theme
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Activities",
            height=600
        )
        return fig
    
    def create_objectives_analysis(self):
        """Analyze and visualize objectives"""
        # Flatten all objectives
        all_objectives = []
        for obj_list in self.df['objectives_processed']:
            all_objectives.extend(obj_list)
        
        if not all_objectives:
            print("No objectives data available")
            return None
        
        # Count objectives
        obj_counts = Counter(all_objectives).most_common(15)
        
        if not obj_counts:
            return None
        
        objectives, counts = zip(*obj_counts)
        
        fig = go.Figure(data=[
            go.Bar(
                y=objectives,
                x=counts,
                orientation='h',
                marker_color=px.colors.qualitative.Set3[:len(objectives)]
            )
        ])
        
        fig.update_layout(
            title='Most Common Objectives',
            xaxis_title='Frequency',
            yaxis_title='Objectives',
            height=600,
            template=self.plotly_theme
        )
        
        return fig
    
    def create_confidence_analysis(self):
        """Analyze confidence scores across different dimensions"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Confidence by Document Type',
                'Confidence by Meeting Number',
                'Confidence by Activity Category',
                'Confidence Score Distribution'
            ]
        )
        
        # 1. Confidence by Document Type
        conf_by_doctype = self.df.groupby('document_type')['confidence_score'].mean()
        fig.add_trace(
            go.Bar(
                x=conf_by_doctype.index,
                y=conf_by_doctype.values,
                name='Avg Confidence by Doc Type',
                marker_color=self.colors['blue']
            ),
            row=1, col=1
        )
        
        # 2. Confidence by Meeting Number
        conf_by_meeting = self.df.groupby('meeting_number')['confidence_score'].mean()
        fig.add_trace(
            go.Scatter(
                x=conf_by_meeting.index,
                y=conf_by_meeting.values,
                mode='lines+markers',
                name='Avg Confidence by Meeting',
                line=dict(color=self.colors['red'])
            ),
            row=1, col=2
        )
        
        # 3. Confidence by Activity Category
        conf_by_category = self.df.groupby('activity_category')['confidence_score'].mean().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(
                y=conf_by_category.index,
                x=conf_by_category.values,
                orientation='h',
                name='Avg Confidence by Category',
                marker_color=self.colors['light_blue']
            ),
            row=2, col=1
        )
        
        # 4. Distribution
        fig.add_trace(
            go.Box(
                y=self.df['confidence_score'],
                name='Confidence Distribution',
                marker_color=self.colors['light_blue']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Confidence Score Analysis",
            title_x=0.5,
            showlegend=False,
            template=self.plotly_theme
        )
        
        return fig
    
    def create_wordclouds(self):
        """Create simple text analysis without wordcloud dependency"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Text Analysis (Top Words)', fontsize=16, fontweight='bold')
        
        # What (Activities) - Top words
        what_text = ' '.join(self.df['what'].dropna().astype(str))
        what_words = Counter(what_text.lower().split()).most_common(10)
        if what_words:
            words, counts = zip(*what_words)
            axes[0, 0].barh(words, counts)
            axes[0, 0].set_title('Top Words in Activities (What)', fontweight='bold')
        
        # Who (Organizations) - Top words
        who_text = ' '.join(self.df['who'].dropna().astype(str))
        who_words = Counter(who_text.lower().split()).most_common(10)
        if who_words:
            words, counts = zip(*who_words)
            axes[0, 1].barh(words, counts)
            axes[0, 1].set_title('Top Words in Organizations (Who)', fontweight='bold')
        
        # Where (Locations) - Top words
        where_text = ' '.join(self.df['where'].dropna().astype(str))
        where_words = Counter(where_text.lower().split()).most_common(10)
        if where_words:
            words, counts = zip(*where_words)
            axes[1, 0].barh(words, counts)
            axes[1, 0].set_title('Top Words in Locations (Where)', fontweight='bold')
        
        # Impact - Top words
        impact_text = ' '.join(self.df['impact'].dropna().astype(str))
        impact_words = Counter(impact_text.lower().split()).most_common(10)
        if impact_words:
            words, counts = zip(*impact_words)
            axes[1, 1].barh(words, counts)
            axes[1, 1].set_title('Top Words in Impact', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        report = {
            'total_activities': len(self.df),
            'unique_meetings': self.df['meeting_number'].nunique(),
            'avg_confidence': self.df['confidence_score'].mean(),
            'document_types': self.df['document_type'].value_counts().to_dict(),
            'activity_categories': self.df['activity_category'].value_counts().to_dict(),
            'top_organizations': {},
            'top_locations': {},
            'temporal_range': {},
            'most_common_objectives': {}
        }
        
        # Top organizations
        all_orgs = []
        for orgs in self.df['who'].dropna():
            all_orgs.extend([org.strip() for org in str(orgs).split(';') if org.strip()])
        report['top_organizations'] = dict(Counter(all_orgs).most_common(10))
        
        # Top locations
        all_locations = []
        for locs in self.df['where'].dropna():
            all_locations.extend([loc.strip() for loc in str(locs).split(';') if loc.strip()])
        report['top_locations'] = dict(Counter(all_locations).most_common(10))
        
        # Temporal range
        years = self.df['year'].dropna()
        if not years.empty:
            report['temporal_range'] = {
                'min_year': int(years.min()),
                'max_year': int(years.max()),
                'year_distribution': years.value_counts().to_dict()
            }
        
        # Most common objectives
        all_objectives = []
        for obj_list in self.df['objectives_processed']:
            all_objectives.extend(obj_list)
        report['most_common_objectives'] = dict(Counter(all_objectives).most_common(10))
        
        return report

# For backward compatibility
GFTADsVisualizer = SimpleGFTADsVisualizer

if __name__ == "__main__":
    print("Simplified visualization module ready!")
