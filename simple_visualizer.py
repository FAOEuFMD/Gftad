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
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#8c564b',
            'dark': '#e377c2'
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
                'Activities by Work Area',
                'Where (Location) Distribution',
                'Activity Categories',
                'Confidence Score Distribution',
                'Organizations Involvement',
                'Temporal Distribution'
            ],
            specs=[
                [{"type": "bar"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "bar"}, {"type": "scatter"}]
            ]
        )


        # 1. Activities by Work Area (area_of_work) - Top left
        if 'area_of_work' in self.df.columns:
            area_expanded = self.df['area_of_work'].dropna().astype(str).str.split(';').explode().str.strip()
            area_counts = area_expanded[area_expanded != ''].value_counts()
            fig.add_trace(
                go.Bar(
                    x=area_counts.index,
                    y=area_counts.values,
                    name='Activities by Work Area',
                    marker_color=self.colors['primary']
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Bar(x=[], y=[], name='Activities by Work Area'),
                row=1, col=1
            )

        # 2. Where (Location) Distribution Pie Chart - Top center
        if 'where' in self.df.columns:
            where_expanded = self.df['where'].dropna().astype(str).str.split(';').explode().str.strip()
            where_counts = where_expanded[where_expanded != ''].value_counts().head(15)
            fig.add_trace(
                go.Pie(
                    labels=where_counts.index,
                    values=where_counts.values,
                    name="Where (Location)",
                    marker_colors=px.colors.qualitative.Pastel
                ),
                row=1, col=2
            )
        else:
            fig.add_trace(
                go.Pie(labels=[], values=[], name="Where (Location)"),
                row=1, col=2
            )

        # 3. Activity Categories - Top right
        categories = self.df['activity_category'].value_counts()
        fig.add_trace(
            go.Bar(
                x=categories.index,
                y=categories.values,
                name='Activity Categories',
                marker_color=self.colors['success']
            ),
            row=1, col=3
        )

        # 4. Confidence Score Distribution - Bottom left
        fig.add_trace(
            go.Histogram(
                x=self.df['confidence_score'],
                nbinsx=20,
                name='Confidence Scores',
                marker_color=self.colors['info']
            ),
            row=2, col=1
        )

        # 5. Organizations (top 10) - Bottom center
        all_orgs = []
        for orgs in self.df['who'].dropna():
            all_orgs.extend([org.strip() for org in str(orgs).split(';') if org.strip()])
        org_counts = Counter(all_orgs).most_common(10)
        if org_counts:
            orgs, counts = zip(*org_counts)
            fig.add_trace(
                go.Bar(
                    x=counts,
                    y=orgs,
                    orientation='h',
                    name='Top Organizations',
                    marker_color=self.colors['warning']
                ),
                row=2, col=2
            )
        else:
            fig.add_trace(
                go.Bar(x=[], y=[], name='Top Organizations'),
                row=2, col=2
            )

        # 6. Temporal Distribution (if year data available) - Bottom right
        year_data = self.df['year'].dropna()
        if not year_data.empty:
            year_counts = year_data.value_counts().sort_index()
            fig.add_trace(
                go.Scatter(
                    x=year_counts.index,
                    y=year_counts.values,
                    mode='markers+lines',
                    name='Activities by Year',
                    marker=dict(size=10, color=self.colors['dark'])
                ),
                row=2, col=3
            )
        else:
            fig.add_trace(
                go.Scatter(x=[], y=[], name='Activities by Year'),
                row=2, col=3
            )

        # Add summary metrics at the top (including Number of Global Steering Committee meetings)
        # This is for Streamlit, not Plotly, so add a helper method for Streamlit to call
        # Fix: drop NaN and empty, convert to string, then count unique meeting numbers
        meeting_numbers = self.df['meeting_number'].dropna().astype(str)
        meeting_numbers = meeting_numbers[meeting_numbers != '' ]
        num_unique_meetings = meeting_numbers.nunique()
        self.overview_metrics = {
            'Total Activities': len(self.df),
            'Number of Global Steering Committee meetings': num_unique_meetings,
            'Avg Confidence': round(self.df['confidence_score'].mean(), 2),
            'Document Types': self.df['document_type'].nunique()
        }

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
        # Filter data with valid years
        timeline_data = self.df[self.df['year'].notna()].copy()
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
                marker_color=self.colors['primary']
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
                line=dict(color=self.colors['secondary'])
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
                marker_color=self.colors['success']
            ),
            row=2, col=1
        )
        
        # 4. Distribution
        fig.add_trace(
            go.Box(
                y=self.df['confidence_score'],
                name='Confidence Distribution',
                marker_color=self.colors['info']
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
