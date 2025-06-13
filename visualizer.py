import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

try:
    from wordcloud import WordCloud
except ImportError:
    print("Installing wordcloud...")
    import os
    os.system("pip install wordcloud")
    from wordcloud import WordCloud

from collections import Counter
import re
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GFTADsVisualizer:
    """Comprehensive visualization suite for GF-TADs extracted data"""
    
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
        plt.style.use('seaborn-v0_8')
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
        
        # Create combined text fields for analysis
        self.df['all_text'] = (
            self.df['what'].fillna('') + ' ' +
            self.df['when'].fillna('') + ' ' +
            self.df['who'].fillna('') + ' ' +
            self.df['where'].fillna('') + ' ' +
            self.df['impact'].fillna('')
        )
        
        # Extract years from 'when' field
        self.df['year'] = self.df['when'].str.extract(r'(\d{4})').astype(float)
        
        # Process objectives (convert string representation of list to actual list)
        self.df['objectives_processed'] = self.df['objectives'].apply(self.process_objectives)
        
        # Create activity categories
        self.df['activity_category'] = self.df['what'].apply(self.categorize_activity)
        
    def process_objectives(self, obj_str):
        """Process objectives string into list"""
        if pd.isna(obj_str) or obj_str == '[]':
            return []
        
        try:
            # Try to evaluate as Python list
            if obj_str.startswith('[') and obj_str.endswith(']'):
                return eval(obj_str)
            else:
                # Split by common delimiters
                return [obj.strip() for obj in obj_str.split(';') if obj.strip()]
        except:
            return [obj_str] if obj_str else []
    
    def categorize_activity(self, activity_text):
        """Categorize activities based on keywords"""
        if pd.isna(activity_text):
            return 'Unknown'
        
        activity_lower = activity_text.lower()
        
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
            rows=3, cols=2,
            subplot_titles=[
                'Activities by Meeting Number',
                'Document Types Distribution',
                'Activity Categories',
                'Confidence Score Distribution',
                'Organizations Involvement',
                'Temporal Distribution'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Activities by Meeting Number
        meeting_counts = self.df['meeting_number'].value_counts().sort_index()
        fig.add_trace(
            go.Scatter(
                x=meeting_counts.index,
                y=meeting_counts.values,
                mode='lines+markers',
                name='Activities per Meeting',
                line=dict(color=self.colors['primary'], width=3)
            ),
            row=1, col=1
        )
        
        # 2. Document Types Distribution
        doc_types = self.df['document_type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=doc_types.index,
                values=doc_types.values,
                name="Document Types",
                marker_colors=[self.colors['primary'], self.colors['secondary']]
            ),
            row=1, col=2
        )
        
        # 3. Activity Categories
        categories = self.df['activity_category'].value_counts()
        fig.add_trace(
            go.Bar(
                x=categories.values,
                y=categories.index,
                orientation='h',
                name='Activity Categories',
                marker_color=self.colors['success']
            ),
            row=2, col=1
        )
        
        # 4. Confidence Score Distribution
        fig.add_trace(
            go.Histogram(
                x=self.df['confidence_score'],
                nbinsx=20,
                name='Confidence Scores',
                marker_color=self.colors['info']
            ),
            row=2, col=2
        )
        
        # 5. Organizations (top 10)
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
                row=3, col=1
            )
        
        # 6. Temporal Distribution (if year data available)
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
                row=3, col=2
            )
        
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
        """Create timeline visualization of activities"""
        # Filter data with valid years
        timeline_data = self.df[self.df['year'].notna()]
        
        if timeline_data.empty:
            print("No temporal data available for timeline")
            return None
        
        fig = px.scatter(
            timeline_data,
            x='year',
            y='meeting_number',
            size='confidence_score',
            color='activity_category',
            hover_data=['what', 'who', 'where'],
            title='Activity Timeline: When and What',
            template=self.plotly_theme
        )
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Meeting Number",
            height=600
        )
        
        return fig
    
    def create_network_analysis(self):
        """Create network analysis of relationships"""
        # This is a simplified network - in practice, you might want to use networkx
        fig = go.Figure()
        
        # Create a matrix of co-occurrences between organizations and activities
        org_activity_matrix = {}
        
        for idx, row in self.df.iterrows():
            orgs = [org.strip() for org in str(row['who']).split(';') if org.strip()]
            activities = [act.strip() for act in str(row['what']).split(';') if act.strip()]
            
            for org in orgs:
                if org not in org_activity_matrix:
                    org_activity_matrix[org] = {}
                for activity in activities:
                    if activity not in org_activity_matrix[org]:
                        org_activity_matrix[org][activity] = 0
                    org_activity_matrix[org][activity] += 1
        
        # Create a simple chord-like visualization
        fig.update_layout(
            title="Organization-Activity Network (Placeholder)",
            template=self.plotly_theme,
            height=600
        )
        
        return fig
    
    def create_wordclouds(self):
        """Create word clouds for different aspects"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Word Clouds Analysis', fontsize=16, fontweight='bold')
        
        # What (Activities)
        what_text = ' '.join(self.df['what'].dropna().astype(str))
        if what_text.strip():
            wordcloud_what = WordCloud(width=400, height=300, background_color='white').generate(what_text)
            axes[0, 0].imshow(wordcloud_what, interpolation='bilinear')
            axes[0, 0].set_title('Activities (What)', fontweight='bold')
            axes[0, 0].axis('off')
        
        # Who (Organizations)
        who_text = ' '.join(self.df['who'].dropna().astype(str))
        if who_text.strip():
            wordcloud_who = WordCloud(width=400, height=300, background_color='white').generate(who_text)
            axes[0, 1].imshow(wordcloud_who, interpolation='bilinear')
            axes[0, 1].set_title('Organizations (Who)', fontweight='bold')
            axes[0, 1].axis('off')
        
        # Where (Locations)
        where_text = ' '.join(self.df['where'].dropna().astype(str))
        if where_text.strip():
            wordcloud_where = WordCloud(width=400, height=300, background_color='white').generate(where_text)
            axes[1, 0].imshow(wordcloud_where, interpolation='bilinear')
            axes[1, 0].set_title('Locations (Where)', fontweight='bold')
            axes[1, 0].axis('off')
        
        # Impact
        impact_text = ' '.join(self.df['impact'].dropna().astype(str))
        if impact_text.strip():
            wordcloud_impact = WordCloud(width=400, height=300, background_color='white').generate(impact_text)
            axes[1, 1].imshow(wordcloud_impact, interpolation='bilinear')
            axes[1, 1].set_title('Impact', fontweight='bold')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
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
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        report = {
            'total_activities': len(self.df),
            'unique_meetings': self.df['meeting_number'].nunique(),
            'avg_confidence': self.df['confidence_score'].mean(),
            'document_types': self.df['document_type'].value_counts().to_dict(),
            'activity_categories': self.df['activity_category'].value_counts().to_dict(),
            'top_organizations': [],
            'top_locations': [],
            'temporal_range': {},
            'most_common_objectives': []
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
    
    def save_visualizations(self, output_dir: str = None):
        """Save all visualizations to files"""
        if output_dir is None:
            output_dir = Path('visualizations')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save interactive plots
        plots = {
            'overview_dashboard': self.create_overview_dashboard(),
            'activity_timeline': self.create_activity_timeline(),
            'objectives_analysis': self.create_objectives_analysis(),
            'confidence_analysis': self.create_confidence_analysis()
        }
        
        for name, fig in plots.items():
            if fig is not None:
                fig.write_html(output_dir / f"{name}_{timestamp}.html")
        
        # Save word clouds
        wordcloud_fig = self.create_wordclouds()
        wordcloud_fig.savefig(output_dir / f"wordclouds_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close(wordcloud_fig)
        
        # Save summary report
        report = self.generate_summary_report()
        with open(output_dir / f"summary_report_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Visualizations saved to: {output_dir}")
        
        return output_dir

if __name__ == "__main__":
    # Example usage
    # Assuming you have extracted data
    
    # Create visualizer (you would replace this with your actual data file)
    # visualizer = GFTADsVisualizer("path/to/your/extracted_data.xlsx")
    
    # Create and show individual visualizations
    # fig1 = visualizer.create_overview_dashboard()
    # fig1.show()
    
    # Save all visualizations
    # visualizer.save_visualizations()
    
    print("Visualization module ready!")
