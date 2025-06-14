import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="GF-TADs Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our custom modules with error handling
EXTRACTOR_AVAILABLE = False
VISUALIZER_AVAILABLE = False

try:
    from simple_extractor import SimpleGFTADsDataExtractor as GFTADsDataExtractor
    EXTRACTOR_AVAILABLE = True
except ImportError as e:
    try:
        from data_extractor import GFTADsDataExtractor
        EXTRACTOR_AVAILABLE = True
    except ImportError:
        pass  # Will show error in the UI later

try:
    from simple_visualizer import SimpleGFTADsVisualizer as GFTADsVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError as e:
    try:
        from visualizer import GFTADsVisualizer
        VISUALIZER_AVAILABLE = True
    except ImportError:
        pass  # Will show error in the UI later

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .highlight {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    """Load data with caching"""
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        st.error("Unsupported file format")
        return None

def main():
    st.title("🎯 GF-TADs Data Analysis Dashboard")
    st.markdown("**Comprehensive analysis of Global Framework for the Progressive Control of Transboundary Animal Diseases documents**")
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    
    # Option to extract new data or load existing
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode:",
        ["Extract New Data", "Load Existing Data", "Demo Mode"]
    )
    
    if analysis_mode == "Extract New Data":
        extract_new_data()
    elif analysis_mode == "Load Existing Data":
        load_existing_data()
    else:
        demo_mode()

def extract_new_data():
    st.header("🔄 Extract New Data from PDFs")
    
    st.markdown("""
    <div class="highlight">
    <strong>📁 Data Extraction Process</strong><br>
    This will process all PDF files in your GSC_Recommendations and GSC_Reports folders
    and extract structured information including what, when, who, where, impact, and objectives.    </div>
    """, unsafe_allow_html=True)
    
    # Path configuration
    base_path = st.text_input(
        "Base Path to GF-TADs Data:",
        value=r"c:\Users\user\EUFMD\Gftad",
        help="Path to the folder containing GSC_Recommendations and GSC_Reports"
    )
    
    if st.button("🚀 Start Data Extraction", type="primary"):
        if not EXTRACTOR_AVAILABLE:
            st.error("❌ Data extractor module is not available. Please check the installation.")
            return
            
        if not Path(base_path).exists():
            st.error("❌ Path does not exist. Please check the path.")
            return
        
        # Check for required folders
        recommendations_path = Path(base_path) / "GSC_Recommendations"
        reports_path = Path(base_path) / "GSC_Reports"
        
        if not recommendations_path.exists() or not reports_path.exists():
            st.error("❌ GSC_Recommendations or GSC_Reports folders not found in the specified path.")
            return
        
        # Start extraction
        with st.spinner("🔍 Extracting data from PDFs... This may take a few minutes."):
            try:
                extractor = GFTADsDataExtractor(base_path)
                df = extractor.process_all_documents()
                
                if df.empty:
                    st.error("❌ No data could be extracted from the PDFs.")
                    return
                
                # Save data
                output_file = extractor.save_extracted_data(df, 'excel')
                
                st.success(f"✅ Data extraction completed successfully!")
                st.info(f"📄 Total activities extracted: {len(df)}")
                st.info(f"💾 Data saved to: {output_file}")
                  # Store in session state for immediate analysis
                st.session_state['extracted_df'] = df
                st.session_state['data_source'] = str(output_file)
                
                # Show preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10))
                
                # Guide user to next step with enhanced instructions
                st.markdown(f"""
                <div class="highlight">
                <strong>🎉 Extraction Complete!</strong><br>
                Your data has been successfully extracted and saved. To analyze and visualize this data:
                <ol>
                <li>📂 Click on <strong>"Load Existing Data"</strong> in the sidebar</li>
                <li>📎 Browse and select the Excel file:<br>
                    <code>{output_file.name}</code></li>
                <li>📊 The analysis will start automatically once the file is uploaded!</li>
                </ol>
                <br>
                <strong>💡 Tip:</strong> The file is saved in your <code>extracted_data</code> folder.
                </div>
                """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error during extraction: {str(e)}")

def load_existing_data():
    st.header("📂 Load Existing Data")
    
    st.markdown("""
    <div class="highlight">
    <strong>📊 Load Previously Extracted Data</strong><br>
    Upload an Excel, CSV, or JSON file containing previously extracted GF-TADs data.
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=['xlsx', 'csv', 'json'],
        help="Upload Excel, CSV, or JSON file with extracted GF-TADs data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data based on file type
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.success(f"✅ Data loaded successfully! ({len(df)} records)")
            
            # Store in session state
            st.session_state['extracted_df'] = df
            st.session_state['data_source'] = uploaded_file.name
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10))
            
            # Proceed to analysis
            analyze_data(df)
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")

def demo_mode():
    st.header("🎭 Demo Mode")
    
    st.markdown("""
    <div class="highlight">
    <strong>🎯 Demo Mode</strong><br>
    This mode shows you what the analysis would look like with sample data.
    To analyze your actual GF-TADs documents, use "Extract New Data" mode.
    </div>
    """, unsafe_allow_html=True)
    
    # Create sample data for demonstration
    sample_data = create_sample_data()
    
    st.subheader("📊 Sample Data Structure")
    st.dataframe(sample_data.head(10))
    
    if st.button("📈 View Demo Analysis"):
        analyze_data(sample_data)

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    sample_activities = [
        "Develop surveillance protocols",
        "Implement early warning systems",
        "Strengthen veterinary services",
        "Coordinate regional response",
        "Enhance laboratory capacity",
        "Monitor disease outbreaks",
        "Establish partnerships",
        "Improve information sharing"
    ]
    
    sample_orgs = ["FAO", "OIE", "WHO", "GF-TADs", "Regional Commission", "Ministry of Agriculture"]
    sample_locations = ["Africa", "Asia", "Europe", "Regional", "National", "Global"]
    sample_objectives = ["capacity building", "surveillance", "preparedness", "coordination", "prevention"]
    
    n_samples = 100
    
    data = {
        'what': np.random.choice(sample_activities, n_samples),
        'when': np.random.choice(['2022', '2023', '2024', 'by 2025', 'annually', 'quarterly'], n_samples),
        'who': np.random.choice(sample_orgs, n_samples),
        'where': np.random.choice(sample_locations, n_samples),
        'impact': np.random.choice(['improve response time', 'reduce disease spread', 'enhance coordination'], n_samples),
        'objectives': [np.random.choice(sample_objectives, np.random.randint(1, 4)).tolist() for _ in range(n_samples)],
        'meeting_number': np.random.choice(range(1, 15), n_samples),
        'document_type': np.random.choice(['Recommendations', 'Reports'], n_samples),
        'confidence_score': np.random.beta(2, 2, n_samples),  # Beta distribution for realistic confidence scores
        'page_number': np.random.randint(1, 20, n_samples)
    }
    
    return pd.DataFrame(data)

def analyze_data(df):
    """Main analysis function"""
    st.header("📊 Data Analysis")
    
    # Check if visualizer is available
    if not VISUALIZER_AVAILABLE:
        st.error("❌ Visualizer module is not available. Showing basic analysis only.")
        
        # Show basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Activities", len(df))
        
        with col2:
            st.metric("Unique Meetings", df['meeting_number'].nunique())
        
        with col3:
            st.metric("Avg Confidence", f"{df['confidence_score'].mean():.2f}")
        
        with col4:
            st.metric("Document Types", df['document_type'].nunique())
        
        # Show data table
        st.subheader("📊 Data Table")
        st.dataframe(df)
        return
    
    # Create visualizer
    visualizer = GFTADsVisualizer(df=df)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Activities", len(df))
    
    with col2:
        st.metric("Unique Meetings", df['meeting_number'].nunique())
    
    with col3:
        st.metric("Avg Confidence", f"{df['confidence_score'].mean():.2f}")
    
    with col4:
        st.metric("Document Types", df['document_type'].nunique())
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "⏰ Timeline", "🎯 Objectives", "📈 Confidence", "☁️ Word Analysis", "📋 Summary"
    ])
    
    with tab1:
        st.subheader("📊 Overview Dashboard")
        overview_fig = visualizer.create_overview_dashboard()
        st.plotly_chart(overview_fig, use_container_width=True)
    
    with tab2:
        st.subheader("⏰ Activity Timeline")
        timeline_fig = visualizer.create_activity_timeline()
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("📅 No temporal data available for timeline visualization")
    
    with tab3:
        st.subheader("🎯 Objectives Analysis")
        objectives_fig = visualizer.create_objectives_analysis()
        if objectives_fig:
            st.plotly_chart(objectives_fig, use_container_width=True)
        else:
            st.info("🎯 No objectives data available")
    
    with tab4:
        st.subheader("📈 Confidence Analysis")
        confidence_fig = visualizer.create_confidence_analysis()
        st.plotly_chart(confidence_fig, use_container_width=True)
    
    with tab5:
        st.subheader("☁️ Word Cloud Analysis")
        try:
            import matplotlib.pyplot as plt
            wordcloud_fig = visualizer.create_wordclouds()
            st.pyplot(wordcloud_fig)
        except Exception as e:
            st.error(f"❌ Error generating word clouds: {str(e)}")
    
    with tab6:
        st.subheader("📋 Summary Report")
        report = visualizer.generate_summary_report()
        
        # Display summary in organized sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 General Statistics**")
            st.json({
                "Total Activities": report['total_activities'],
                "Unique Meetings": report['unique_meetings'],
                "Average Confidence": round(report['avg_confidence'], 3)
            })
            
            st.markdown("**📄 Document Types**")
            st.json(report['document_types'])
            
            st.markdown("**🏷️ Activity Categories**")
            st.json(report['activity_categories'])
        
        with col2:
            st.markdown("**🏢 Top Organizations**")
            st.json(report['top_organizations'])
            
            st.markdown("**🌍 Top Locations**")
            st.json(report['top_locations'])
            
            if report['temporal_range']:
                st.markdown("**📅 Temporal Range**")
                st.json(report['temporal_range'])
    
    # Data export section
    st.subheader("💾 Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Excel"):
            output = df.to_excel(index=False)
            st.download_button(
                label="📥 Download Excel File",
                data=output,
                file_name=f"gftads_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"gftads_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"gftads_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
