import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="GF-TADs Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


import re
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


# Always use the advanced extractor
try:
    from data_extractor import GFTADsDataExtractor
    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False

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
    /* Make selected multiselect tags grey instead of red */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #e0e0e0 !important;
        color: #333 !important;
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

    # Tabs: Dashboard (first), Extract More Data (second)
    tab_dashboard, tab_extract = st.tabs(["Dashboard", "Extract More Data"])

    with tab_dashboard:
        # --- Dashboard Tab ---
        # Path to database (default)
        base_path = r"c:\Users\user\EUFMD\Gftad"
        db_path = str(Path(base_path) / "extracted_data" / "gftads_database.xlsx")
        st.header("📊 Visualization Dashboard (Live from Database)")
        if Path(db_path).exists():
            try:
                df = pd.read_excel(db_path)
                st.success(f"Loaded {len(df)} records from database.")

                # --- FILTERS ---

                filter_col1, filter_col2, filter_col3 = st.columns(3)

                # Year filter (from 'when' column, extract 4-digit years)
                with filter_col1:
                    years = set()
                    df_when = df['when'].dropna().astype(str)
                    for val in df_when:
                        years.update([y for y in re.findall(r'\b(20\d{2}|19\d{2})\b', val)])
                    years = sorted(years)
                    selected_years = st.multiselect(
                        "Filter by Year",
                        options=years,
                        default=[],
                        help="Select one or more years to filter. Leave empty to show all."
                    )

                # Objectives filter (from 'objectives' column, which may be list or string)
                with filter_col2:
                    obj_set = set()
                    for val in df['objectives'].dropna():
                        if isinstance(val, list):
                            for o in val:
                                for obj in str(o).split(','):
                                    for obj2 in obj.split(';'):
                                        obj_set.add(obj2.strip())
                        elif isinstance(val, str):
                            # Try to parse as list
                            if val.startswith('[') and val.endswith(']'):
                                try:
                                    import ast
                                    parsed = ast.literal_eval(val)
                                    for o in parsed:
                                        for obj in str(o).split(','):
                                            for obj2 in obj.split(';'):
                                                obj_set.add(obj2.strip())
                                except Exception:
                                    for obj in val.split(','):
                                        for obj2 in obj.split(';'):
                                            obj_set.add(obj2.strip())
                            else:
                                for obj in val.split(','):
                                    for obj2 in obj.split(';'):
                                        obj_set.add(obj2.strip())
                    objectives = sorted([o for o in obj_set if o])
                    selected_objectives = st.multiselect(
                        "Filter by Objective",
                        options=objectives,
                        default=[],
                        help="Select one or more objectives to filter. Leave empty to show all."
                    )

                # Where filter (from 'where' column, split by semicolon and comma)
                with filter_col3:
                    where_set = set()
                    for val in df['where'].dropna().astype(str):
                        for w in val.split(';'):
                            for w2 in w.split(','):
                                w2 = w2.strip()
                                if w2:
                                    where_set.add(w2)
                    wheres = sorted(where_set)
                    selected_wheres = st.multiselect(
                        "Filter by Where",
                        options=wheres,
                        default=[],
                        help="Select one or more locations to filter. Leave empty to show all."
                    )

                # --- APPLY FILTERS ---
                df_filtered = df.copy()
                # Filter by year (if any selected)
                if selected_years and years:
                    df_filtered = df_filtered[df_filtered['when'].astype(str).apply(lambda x: any(y in x for y in selected_years))]
                # Filter by objectives
                if selected_objectives and objectives:
                    def obj_match(val):
                        objs = set()
                        if isinstance(val, list):
                            for o in val:
                                for obj in str(o).split(','):
                                    for obj2 in obj.split(';'):
                                        objs.add(obj2.strip())
                        elif isinstance(val, str):
                            if val.startswith('[') and val.endswith(']'):
                                try:
                                    import ast
                                    parsed = ast.literal_eval(val)
                                    for o in parsed:
                                        for obj in str(o).split(','):
                                            for obj2 in obj.split(';'):
                                                objs.add(obj2.strip())
                                except Exception:
                                    for obj in val.split(','):
                                        for obj2 in obj.split(';'):
                                            objs.add(obj2.strip())
                            else:
                                for obj in val.split(','):
                                    for obj2 in obj.split(';'):
                                        objs.add(obj2.strip())
                        objs = set([o for o in objs if o])
                        return any(o in selected_objectives for o in objs)
                    df_filtered = df_filtered[df_filtered['objectives'].apply(obj_match)]
                # Filter by where
                if selected_wheres and wheres:
                    def where_match(val):
                        whs = set()
                        for w in str(val).split(';'):
                            for w2 in w.split(','):
                                w2 = w2.strip()
                                if w2:
                                    whs.add(w2)
                        return any(w in whs for w in selected_wheres)
                    df_filtered = df_filtered[df_filtered['where'].apply(where_match)]

                analyze_data(df_filtered)
            except Exception as e:
                st.error(f"❌ Could not load database: {e}")
        else:
            st.info("No database found yet. Extract data to create it.")

    with tab_extract:
        # --- Extract More Data Tab ---
        extract_new_data()
def extract_new_data():
    st.header("🔄 Extract New Data from PDFs")
    st.markdown("""
    <div class="highlight">
    <strong>📁 Data Extraction Process</strong><br>
    This will process all PDF files in your GSC_Recommendations and GSC_Reports folders
    and extract structured information including what, when, who, where, impact, and objectives.<br>
    <strong>After extraction, the dashboard tab will show the latest data from your database.</strong>
    </div>
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
        # Find all PDF files in both folders
        pdf_files = list(recommendations_path.glob("*.pdf")) + list(reports_path.glob("*.pdf"))
        if not pdf_files:
            st.error("❌ No PDF files found in the specified folders.")
            return
        # Start extraction and append to database
        with st.spinner("🔍 Extracting and updating database... This may take a few minutes."):
            try:
                db_path = str(Path(base_path) / "extracted_data" / "gftads_database.xlsx")
                extractor = GFTADsDataExtractor(base_path, db_path=db_path)
                updated_df = extractor.process_and_update_database(pdf_files)
                if updated_df.empty:
                    st.error("❌ No data could be extracted from the PDFs.")
                    return
                st.success(f"✅ Data extraction and database update completed!")
                st.info(f"📄 Total activities in database: {len(updated_df)}")
                st.info(f"💾 Database file: {db_path}")
                st.session_state['extracted_df'] = updated_df
                st.session_state['data_source'] = db_path
                st.subheader("📊 Database Preview (last 10 rows)")
                st.dataframe(updated_df.tail(10))
            except Exception as e:
                st.error(f"❌ Error during extraction: {str(e)}")



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
