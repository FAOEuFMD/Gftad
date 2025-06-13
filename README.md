# GF-TADs Data Analysis and Visualization

This project provides a comprehensive solution for extracting, analyzing, and visualizing data from GF-TADs (Global Framework for the Progressive Control of Transboundary Animal Diseases) documents.

## 🎯 Features

### Data Extraction
- **PDF Processing**: Extract text from PDF documents using multiple libraries (pdfplumber, PyPDF2)
- **Entity Extraction**: Identify What, When, Who, Where, Impact, and Objectives from unstructured text
- **Natural Language Processing**: Use spaCy and NLTK for advanced text analysis
- **Confidence Scoring**: Assign confidence scores to extracted information

### Data Analysis
- **Temporal Analysis**: Track activities and recommendations over time
- **Organization Analysis**: Identify key stakeholders and their involvement
- **Objective Categorization**: Classify activities by their strategic objectives
- **Impact Assessment**: Analyze the expected impact of different activities

### Visualization
- **Interactive Dashboards**: Comprehensive overview of all extracted data
- **Timeline Visualizations**: Track activities across meetings and years
- **Word Clouds**: Visual representation of key terms and concepts
- **Network Analysis**: Relationships between organizations and activities
- **Statistical Analysis**: Confidence scores and data quality metrics

### Web Interface
- **Streamlit Dashboard**: User-friendly web interface for data analysis
- **Real-time Processing**: Extract and analyze data in real-time
- **Export Capabilities**: Download results in Excel, CSV, or JSON format

## 📋 Requirements

### Python Dependencies
- PyPDF2==3.0.1
- pdfplumber==0.10.0
- pandas==2.1.4
- numpy==1.25.2
- matplotlib==3.8.2
- seaborn==0.13.0
- plotly==5.17.0
- wordcloud==1.9.2
- spacy==3.7.2
- nltk==3.8.1
- scikit-learn==1.3.2
- textblob==0.17.1
- streamlit==1.29.0

### Additional Setup
```bash
# Install spaCy English model
python -m spacy download en_core_web_sm

# NLTK data will be downloaded automatically when first run
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or download the project files
# Navigate to the project directory
cd path/to/gftad/project

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

### 2. Prepare Your Data
Organize your GF-TADs documents in the following structure:
```
your_project_folder/
├── GSC_Recommendations/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── GSC_Reports/
│   ├── report1.pdf
│   ├── report2.pdf
│   └── ...
```

### 3. Run the Analysis

#### Option A: Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
Then open your browser to `http://localhost:8501`

#### Option B: Command Line
```python
from data_extractor import GFTADsDataExtractor
from visualizer import GFTADsVisualizer

# Extract data
extractor = GFTADsDataExtractor("path/to/your/data/folder")
df = extractor.process_all_documents()
output_file = extractor.save_extracted_data(df, 'excel')

# Create visualizations
visualizer = GFTADsVisualizer(df=df)
visualizer.save_visualizations()
```

## 📊 Usage Examples

### Extract Data from PDFs
```python
from data_extractor import GFTADsDataExtractor

# Initialize extractor
extractor = GFTADsDataExtractor(r"c:\path\to\your\gftad\folder")

# Process all documents
df = extractor.process_all_documents()

# Save results
output_file = extractor.save_extracted_data(df, format='excel')
print(f"Data saved to: {output_file}")
```

### Create Visualizations
```python
from visualizer import GFTADsVisualizer

# Create visualizer
visualizer = GFTADsVisualizer("path/to/extracted_data.xlsx")

# Generate overview dashboard
overview = visualizer.create_overview_dashboard()
overview.show()

# Create timeline visualization
timeline = visualizer.create_activity_timeline()
timeline.show()

# Save all visualizations
visualizer.save_visualizations("output/folder")
```

### Generate Summary Report
```python
# Generate comprehensive summary
report = visualizer.generate_summary_report()
print(f"Total activities: {report['total_activities']}")
print(f"Average confidence: {report['avg_confidence']:.2f}")
```

## 📁 Output Files

The system generates several types of output files:

### Extracted Data
- `gftads_extracted_data_YYYYMMDD_HHMMSS.xlsx` - Main data file with all extracted information
- `gftads_extracted_data_YYYYMMDD_HHMMSS.csv` - CSV version for broader compatibility
- `gftads_extracted_data_YYYYMMDD_HHMMSS.json` - JSON format for programmatic access

### Visualizations
- `overview_dashboard_YYYYMMDD_HHMMSS.html` - Interactive overview dashboard
- `activity_timeline_YYYYMMDD_HHMMSS.html` - Timeline visualization
- `objectives_analysis_YYYYMMDD_HHMMSS.html` - Objectives analysis
- `confidence_analysis_YYYYMMDD_HHMMSS.html` - Confidence score analysis
- `wordclouds_YYYYMMDD_HHMMSS.png` - Word cloud visualizations
- `summary_report_YYYYMMDD_HHMMSS.json` - Comprehensive summary report

## 🔧 Configuration

### Customizing Keywords
You can customize the keywords used for entity extraction by modifying the `setup_keywords()` method in the `GFTADsDataExtractor` class:

```python
self.keywords = {
    'time_indicators': ['by', 'before', 'after', 'during', ...],
    'action_verbs': ['develop', 'implement', 'establish', ...],
    'organizations': ['FAO', 'OIE', 'WHO', ...],
    # Add your custom keywords here
}
```

### Adjusting Confidence Scoring
Modify the `calculate_confidence_score()` method to adjust how confidence scores are calculated based on your specific requirements.

## 🐛 Troubleshooting

### Common Issues

1. **PDF Extraction Fails**
   - Ensure PDFs are not password-protected
   - Try different PDF processing libraries if one fails
   
2. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Memory Issues with Large PDFs**
   - Process documents in smaller batches
   - Increase system memory or use a machine with more RAM

4. **Poor Entity Extraction Quality**
   - Adjust keywords in the configuration
   - Improve preprocessing steps
   - Consider using more advanced NLP models

### Performance Tips

- **Parallel Processing**: For large document collections, consider implementing parallel processing
- **Caching**: Use caching for repeated analyses
- **Preprocessing**: Clean and preprocess text for better extraction quality

## 📈 Advanced Usage

### Custom Entity Extraction
You can extend the entity extraction by creating custom extraction methods:

```python
def extract_custom_entities(self, text):
    # Your custom extraction logic here
    return extracted_entities
```

### Integration with Other Systems
The extracted data can be easily integrated with:
- Business Intelligence tools (Power BI, Tableau)
- Database systems (SQL Server, PostgreSQL)
- Web applications and APIs
- Machine learning pipelines

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new visualization types
- Improving entity extraction algorithms
- Enhancing the user interface
- Adding support for additional document formats

## 📄 License

This project is provided as-is for educational and research purposes.

## 📞 Support

For questions or issues, please check the troubleshooting section above or refer to the inline documentation in the code files.
