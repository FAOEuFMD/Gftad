"""
Main execution script for GF-TADs data analysis
This script provides a simple command-line interface to run the complete analysis pipeline
"""

import sys
import argparse
from pathlib import Path
import logging
from data_extractor import GFTADsDataExtractor
from visualizer import GFTADsVisualizer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gftads_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='GF-TADs Data Analysis Pipeline')
    parser.add_argument('base_path', help='Path to the folder containing GSC_Recommendations and GSC_Reports')
    parser.add_argument('--extract', action='store_true', help='Extract new data from PDFs')
    parser.add_argument('--visualize', help='Path to existing data file for visualization')
    parser.add_argument('--output', default='output', help='Output directory for results')
    parser.add_argument('--format', choices=['excel', 'csv', 'json'], default='excel', help='Output format for extracted data')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    base_path = Path(args.base_path)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    # Check if base path exists
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        sys.exit(1)
    
    try:
        if args.extract:
            # Extract new data
            logger.info("Starting data extraction...")
            extractor = GFTADsDataExtractor(str(base_path))
            df = extractor.process_all_documents()
            
            if df.empty:
                logger.error("No data could be extracted from PDFs")
                sys.exit(1)
            
            # Save extracted data
            output_file = extractor.save_extracted_data(df, args.format)
            logger.info(f"Data extraction completed. Saved to: {output_file}")
            
            # Create visualizations
            logger.info("Creating visualizations...")
            visualizer = GFTADsVisualizer(df=df)
            viz_output = visualizer.save_visualizations(output_path / "visualizations")
            logger.info(f"Visualizations saved to: {viz_output}")
            
        elif args.visualize:
            # Visualize existing data
            data_file = Path(args.visualize)
            if not data_file.exists():
                logger.error(f"Data file does not exist: {data_file}")
                sys.exit(1)
            
            logger.info(f"Loading data from: {data_file}")
            visualizer = GFTADsVisualizer(str(data_file))
            
            logger.info("Creating visualizations...")
            viz_output = visualizer.save_visualizations(output_path / "visualizations")
            logger.info(f"Visualizations saved to: {viz_output}")
            
        else:
            # Full pipeline: extract and visualize
            logger.info("Running full pipeline: extraction + visualization...")
            
            # Extract data
            extractor = GFTADsDataExtractor(str(base_path))
            df = extractor.process_all_documents()
            
            if df.empty:
                logger.error("No data could be extracted from PDFs")
                sys.exit(1)
            
            # Save extracted data
            output_file = extractor.save_extracted_data(df, args.format)
            logger.info(f"Data extraction completed. Saved to: {output_file}")
            
            # Create visualizations
            visualizer = GFTADsVisualizer(df=df)
            viz_output = visualizer.save_visualizations(output_path / "visualizations")
            logger.info(f"Visualizations saved to: {viz_output}")
        
        logger.info("Analysis pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
