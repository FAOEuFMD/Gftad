import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# PDF processing
try:
    import PyPDF2
except ImportError:
    print("Installing PyPDF2...")
    os.system("pip install PyPDF2")
    import PyPDF2

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    os.system("pip install pdfplumber")
    import pdfplumber

# NLP and text processing
try:
    import spacy
    # Try to load the model
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy English model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp_model = spacy.load("en_core_web_sm")
except ImportError:
    print("Installing spacy...")
    os.system("pip install spacy")
    import spacy
    nlp_model = None

try:
    from textblob import TextBlob
except ImportError:
    print("Installing textblob...")
    os.system("pip install textblob")
    from textblob import TextBlob

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
except ImportError:
    print("Installing nltk...")
    os.system("pip install nltk")
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag

# Machine Learning
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
except ImportError:
    print("Installing scikit-learn...")
    os.system("pip install scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Installing visualization packages...")
    os.system("pip install matplotlib seaborn plotly")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

try:
    from wordcloud import WordCloud
except ImportError:
    print("Installing wordcloud...")
    os.system("pip install wordcloud")
    from wordcloud import WordCloud

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExtractedActivity:
    """Data class to store extracted activity information"""
    what: str
    when: str
    who: str
    where: str
    impact: str
    objectives: List[str]
    meeting_number: str
    document_type: str
    page_number: int
    confidence_score: float
    raw_text: str

class GFTADsDataExtractor:
    """Main class for extracting structured data from GF-TADs documents"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "extracted_data"
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize NLP models
        self.setup_nlp()
        # Keywords for entity extraction
        self.setup_keywords()
        
    def setup_nlp(self):
        """Initialize NLP models and tools"""
        try:
            # Try to load spaCy model
            self.nlp = nlp_model if nlp_model else spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
            self.nlp = None
        except Exception as e:
            logger.warning(f"Error loading spaCy model: {e}")
            self.nlp = None
            
        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Error loading NLTK stopwords: {e}")
            self.stop_words = set()
        
    def setup_keywords(self):
        """Define keywords for different categories"""
        self.keywords = {
            'time_indicators': [
                'by', 'before', 'after', 'during', 'within', 'until', 'from', 'to',
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december',
                '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030',
                'quarterly', 'annually', 'monthly', 'yearly', 'deadline', 'timeline'
            ],
            'action_verbs': [
                'develop', 'implement', 'establish', 'create', 'enhance', 'strengthen',
                'improve', 'coordinate', 'facilitate', 'support', 'promote', 'ensure',
                'monitor', 'evaluate', 'assess', 'review', 'update', 'maintain'
            ],
            'organizations': [
                'FAO', 'OIE', 'WHO', 'GF-TADs', 'GFTAD', 'GSC', 'Regional Commission',
                'government', 'ministry', 'department', 'secretariat', 'committee',
                'working group', 'task force', 'partnership', 'alliance'
            ],
            'locations': [
                'africa', 'asia', 'europe', 'america', 'oceania', 'regional', 'national',
                'local', 'global', 'international', 'country', 'region', 'continent'
            ],
            'objectives': [
                'capacity building', 'surveillance', 'preparedness', 'response',
                'prevention', 'control', 'eradication', 'coordination', 'collaboration',
                'information sharing', 'early warning', 'risk assessment', 'emergency'
            ],
            'impact_indicators': [
                'reduce', 'increase', 'improve', 'enhance', 'strengthen', 'impact',
                'outcome', 'result', 'benefit', 'effect', 'consequence', 'success',
                'achievement', 'progress', 'advancement'
            ]
        }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Tuple[str, int]]:
        """Extract text from PDF file with page numbers"""
        text_pages = []
        
        try:
            # Try with pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_pages.append((text, page_num))
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        text = page.extract_text()
                        if text:
                            text_pages.append((text, page_num))
            except Exception as e:
                logger.error(f"Failed to extract text from {pdf_path}: {e}")
                
        return text_pages
    
    def extract_meeting_info(self, filename: str) -> Tuple[str, str]:
        """Extract meeting number and document type from filename"""
        # Extract meeting number
        meeting_match = re.search(r'(\d+)(st|nd|rd|th)', filename.lower())
        meeting_number = meeting_match.group(1) if meeting_match else "Unknown"
        
        # Determine document type
        doc_type = "Recommendations" if "recommen" in filename.lower() else "Reports"
        
        return meeting_number, doc_type
    
    def extract_entities_with_context(self, text: str) -> Dict[str, List[str]]:
        """Extract entities with context using multiple approaches"""
        entities = {
            'what': [],
            'when': [],
            'who': [],
            'where': [],
            'objectives': [],
            'impact': []
        }
        
        # Clean and preprocess text
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Extract temporal information (when)
            if any(keyword in sentence_lower for keyword in self.keywords['time_indicators']):
                time_entities = self.extract_time_entities(sentence)
                entities['when'].extend(time_entities)
            
            # Extract actions (what)
            if any(verb in sentence_lower for verb in self.keywords['action_verbs']):
                action_entities = self.extract_action_entities(sentence)
                entities['what'].extend(action_entities)
            
            # Extract organizations (who)
            if any(org in sentence_lower for org in self.keywords['organizations']):
                org_entities = self.extract_organization_entities(sentence)
                entities['who'].extend(org_entities)
            
            # Extract locations (where)
            if any(loc in sentence_lower for loc in self.keywords['locations']):
                location_entities = self.extract_location_entities(sentence)
                entities['where'].extend(location_entities)
            
            # Extract objectives
            if any(obj in sentence_lower for obj in self.keywords['objectives']):
                objective_entities = self.extract_objective_entities(sentence)
                entities['objectives'].extend(objective_entities)
            
            # Extract impact information
            if any(impact in sentence_lower for impact in self.keywords['impact_indicators']):
                impact_entities = self.extract_impact_entities(sentence)
                entities['impact'].extend(impact_entities)
        
        # Remove duplicates and clean
        for key in entities:
            entities[key] = list(set([item.strip() for item in entities[key] if item.strip()]))
        
        return entities
    
    def extract_time_entities(self, sentence: str) -> List[str]:
        """Extract time-related entities from sentence"""
        time_entities = []
        
        # Date patterns
        date_patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',  # DD/MM/YYYY or MM/DD/YYYY
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            r'\b(\d{4})\b',  # Year
            r'\b(by\s+\d{4})\b',  # by YYYY
            r'\b(within\s+\d+\s+(?:months?|years?))\b',  # within X months/years
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            time_entities.extend(matches)
            
        return time_entities
    
    def extract_action_entities(self, sentence: str) -> List[str]:
        """Extract action-related entities from sentence"""
        # Look for action verbs and their objects
        action_entities = []
        
        # Use regex to find action patterns
        action_patterns = [
            r'\b(develop|implement|establish|create|enhance|strengthen|improve)\s+([^.;,]+)',
            r'\b(coordinate|facilitate|support|promote|ensure)\s+([^.;,]+)',
            r'\b(monitor|evaluate|assess|review|update|maintain)\s+([^.;,]+)',
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    action_entities.append(f"{match[0]} {match[1]}")
                else:
                    action_entities.append(match)
        
        return action_entities
    
    def extract_organization_entities(self, sentence: str) -> List[str]:
        """Extract organization-related entities from sentence"""
        org_entities = []
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(sentence)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON']:
                    org_entities.append(ent.text)
        
        # Also use keyword matching
        for org in self.keywords['organizations']:
            if org.lower() in sentence.lower():
                org_entities.append(org)
        
        return org_entities
    
    def extract_location_entities(self, sentence: str) -> List[str]:
        """Extract location-related entities from sentence"""
        location_entities = []
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(sentence)
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entity, Location
                    location_entities.append(ent.text)
        
        # Also use keyword matching
        for loc in self.keywords['locations']:
            if loc.lower() in sentence.lower():
                location_entities.append(loc)
        
        return location_entities
    
    def extract_objective_entities(self, sentence: str) -> List[str]:
        """Extract objective-related entities from sentence"""
        objective_entities = []
        
        for obj in self.keywords['objectives']:
            if obj.lower() in sentence.lower():
                # Extract context around the objective
                words = sentence.split()
                for i, word in enumerate(words):
                    if obj.lower() in word.lower():
                        # Get surrounding context
                        start = max(0, i-3)
                        end = min(len(words), i+4)
                        context = ' '.join(words[start:end])
                        objective_entities.append(context)
        
        return objective_entities
    
    def extract_impact_entities(self, sentence: str) -> List[str]:
        """Extract impact-related entities from sentence"""
        impact_entities = []
        
        # Look for impact patterns
        impact_patterns = [
            r'\b(reduce|decrease)\s+([^.;,]+?)(?:by|to)\s+([^.;,]+)',
            r'\b(increase|improve|enhance)\s+([^.;,]+)',
            r'\b(impact|outcome|result|benefit)\s+(?:of|from)\s+([^.;,]+)',
        ]
        
        for pattern in impact_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    impact_entities.append(' '.join(match))
                else:
                    impact_entities.append(match)
        
        return impact_entities
    
    def calculate_confidence_score(self, entities: Dict[str, List[str]], text: str) -> float:
        """Calculate confidence score based on entity extraction quality"""
        total_entities = sum(len(entities[key]) for key in entities)
        text_length = len(text.split())
        
        # Base score on entity density and diversity
        entity_density = total_entities / text_length if text_length > 0 else 0
        entity_diversity = len([key for key in entities if entities[key]]) / len(entities)
        
        confidence = (entity_density * 0.5 + entity_diversity * 0.5)
        return min(confidence, 1.0)  # Cap at 1.0
    
    def process_document(self, pdf_path: Path) -> List[ExtractedActivity]:
        """Process a single PDF document and extract activities"""
        logger.info(f"Processing: {pdf_path.name}")
        
        meeting_number, doc_type = self.extract_meeting_info(pdf_path.name)
        text_pages = self.extract_text_from_pdf(pdf_path)
        
        activities = []
        
        for text, page_num in text_pages:
            if len(text.strip()) < 100:  # Skip very short pages
                continue
                
            # Extract entities
            entities = self.extract_entities_with_context(text)
            
            # Calculate confidence score
            confidence = self.calculate_confidence_score(entities, text)
            
            # Create activity object
            activity = ExtractedActivity(
                what='; '.join(entities['what'][:3]),  # Top 3 actions
                when='; '.join(entities['when'][:3]),  # Top 3 time references
                who='; '.join(entities['who'][:3]),    # Top 3 organizations
                where='; '.join(entities['where'][:3]), # Top 3 locations
                impact='; '.join(entities['impact'][:3]), # Top 3 impacts
                objectives=entities['objectives'][:5],   # Top 5 objectives
                meeting_number=meeting_number,
                document_type=doc_type,
                page_number=page_num,
                confidence_score=confidence,
                raw_text=text[:500] + "..." if len(text) > 500 else text  # Truncate for storage
            )
            
            activities.append(activity)
        
        return activities
    
    def process_all_documents(self) -> pd.DataFrame:
        """Process all PDF documents found in the base path and subfolders"""
        all_activities = []
        
        # Find all PDF files recursively
        all_pdfs = list(self.base_path.glob("**/*.pdf"))
        
        # Categorize PDFs based on filename patterns
        for pdf_file in all_pdfs:
            try:
                # Skip files in extracted_data folder
                if "extracted_data" in str(pdf_file):
                    continue
                    
                activities = self.process_document(pdf_file)
                all_activities.extend(activities)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        # Convert to DataFrame
        df_data = [asdict(activity) for activity in all_activities]
        df = pd.DataFrame(df_data)
        
        # Add processing timestamp
        if not df.empty:
            df['processed_at'] = datetime.now()
        
        return df
    
    def save_extracted_data(self, df: pd.DataFrame, format: str = 'excel'):
        """Save extracted data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'excel':
            output_file = self.output_path / f"gftads_extracted_data_{timestamp}.xlsx"
            df.to_excel(output_file, index=False)
        elif format.lower() == 'csv':
            output_file = self.output_path / f"gftads_extracted_data_{timestamp}.csv"
            df.to_csv(output_file, index=False)
        elif format.lower() == 'json':
            output_file = self.output_path / f"gftads_extracted_data_{timestamp}.json"
            df.to_json(output_file, orient='records', indent=2)
        
        logger.info(f"Data saved to: {output_file}")
        return output_file

if __name__ == "__main__":
    # Example usage
    base_path = r"c:\Users\user\EUFMD\Gftad"
    extractor = GFTADsDataExtractor(base_path)
    
    # Process all documents
    df = extractor.process_all_documents()
    
    # Save results
    output_file = extractor.save_extracted_data(df, 'excel')
    
    print(f"Extraction complete! Results saved to: {output_file}")
    print(f"Total activities extracted: {len(df)}")
    print(f"Average confidence score: {df['confidence_score'].mean():.2f}")
