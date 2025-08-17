import re, os, logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import spacy, pdfplumber, nltk, pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ExtractedActivity:
    what: str
    when: str
    who: str
    where: str
    impact: str
    area_of_work: List[str]
    mel_objective: str
    disease: str
    meeting_number: str
    document_type: str
    page_number: int
    confidence_score: float
    raw_text: str

class GFTADsDataExtractor:

    def process_and_update_database(self, pdf_paths: list = None) -> pd.DataFrame:
        """Process given PDFs (or all in base_path if None), update the persistent database, and return the updated DataFrame."""
        if pdf_paths is None:
            pdf_paths = list(Path(self.base_path).glob("*.pdf"))
        all_acts = []
        for pdf_file in pdf_paths:
            acts = self.process_document(pdf_file)
            all_acts.extend(acts)
        # Convert area_of_work (list) to string for DataFrame and deduplication
        dict_acts = [asdict(a) for a in all_acts]
        for d in dict_acts:
            if isinstance(d.get('area_of_work'), list):
                # Ensure all elements are string before joining
                d['area_of_work'] = '; '.join([str(x) for x in d['area_of_work']])
            elif not isinstance(d.get('area_of_work'), str):
                # If it's not a string or list, convert to string
                d['area_of_work'] = str(d.get('area_of_work'))
        new_df = pd.DataFrame(dict_acts)
        new_df["processed_at"] = datetime.now()
        db_file = Path(self.db_path)
        if db_file.exists():
            if db_file.suffix == ".csv":
                db_df = pd.read_csv(db_file)
            else:
                db_df = pd.read_excel(db_file)
            combined_df = pd.concat([db_df, new_df], ignore_index=True)
            # Ensure area_of_work is string for deduplication
            if 'area_of_work' in combined_df.columns:
                combined_df['area_of_work'] = combined_df['area_of_work'].astype(str)
            combined_df = combined_df.drop_duplicates()
        else:
            combined_df = new_df
        if db_file.suffix == ".csv":
            combined_df.to_csv(db_file, index=False)
        else:
            combined_df.to_excel(db_file, index=False)
        return combined_df
    def __init__(self, base_path: str, db_path: str = None):
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "extracted_data"
        self.output_path.mkdir(exist_ok=True)
        self.db_path = db_path or str(self.output_path / "gftads_database.xlsx")

        self.setup_nlp()
        self.setup_keywords()

    def setup_nlp(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not loaded. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        try:
            self.stop_words = set(stopwords.words("english"))
        except:
            nltk.download("stopwords")
            self.stop_words = set(stopwords.words("english"))

    def setup_keywords(self):
        self.disease_keywords = {
            "ASF": "African swine fever",
            "PPR": "Peste des petits ruminants",
            "FMD": "Foot-and-mouth disease",
            "LSD": "Lumpy skin disease",
            "HPAI": "Highly pathogenic avian influenza",
            "Rabies": "Rabies",
            "RVF": "Rift Valley fever",
            "CBPP": "Contagious bovine pleuropneumonia"
        }
        self.area_mapping = {
            "training": "Capacity Development",
            "capacity": "Capacity Development",
            "guidelines": "Capacity Development",
            "coordination": "Coordination",
            "meeting": "Coordination",
            "framework": "Policy/Strategy",
            "strategy": "Policy/Strategy",
            "surveillance": "Surveillance",
            "diagnostic": "Diagnostics",
            "advocacy": "Advocacy",
            "funding": "Resource Mobilization",
        }
        self.mel_mapping = {
            "strategy": "Objective 1.2 – Formulate regional/subregional strategies",
            "mechanism": "Objective 1.3 – Establish harmonized planning mechanisms",
            "capacity": "Objective 2.1 – Address capacity gaps",
            "training": "Objective 2.1 – Address capacity gaps",
            "planning": "Objective 2.2 – Strengthen multi-disciplinary planning",
            "monitor": "Objective 2.3 – Provide harmonized monitoring tools",
            "stakeholder": "Objective 3.1 – Strengthen engagement/coordination",
            "advocacy": "Objective 3.2 – Improve advocacy skills",
            "funding": "Objective 3.3 – Promote sustainable funding mechanisms",
        }

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Tuple[str, int]]:
        text_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text: text_pages.append((text, page_num))
        return text_pages

    def extract_entities(self, text: str) -> Dict[str, str]:
        """Entity extraction with MEL + diseases"""
        area = set()
        mel_obj = "Unmapped"
        disease = "All priority TADs"

        # Disease detection
        for k,v in self.disease_keywords.items():
            if re.search(rf"\b{k}\b", text, re.IGNORECASE):
                disease = v

        # Area + MEL mapping
        for k,v in self.area_mapping.items():
            if k in text.lower(): area.add(v)
        for k,v in self.mel_mapping.items():
            if k in text.lower(): mel_obj = v

        return {"area_of_work": list(area), "mel_objective": mel_obj, "disease": disease}

    def calculate_confidence(self, text: str, entities: Dict[str,str]) -> float:
        score = 0
        if entities["area_of_work"]: score += 0.3
        if entities["disease"] != "All priority TADs": score += 0.3
        if "meeting" in text.lower() or "training" in text.lower(): score += 0.4
        return min(1.0, score)

    def process_document(self, pdf_path: Path) -> List[ExtractedActivity]:
        logger.info(f"Processing {pdf_path}")
        text_pages = self.extract_text_from_pdf(pdf_path)
        activities = []
        meeting_number = re.search(r'(\d+)(st|nd|rd|th)', pdf_path.name)
        meeting_number = meeting_number.group(1) if meeting_number else "Unknown"
        doc_type = "Report"

        for text, page_num in text_pages:
            # Split into paragraphs
            for para in text.split("\n"):
                if not para.strip(): continue
                if len(para) < 80: continue  # avoid short headers
                if not re.search(r"(meeting|training|workshop|launched|established|developed|conducted)", para, re.I):
                    continue  # skip descriptive text

                entities = self.extract_entities(para)
                confidence = self.calculate_confidence(para, entities)

                act = ExtractedActivity(
                    what=para[:120],
                    when=str(datetime.now().year),  # fallback
                    who="GF-TADs / Partners",
                    where="Global",
                    impact="",
                    area_of_work=entities["area_of_work"],
                    mel_objective=entities["mel_objective"],
                    disease=entities["disease"],
                    meeting_number=meeting_number,
                    document_type=doc_type,
                    page_number=page_num,
                    confidence_score=confidence,
                    raw_text=para
                )
                activities.append(act)
        return activities

    def process_all_documents(self, folder: str) -> pd.DataFrame:
        all_acts = []
        for pdf_file in Path(folder).glob("*.pdf"):
            acts = self.process_document(pdf_file)
            all_acts.extend(acts)
        df = pd.DataFrame([asdict(a) for a in all_acts])
        df["processed_at"] = datetime.now()
        return df

if __name__ == "__main__":
    base_path = r"c:\Users\user\EUFMD\Gftad"
    extractor = GFTADsDataExtractor(base_path)
    df = extractor.process_all_documents(base_path)
    df.to_excel(extractor.output_path / "gftads_activities_with_mel.xlsx", index=False)
    print(f"Saved {len(df)} activities with MEL mapping")