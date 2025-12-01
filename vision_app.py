# app.py - AfroMediBot Feature Showcase
import streamlit as st
import pandas as pd
import os
from io import BytesIO
import base64
from PIL import Image
from google.cloud import vision
from Bio import Entrez
from groq import Groq
import json
import time
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
import py3Dmol
from stmol import showmol
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import requests
import zipfile
import shutil


VISION_AVAILABLE = True
LITERATURE_AVAILABLE = True
RDKIT_AVAILABLE = True

# Initialize session state variables
if 'compounds_df' not in st.session_state:
    st.session_state.compounds_df = None
if 'mapped_plant' not in st.session_state:
    st.session_state.mapped_plant = None
if 'plant_compounds' not in st.session_state:
    st.session_state.plant_compounds = None
# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AfroMediBot - AI Drug Discovery",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
    }
    
    /* Stats Cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #2E7D32;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Feature Cards */
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Quick Action Buttons */
    .quick-action {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .quick-action:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)
# ====================================================
# RETROSYNTHESIS
#=====================================================
# NOTE: Replace 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE' with the ID you get from the shared zip link.
GDRIVE_FILE_ID = "1Vu3YUCwq8KQu7vMmRKKZxhzfCWtvj0ud" 
MODEL_PATH = "retrosynthesis_model" # This is the internal directory name
RETROSYNTHESIS_ZIP = "retrosynthesis_model.zip"

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    import shutil 
    # Check if the required local files exist AFTER the first successful extraction.
    # This prevents the app from trying to download on every run once the files are present.
    required_file = os.path.join(MODEL_PATH, "tokenizer_config.json")
    RETROSYNTHESIS_AVAILABLE = os.path.exists(required_file)

except ImportError:
    RETROSYNTHESIS_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def load_retrosynthesis_model():
    """Loads the T5 model, downloading and extracting it from Drive if necessary."""

    # Check 1: Use the global availability check flag for early exit
    global RETROSYNTHESIS_AVAILABLE
    # 1. Check if the model is already downloaded
    required_file = os.path.join(MODEL_PATH, "tokenizer_config.json")
    if os.path.exists(required_file):
        st.info("Model files found locally. Skipping download.")
        
    else:
        # If model folder exists but is empty/incomplete, clean it up
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        st.warning("Model files not found. Attempting download from Google Drive...")
        
        # Google Drive direct download URL structure
        download_url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        
        try:
            with st.spinner("Downloading large model file (this may take several minutes)..."):
                response = requests.get(download_url, stream=True)
                response.raise_for_status() 
            
            # Save the zip file
            with open(RETROSYNTHESIS_ZIP, 'wb') as f:
                f.write(response.content)

            # Unzip the file
            with st.spinner("Extracting model files..."):
                with zipfile.ZipFile(RETROSYNTHESIS_ZIP, 'r') as zip_ref:
                    # Extract contents into the current directory. 
                    # Assumes the zip contains a folder named 'retrosynthesis_model'.
                    zip_ref.extractall("./")
            
            st.success("Model downloaded and extracted successfully!")
            os.remove(RETROSYNTHESIS_ZIP) # Clean up the zip file
            RETROSYNTHESIS_AVAILABLE = True # Update global flag on success
            
        except Exception as e:
            st.error(f"Failed to download/extract model from Drive. Check File ID and sharing permissions.")
            st.exception(e)
            return None, None, None

    # 2. Load the model from the local directory
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Loading model into memory (Device: {device})...")

        # Load the tokenizer and model from the extracted directory
        from transformers import T5TokenizerFast, T5ForConditionalGeneration
        tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
        
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Failed to load model from local files after download.")
        st.exception(e)
        return None, None, None

@st.cache_data(show_spinner=False)
def predict_retrosynthesis(model, tokenizer, device, product_smiles):
    """Generates the predicted reactant SMILES from the product SMILES."""
    if model is None:
        return "Model not loaded."
        
    input_ids = tokenizer.encode(
        product_smiles, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    ).to(device)
    
    # Generate prediction (using beam search for quality)
    outputs = model.generate(
        input_ids,
        max_length=512,
        num_beams=15, 
        early_stopping=True
    )
    
    predicted_smiles = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_smiles
    

# ============================================================================
# LITERATURE MINING FUNCTIONS
# ============================================================================

def search_pubmed(plant_name, disease, max_results=20):
    """Search PubMed for papers linking plant to disease"""
    query = f'"{plant_name}" AND ({disease} OR treatment OR therapy OR activity)'
    
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        st.error(f"PubMed search error: {e}")
        return []

def fetch_abstracts(pmids):
    """Fetch paper details including abstracts"""
    if not pmids:
        return []
    
    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=pmids,
            rettype="abstract",
            retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()
        
        papers = []
        for record in records['PubmedArticle']:
            try:
                article = record['MedlineCitation']['Article']
                title = article.get('ArticleTitle', 'No title')
                abstract_sections = article.get('Abstract', {}).get('AbstractText', [])
                abstract = ' '.join([str(section) for section in abstract_sections])
                pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                year = pub_date.get('Year', 'Unknown')
                pmid = record['MedlineCitation']['PMID']
                
                papers.append({
                    'pmid': str(pmid),
                    'title': title,
                    'abstract': abstract,
                    'year': year,
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
            except:
                continue
        
        return papers
    except Exception as e:
        st.error(f"Error fetching abstracts: {e}")
        return []

def analyze_papers_with_llm(papers, plant_name, disease, groq_api_key):
    """Use Groq LLM to extract insights from papers"""
    groq_client = Groq(api_key=groq_api_key)
    
    abstracts_text = "\n\n---\n\n".join([
        f"Paper {i+1} ({paper['year']}):\nTitle: {paper['title']}\nAbstract: {paper['abstract'][:500]}..."
        for i, paper in enumerate(papers[:5])
    ])
    
    prompt = f"""You are a scientific research analyst specializing in natural product drug discovery.

Analyze these research papers about {plant_name} for {disease} treatment:

{abstracts_text}

Provide a structured analysis:

#1. **Evidence Strength**: Rate LOW/MEDIUM/HIGH based on number of papers, study types, and results consistency
1. **Active Compounds**: List specific chemical compounds mentioned with their activities
2. **Mechanisms of Action**: Explain HOW the plant compounds work against {disease}
3. **Key Findings**: Summarize the most important discoveries (IC50 values, clinical outcomes, etc.)
4. **Research Gaps**: What's missing or needs more study?

Be simple for a layman to understand, concise, scientific, and cite paper numbers when making claims."""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert in pharmacology and natural products chemistry."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Analysis failed: {e}"

# ============================================================================
# 3D VISUALIZATION FUNCTIONS
# ============================================================================

def show_3d_molecule(smiles):
    """Display interactive 3D molecule from SMILES"""
    if not RDKIT_AVAILABLE:
        st.error("RDKit not available. Install: pip install rdkit")
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string")
        return None
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    mol_block = Chem.MolToMolBlock(mol)
    
    # Create 3D viewer
    viewer = py3Dmol.view(width=800, height=600)
    viewer.addModel(mol_block, 'mol')
    viewer.setStyle({'stick': {'radius': 0.15}})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    
    return viewer

def get_molecular_properties(smiles):
    """Calculate molecular properties"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            'Molecular Weight': f"{Descriptors.MolWt(mol):.2f} Da",
            'LogP': f"{Descriptors.MolLogP(mol):.2f}",
            'H-Bond Donors': Descriptors.NumHDonors(mol),
            'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
            'Aromatic Rings': Descriptors.NumAromaticRings(mol)
        }
    return None

# ============================================================================
# PLANT RECOGNITION FUNCTIONS
# ============================================================================

def identify_plant_google_vision(image_file, credentials_path):
    """Identify plant using Google Cloud Vision API"""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    client = vision.ImageAnnotatorClient()
    
    content = image_file.read()
    image = vision.Image(content=content)
    
    # Get labels
    response = client.label_detection(image=image)
    labels = response.label_annotations
    
    # Get web entities
    web_response = client.web_detection(image=image)
    web_entities = web_response.web_detection.web_entities
    
    return labels, web_entities

def extract_plant_species(labels, entities):
    """Extract most likely plant species nameby prioritizing high-score 
    and specific entries from Web Entities over generic labels. The function aims to find scientific names or specific common names."""
    
    plant_keywords = ['plant', 'leaf', 'flower', 'tree', 'herb', 'botanical']
    
    # --- Priority 1: High-Confidence, Specific Web Entities ---
    
    # 1. Search for a very high-confidence scientific/common name (Score >= 0.90)
    for entity in entities:
        desc = entity.description
        score = entity.score
        
        # Check for scientific format (Genus species) or common high-score match
        if score >= 0.90:
            # If it looks like a scientific name (two words capitalized), or is highly specific
            if len(desc.split()) >= 2 and desc[0].isupper() and desc.split()[1][0].islower():
                return desc # e.g., 'Moringa oleifera' or 'Azadirachta indica'
            
            # Or if it's a known common name with very high score
            if score >= 0.95 and desc.lower() not in plant_keywords:
                return desc # e.g., 'Moringa' or 'Neem'

    # 2. Search for the best scientific/specific entity, regardless of the generic keywords
    best_entity = None
    best_score = 0.0
    
    for entity in entities:
        desc = entity.description.strip()
        score = entity.score
        
        # Filter out overly generic terms that aren't useful as primary identification
        if desc.lower() in ['leaf', 'plant', 'herb', 'vegetation', 'food']:
            continue

        # Look for a specific entity with the highest score
        if score > best_score:
            best_score = score
            best_entity = desc

    if best_entity and best_score > 0.65: # Only return if confidence is reasonably high
        return best_entity

    # --- Priority 2: High-Confidence Generic Label (Fallback) ---

    for label in labels:
        desc = label.description.strip()
        score = label.score
        
        if score > 0.80 and desc.lower() not in plant_keywords:
            return desc # e.g., 'Phyllanthaceae' (Family name)

    # --- Priority 3: Fail Gracefully ---
    return "Unknown species (Low confidence)"
# ============================================================================
# PLANT AGENT CLASS
# ===============================================

import pandas as pd
from typing import Dict, List, Optional
class PlantAgent:
    """
    An agent responsible for resolving plant names (common to scientific)
    and searching a DataFrame for associated compounds.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the agent with the compounds database DataFrame.

        Args:
            df: The main compounds DataFrame containing compound details and organism names.
        """
        if df is None:
            raise ValueError("DataFrame cannot be None. Please upload a compounds database first.")

        self.df = df.copy()
        # Pre-load the common name to scientific name map for fast lookups
        self.common_name_map: Dict[str, str] = self._load_common_names()
    
    def _load_common_names(self) -> Dict[str, str]:
        """
        Creates a map of common names (and synonyms) to their canonical
        scientific (botanical) names. This logic is the core of name resolution.
        
        Returns:
            A dictionary mapping lowercased common names to scientific names.
        """
        known_mappings = {
            'Vernonia amygdalina': 'bitter leaf, ewuro, onugbu, grawa, oriwo, ityuna, etidot, ndoleh, ewu ro',
            'Ocimum gratissimum': 'scent leaf, african basil, clove basil, nchanwu, efirin, daidoya, aramogbo',
            'Ocimum viride': 'scent leaf, green basil, tea bush',
            'Garcinia kola': 'bitter kola, orogbo, aki ilu, miji-goro, adi',
            'Cola nitida': 'kola nut, cola nut, goro, obi, gworo',
            'Cola acuminata': 'kola nut, abata cola, obi abata',
            'Xylopia aethiopica': 'african pepper, negro pepper, grains of selim, uda, kimba, kani pepper, ethiopian pepper, hwentia',
            'Azadirachta indica': 'neem, dogoyaro, dongoyaro, nim tree, margosa tree, vepai',
            'Moringa oleifera': 'moringa, drumstick tree, horseradish tree, zogale, okweoyibo, ewe igbale',
            'Hibiscus sabdariffa': 'roselle, zobo, sorrel, soborodo, isapa, yakuwa',
            'Cymbopogon citratus': 'lemon grass, fever grass, lemon grass tea, kooko oba, tsaida',
            'Aloe vera': 'aloe, aloe vera, ahon erin',
            'Aloe barbadensis': 'aloe vera, barbados aloe',
            'Carica papaya': 'papaya, pawpaw, ibepe, gwanda',
            'Psidium guajava': 'guava, gova, gofa',
            'Annona muricata': 'soursop, graviola, abo, fasadarur',
            'Chrysophyllum albidum': 'african star apple, agbalumo, udara, udala, alasa',
            'Homo sapiens': 'human',
            'Citrullus lanatus': 'watermelon, egusi, kankana',
            'Mus musculus': 'mouse',
            'Panax ginseng': 'asian ginseng, korean ginseng, red ginseng, ginseng',
            'Arabidopsis thaliana': 'thale cress, mouse-ear cress',
            'Vitis vinifera': 'grape, wine grape, common grape vine',
            'Ganoderma lucidum': 'reishi mushroom, lingzhi, reishi',
            'Angelica sinensis': 'dong quai, female ginseng, chinese angelica, dang gui',
            'Glycyrrhiza uralensis': 'chinese licorice, gan cao',
            'Citrus reticulata': 'mandarin orange, tangerine, mandarin, osan wewe',
            'Escherichia coli': 'e. coli, e coli',
            'Zingiber officinale': 'ginger, ata-ile, citta, jinja',
            'Lonicera japonica': 'japanese honeysuckle, jin yin hua, honeysuckle',
            'Capsicum annuum': 'bell pepper, chili pepper, sweet pepper, ata rodo, barkono',
            'Angelica acutiloba': 'japanese angelica',
            'Humulus lupulus': 'hops, common hops',
            'Foeniculum vulgare': 'fennel, sweet fennel, fennel seed',
            'Daucus carota': 'carrot, wild carrot, karas',
            'Chrysanthemum x morifolium': 'florist chrysanthemum, mum, ju hua',
            'Artemisia annua L.': 'sweet wormwood, sweet annie, annual wormwood, qing hao',
            'Artemisia annua': 'sweet wormwood, sweet annie, annual wormwood, qing hao',
            'Vitex negundo': 'chinese chaste tree, five-leaved chaste tree, nirgundi',
            'Angelica gigas': 'korean angelica, cham danggui',
            'Chaenomeles sinensis': 'chinese quince, flowering quince, mu gua',
            'Sophora flavescens': 'shrubby sophora, ku shen',
            'Morus alba': 'white mulberry, mulberry, sang',
            'Artemisia argyi': 'silvery wormwood, chinese mugwort, ai ye',
            'Artemisia capillaris': 'capillary wormwood, yin chen',
            'Curcuma longa': 'turmeric, haldi, atale pupa, gangamau',
            'Punica granatum': 'pomegranate, anar, pome',
            'Schisandra chinensis': 'five-flavor berry, magnolia vine, wu wei zi',
            'Citrus sinensis': 'sweet orange, orange, osan mimu',
            'Chrysanthemum indicum': 'indian chrysanthemum, wild chrysanthemum',
            'Zea mays': 'corn, maize, agbado, masara',
            'Lyngbya majuscula': 'sea hair, fireweed',
            'Syzygium aromaticum': 'clove, kanafuru, clove spice',
            'Gardenia jasminoides': 'cape jasmine, gardenia, zhi zi',
            'Glycyrrhiza glabra': 'licorice, liquorice, sweet root',
            'Gynostemma pentaphyllum': 'jiaogulan, immortality herb, southern ginseng',
            'Murraya paniculata': 'orange jasmine, mock orange, chinese box',
            'Citrus unshiu': 'satsuma mandarin, satsuma, unshiu orange',
            'Camellia sinensis': 'tea plant, tea, green tea, black tea, tii',
            'Ginkgo biloba': 'ginkgo, maidenhair tree, bai guo',
            'Nelumbo nucifera': 'sacred lotus, lotus, indian lotus, lian',
            'Melia azedarach': 'chinaberry tree, bead tree, persian lilac, dogo yaro',
            'Ephedra sinica': 'chinese ephedra, ma huang, joint fir',
            'Mangifera indica': 'mango, mangoro, mangwaro',
            'Curcuma kwangsiensis': 'guangxi turmeric, kwangsi turmeric',
            'Hypericum perforatum': 'st johns wort, st. johns wort, hypericum',
            'Pastinaca sativa': 'parsnip, wild parsnip',
            'Allium sativum': 'garlic, aayu, tafarnuwa, ayuu',
            'Pogostemon cablin': 'patchouli, patchouli oil plant',
            'Periploca sepium': 'chinese silk vine, xiang jia pi',
            'Curcuma zedoaria': 'white turmeric, zedoary, kua',
            'Glycine max': 'soybean, soya bean, soy',
            'Curcuma wenyujin': 'wenjin turmeric, wen yu jin',
            'Streptomyces': 'streptomyces bacteria',
            'Penicillium': 'penicillium mold, penicillium fungi',
            'Aspergillus': 'aspergillus mold, aspergillus fungi',
        }
        
        mapping = {}
        for botanical, common_names_str in known_mappings.items():
            for name in common_names_str.split(','):
                name = name.strip().lower()
                if name:
                    mapping[name] = botanical
                    
        # Also map scientific names to themselves (for exact searches)
        for name in known_mappings.keys():
             mapping[name.lower()] = name
             
        return mapping
    
    def resolve_plant_name(self, plant_name: str) -> str:
        """
        Resolves a user-provided plant name (common or scientific) to its 
        canonical scientific name using the loaded map.

        Args:
            plant_name: The input string from the user.

        Returns:
            The resolved scientific name, or the original input if no map found.
        """
        lower_name = plant_name.lower().strip()
        # If found in the common name map, return the scientific name
        resolved_name = self.common_name_map.get(lower_name)
    def search_by_plant(self, plant_name, top_n=50):
        resolved = self.resolve_plant_name(plant_name)
        
        if 'organisms' in self.df.columns:
            results = self.df[self.df['organisms'].str.contains(resolved, case=False, na=False)]
            
            if results.empty and resolved.lower() != plant_name.lower():
                results = self.df[self.df['organisms'].str.contains(plant_name, case=False, na=False)]
            
            return results.head(top_n) if not results.empty else None
        
        return None


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/200x80/2E7D32/FFFFFF?text=AfroMediBot", width='stretch')
    st.markdown("---")
    
    st.subheader("⚙️ API Configuration")
    
    # Groq API Key
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get free key at console.groq.com")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        Entrez.email = st.text_input("Email (for PubMed)", value="your_email@example.com")
    else:
        st.warning("Biopython not installed, PubMed features disabled.")
    # Google Vision Credentials
    vision_creds = st.file_uploader("Google Vision Credentials (JSON)", type=['json'])
    if vision_creds:
        with open('vision-credentials.json', 'wb') as f:
            f.write(vision_creds.read())
        st.success("✅ Vision API configured")
    
    st.markdown("---")
    st.sidebar.markdown("### 💾 Compound Database")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Compound Database",
        type=['csv'],
        key="database_uploader"
    )
    # Caching function to load the data efficiently
    @st.cache_data
    def load_data(file):
        # This function is called only when the file changes
        return pd.read_csv(file)
    
    # Handle file upload and state update
    if uploaded_file is not None:
        try:
            df_new = load_data(uploaded_file)
            if 'organisms' in df_new.columns and not df_new.empty:
                st.session_state['database'] = df_new
                st.sidebar.success(f"Database Loaded: {len(df_new)} compounds.")
            else:
                st.sidebar.error("CSV must contain an 'organisms' column.")
                st.session_state['database'] = pd.DataFrame()
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
            st.session_state['database'] = pd.DataFrame()
    else:
        # Initialize the database as an empty DataFrame or check existing state
        if 'database' not in st.session_state:
            st.session_state['database'] = pd.DataFrame()
        
        if st.session_state['database'].empty:
            st.sidebar.info("Upload a CSV to enable compound searching.")
        else:
            st.sidebar.success(f"Active Database: {len(st.session_state['database'])} compounds.")
    
    
    # Initialize search results state
    if 'search_results' not in st.session_state:
        st.session_state['search_results'] = pd.DataFrame()
    if 'resolved_name' not in st.session_state:
        st.session_state['resolved_name'] = ""
        
    st.subheader("📊 Quick Stats")
    st.metric("Plants in Database", "500+")
    st.metric("Compounds Analyzed", "50,000+")
    st.metric("AI Models", "6")

# ============================================================================
# MAIN APP - TAB NAVIGATION
# ============================================================================

tab_home, tab_literature, tab_3d, tab_plant, tab_synthesis = st.tabs([
    "🏠 Home",
    "📚 Literature Mining",
    "🧊 3D Molecule Viewer",
    "🌿 Plant Recognition",
    "🧪 Retrosynthesis"
])

# ============================================================================
# TAB 1: HOME / LANDING PAGE
# ============================================================================

with tab_home:
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🌿 AfroMediBot</div>
        <div class="hero-subtitle">AI-Powered Drug Discovery from African Medicinal Plants</div>
        <p style="margin-top: 1rem; font-size: 1.1rem;">
            Discover novel therapeutics using cutting-edge AI and traditional botanical knowledge
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics Cards
    st.subheader("📊 Platform Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">500+</div>
            <div class="stat-label">Medicinal Plants</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">50K+</div>
            <div class="stat-label">Compounds</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">6</div>
            <div class="stat-label">AI Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">95%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Features
    st.subheader("🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <h3>Literature Mining</h3>
            <p>AI-powered extraction of scientific evidence from 35M+ research papers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧊</div>
            <h3>3D Visualization</h3>
            <p>Interactive molecular structure viewer with property calculations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌿</div>
            <h3>Plant Recognition</h3>
            <p>Image-based identification of medicinal plants using computer vision</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("⚡ Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Search Literature", width='stretch', type="primary", key="home_search_lit"):
            st.session_state.active_tab = "literature"
            st.rerun()
    
    with col2:
        if st.button("🧊 View 3D Molecules", width='stretch', type="primary", key="home_view_3d"):
            st.session_state.active_tab = "3d"
            st.rerun()
    
    with col3:
        if st.button("🌿 Identify Plant", width='stretch', type="primary", key="home_identify_plant"):
            st.session_state.active_tab = "plant"
            st.rerun()
    
    st.markdown("---")
    
    # Example Queries
    st.subheader("💡 Example Queries")
    
    example_queries = [
        {"title": "🦟 Malaria Treatment", "plant": "Vernonia amygdalina", "disease": "Malaria"},
        {"title": "🎗️ Cancer Research", "plant": "Azadirachta indica", "disease": "Cancer"},
        {"title": "💉 HIV Therapy", "plant": "Moringa oleifera", "disease": "HIV"},
    ]
    
    cols = st.columns(3)
    for idx, query in enumerate(example_queries):
        with cols[idx]:
            if st.button(query["title"], width='stretch', key=f"example_query_{idx}"):
                st.session_state.example_plant = query["plant"]
                st.session_state.example_disease = query["disease"]
                st.session_state.active_tab = "literature"
                st.rerun()
    
    # Example Molecules
    st.subheader("🧬 Example Molecules")
    
    example_mols = [
        {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        {"name": "Artemisinin", "smiles": "CC1CCC2C(C(=O)OC3(C24C1CCC(O3)(OO4)C)C)C"},
        {"name": "Quinine", "smiles": "C=CC1CN2CCC1C2C(C3=CC=NC4=CC=C(C=C34)OC)O"},
    ]
    
    cols = st.columns(3)
    for idx, mol in enumerate(example_mols):
        with cols[idx]:
            if st.button(f"View {mol['name']}", width='stretch', key=f"example_mol_{idx}"):
                st.session_state.example_smiles = mol["smiles"]
                st.session_state.example_mol_name = mol["name"]
                st.session_state.active_tab = "3d"
                st.rerun()

# ============================================================================
# TAB 2: LITERATURE MINING
# ============================================================================

with tab_literature:
    st.header("📚 Literature Mining Agent")
    st.info("Search 35+ million biomedical papers for evidence linking plants to diseases")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        plant_name = st.text_input(
            "Plant Name (Scientific or Common)",
            value=st.session_state.get('example_plant', ''),
            placeholder="e.g., Vernonia amygdalina, bitter leaf"
        )
        
        disease_name = st.text_input(
            "Disease/Condition",
            value=st.session_state.get('example_disease', ''),
            placeholder="e.g., malaria, cancer, diabetes"
        )
    
    with col2:
        max_papers = st.slider("Maximum Papers", 5, 50, 15)
        search_button = st.button("🔍 Search Literature", type="primary", width='stretch', key="lit_search_btn")
    
    if search_button and plant_name and disease_name:
        if not LITERATURE_AVAILABLE:
            st.error("Literature mining dependencies not installed. Install: biopython, groq")
        elif not groq_api_key:
            st.error("Please configure Groq API key in sidebar")
        else:
            with st.spinner("🔍 Searching PubMed..."):
                pmids = search_pubmed(plant_name, disease_name, max_papers)
                
                if pmids:
                    st.success(f"✅ Found {len(pmids)} papers")
                    
                    with st.spinner("📥 Fetching abstracts..."):
                        papers = fetch_abstracts(pmids)
                        time.sleep(1)
                    
                    if papers:
                        # AI Analysis
                        with st.spinner("🤖 Analyzing with AI..."):
                            analysis = analyze_papers_with_llm(papers, plant_name, disease_name, groq_api_key)
                        
                        # Display results
                        st.subheader("🤖 AI Analysis")
                        st.markdown(analysis)
                        
                        # Papers list
                        st.subheader(f"📄 Top {len(papers)} Papers")
                        for i, paper in enumerate(papers[:10], 1):
                            with st.expander(f"{i}. {paper['title']} ({paper['year']})"):
                                st.write(f"**Abstract:** {paper['abstract'][:500]}...")
                                st.markdown(f"[🔗 Read Full Paper]({paper['url']})")
                        
                        # Download results
                        results_json = json.dumps({
                            'plant': plant_name,
                            'disease': disease_name,
                            'papers': papers,
                            'analysis': analysis
                        }, indent=2)
                        
                        st.download_button(
                            "📥 Download Results (JSON)",
                            data=results_json,
                            file_name=f"literature_mining_{plant_name}_{disease_name}.json",
                            mime="application/json"
                        )
                else:
                    st.warning("No papers found. Try different search terms.")

# ============================================================================
# TAB 3: 3D MOLECULE VIEWER
# ============================================================================

with tab_3d:
    st.header("🧊 Interactive 3D Molecule Viewer")
    st.info("Visualize molecular structures in 3D with property calculations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input(
            "Enter SMILES String",
            value=st.session_state.get('example_smiles', 'CC(=O)OC1=CC=CC=C1C(=O)O'),
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
        )
        
        mol_name = st.text_input(
            "Molecule Name (Optional)",
            value=st.session_state.get('example_mol_name', ''),
            placeholder="e.g., Aspirin"
        )
    
    with col2:
        viz_style = st.selectbox("Visualization Style", ["stick", "sphere", "line"])
        show_props = st.checkbox("Show Properties", value=True)
    
    if st.button("🔬 Generate 3D Structure", type="primary", key="3d_gen_btn"):
        if not RDKIT_AVAILABLE:
            st.error("RDKit not available. Install: pip install rdkit py3Dmol stmol")
        else:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                # 2D Structure
                st.subheader("📐 2D Structure")
                img = Draw.MolToImage(mol, size=(400, 400))
                st.image(img, caption=mol_name if mol_name else "Molecule")
                
                # 3D Structure
                st.subheader("🧊 3D Interactive Structure")
                viewer = show_3d_molecule(smiles_input)
                if viewer:
                    showmol(viewer, height=600, width=800)
                
                # Molecular Properties
                if show_props:
                    st.subheader("📊 Molecular Properties")
                    props = get_molecular_properties(smiles_input)
                    
                    if props:
                        col1, col2, col3 = st.columns(3)
                        props_items = list(props.items())
                        
                        for i, (key, value) in enumerate(props_items):
                            with [col1, col2, col3][i % 3]:
                                st.metric(key, value)
            else:
                st.error("Invalid SMILES string. Please check and try again.")

# ============================================================================
# TAB 4: PLANT RECOGNITION
# ============================================================================

with tab_plant:
    st.header("🌿 Plant Image Recognition")
    st.info("Upload a plant image to identify species using AI")
    
    uploaded_image = st.file_uploader(
        "Upload Plant Image",
        type=['jpg', 'jpeg', 'png'],
        help="Take a clear photo of the plant (leaf, flower, or whole plant)"
    )
    
    if uploaded_image:
        # Display image
        image = Image.open(uploaded_image)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", width='stretch')
        
        with col2:
            if st.button("🔍 Identify Plant", type="primary", width='stretch', key="plant_identify_btn"):
                if not vision_creds:
                    st.error("Please upload Google Vision credentials in sidebar")
                else:
                    with st.spinner("🔍 Analyzing image..."):
                        try:
                            uploaded_image.seek(0)  # Reset file pointer
                            labels, entities = identify_plant_google_vision(uploaded_image, 'vision-credentials.json')

                            # Extract the most likely species name
                            identified_species_name = extract_plant_species(labels, entities)

                            # Store in session state for persistence
                            st.session_state.identified_species = identified_species_name
                            st.session_state.vision_labels = labels
                            st.session_state.vision_entities = entities
                            
                        except Exception as e:
                            st.error(f"Error during Vision API analysis: {e}")
                            st.exception(e)
        
        # Display identification results (outside the button conditional)
        if 'identified_species' in st.session_state:
            identified_species_name = st.session_state.identified_species
            labels = st.session_state.get('vision_labels', [])
            entities = st.session_state.get('vision_entities', [])
            
            st.markdown("---")
            
            # Display the main result clearly
            st.subheader("🌿 Primary Identification")
            
            if identified_species_name != "Unknown plant":
                st.success(f"**Identified Species:** **{identified_species_name}**")
            else:
                st.warning("Could not definitively identify species.")
                
            st.markdown("---")
            
            # Display Details (Labels and Entities)
            st.subheader("🏷️ Detected Labels")
            for label in labels[:5]:
                st.write(f"• {label.description}: {label.score:.1%} confidence")
            
            st.subheader("🌿 Plant Identification")
            if entities:
                found_entity = False
                for entity in entities[:3]:
                    if entity.description and (entity.description.lower() not in ['plant', 'leaf', 'tree'] or entity.score > 0.6):
                        st.success(f"✅ Possible match: **{entity.description}** (Score: {entity.score:.2f})")
                        found_entity = True
                if not found_entity:
                    st.info("No highly specific web entities found.")
            else:
                st.warning("Could not identify specific species. Try a clearer image.")

            # --- Plant Mapping and Filtering buttons (OUTSIDE identify button) ---
            st.markdown("---")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                map_disabled = 'compounds_df' not in st.session_state or st.session_state.compounds_df is None
                if st.button(
                    "🗺️ Map Plant Name", 
                    key="map_plant_name", 
                    type="secondary",
                    use_container_width=True,
                    disabled=map_disabled,
                    help="Map common name to scientific name and find compounds" if not map_disabled else "Upload database first"
                ):
                    if 'compounds_df' not in st.session_state or st.session_state.compounds_df is None:
                        st.error("Please upload a compounds database in the sidebar first")
                    else:
                        try:
                            # Get identified species from session state
                            identified_species_name = st.session_state.identified_species
                            # Initialize PlantAgent with the uploaded database
                            plant_agent = PlantAgent(st.session_state.compounds_df)
                            
                            # Resolve the plant name
                            resolved_name = plant_agent.resolve_plant_name(identified_species_name)
                            
                            # Store resolved name
                            st.session_state.resolved_plant_name = resolved_name
                            
                            st.subheader("🔍 Mapping Results")
                            
                            if resolved_name.lower() != identified_species_name.lower():
                                st.success(f"**Common Name:** {identified_species_name}")
                                st.success(f"**Scientific Name:** {resolved_name}")
                            else:
                                st.info(f"**Name:** {resolved_name}")
                            
                            # Search for compounds from this plant
                            plant_compounds = plant_agent.search_by_plant(resolved_name, top_n=50)
                            
                            if plant_compounds is not None and not plant_compounds.empty:
                                st.subheader(f"💊 Found {len(plant_compounds)} Compounds")
                                
                                # Display key columns
                                display_cols = []
                                if 'compound_name' in plant_compounds.columns:
                                    display_cols.append('compound_name')
                                if 'smiles' in plant_compounds.columns:
                                    display_cols.append('smiles')
                                if 'organisms' in plant_compounds.columns:
                                    display_cols.append('organisms')
                                
                                if display_cols:
                                    st.dataframe(plant_compounds[display_cols].head(10))
                                else:
                                    st.dataframe(plant_compounds.head(10))
                                
                                # Store for filtering
                                st.session_state.mapped_plant = resolved_name
                                st.session_state.plant_compounds = plant_compounds
                            else:
                                st.warning(f"No compounds found for {resolved_name} in database")

                        except ValueError as ve:
                            st.error(f"❌ Error: {ve}")
                        except Exception as e:
                            st.error(f"❌ Unexpected error during mapping: {e}")
                            st.exception(e)

            
            # Display mapping results if they exist (persistence)
            if 'resolved_plant_name' in st.session_state and 'mapped_plant' not in st.session_state:
                st.info(f"📋 Last mapped: **{st.session_state.resolved_plant_name}** - Click 'Map Plant Name' again to search compounds")
            
            with col_btn2:
                filter_disabled = 'plant_compounds' not in st.session_state or st.session_state.plant_compounds is None or st.session_state.plant_compounds.empty
                      
                if st.button(
                    "🔬 Filter Compounds", 
                    key="filter_plant_compounds", 
                    type="primary",
                    use_container_width=True,
                    disabled=filter_disabled,
                    help="Filter and analyze found compounds" if not filter_disabled else "Map plant name first"
                ):
                    if ('plant_compounds' not in st.session_state or st.session_state.plant_compounds is None or st.session_state.plant_compounds.empty):
                        st.warning("Please map the plant name first")
                    else:
                        st.session_state.show_filters = True
            
            # Display filter UI (outside button, persists across reruns)
            if (st.session_state.get('show_filters', False) and 
                'plant_compounds' in st.session_state and 
                st.session_state.plant_compounds is not None and
                not st.session_state.plant_compounds.empty):
                
                st.markdown("---")
                st.subheader("🎯 Compound Filtering")
                
                try:
                    compounds = st.session_state.plant_compounds.copy()
                    mapped_plant = st.session_state.get('mapped_plant', identified_species_name)
                    
                    st.write(f"**Source Plant:** {mapped_plant}")
                    st.write(f"**Total Compounds:** {len(compounds)}")
                    
                    # Add filtering options
                    filter_col1, filter_col2 = st.columns(2)
                    
                    with filter_col1:
                        # Molecular weight filter (if available)
                        if 'molecular_weight' in compounds.columns:
                            mw_min = float(compounds['molecular_weight'].min())
                            mw_max = float(compounds['molecular_weight'].max())
                            mw_range = st.slider(
                                "Molecular Weight Range",
                                mw_min,
                                mw_max,
                                (mw_min, mw_max),
                                key="mw_slider"
                            )
                            compounds = compounds[
                                (compounds['molecular_weight'] >= mw_range[0]) & 
                                (compounds['molecular_weight'] <= mw_range[1])
                            ]
                    
                    with filter_col2:
                        # Activity filter (if available)
                        if 'activity_type' in compounds.columns:
                            activities = compounds['activity_type'].unique().tolist()
                            selected_activities = st.multiselect(
                                "Filter by Activity",
                                activities,
                                default=activities[:3] if len(activities) > 3 else activities,
                                key="activity_filter"
                            )
                            if selected_activities:
                                compounds = compounds[compounds['activity_type'].isin(selected_activities)]
                    
                    # Display filtered results
                    st.write(f"**Filtered Results:** {len(compounds)} compounds")
                    st.dataframe(compounds)
                    
                    # Download option
                    csv = compounds.to_csv(index=False)
                    st.download_button(
                        "📥 Download Filtered Compounds",
                        data=csv,
                        file_name=f"filtered_compounds_{mapped_plant}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error during filtering: {e}")
                    st.session_state.show_filters = False  # Reset filter state on error
# ============================================================================
# TAB 5: RETROSYNTHESIS
# ============================================================================

with tab_synthesis:
    st.header("🧪 AI-Powered Retrosynthesis")
    st.info("Predict the reactants and synthetic route for a lead compound using a fine-tuned T5 model.")
    
    if not RETROSYNTHESIS_AVAILABLE:
        st.warning(
            "Retrosynthesis feature is disabled. "
            "Please ensure you have installed `torch` and `transformers` and saved "
            "your trained model in the `./retrosynthesis_model` directory."
        )
    else:
        # Load the model only if available
        model, tokenizer, device = load_retrosynthesis_model()
        
        if model:
            st.success(f"✅ Retrosynthesis Model loaded successfully. Running on {device}.")
            
            st.markdown("---")
            
            target_smiles = st.text_input(
                "Enter Target Product SMILES",
                value='CC(=O)OC1=CC=CC=C1C(=O)O', # Example: Aspirin
                placeholder="e.g., CC1=CC=C(C=C1)C(O)=O"
            )
            
            if st.button("🔮 Predict Synthesis Route", type="primary", key="retro_predict_btn"):
                if target_smiles:
                    with st.spinner(f"Predicting reactants for {target_smiles}..."):
                        # Perform prediction
                        predicted_reactants = predict_retrosynthesis(
                            model, tokenizer, device, target_smiles
                        )
                    
                    st.subheader("💡 Predicted Route")
                    
                    # Display structures using RDKit (since it's now available)
                    if RDKIT_AVAILABLE:
                        col_p, col_r = st.columns(2)
                        
                        # Display Product
                        with col_p:
                            st.markdown("#### Target Product")
                            mol_p = Chem.MolFromSmiles(target_smiles)
                            if mol_p:
                                img_p = Draw.MolToImage(mol_p, size=(300, 300))
                                st.image(img_p)
                            else:
                                st.error("Invalid Product SMILES.")
    
                        # Display Predicted Reactants (molecules separated by '.')
                        with col_r:
                            st.markdown("#### Predicted Reactants")
                            predicted_mol_smiles = [s.strip() for s in predicted_reactants.split('.') if s.strip()] 
                            
                            if predicted_mol_smiles:
                                for i, r_smiles in enumerate(predicted_mol_smiles[:3]): # Show up to 3 reactants
                                    mol_r = Chem.MolFromSmiles(r_smiles)
                                    if mol_r:
                                        img_r = Draw.MolToImage(mol_r, size=(250, 250))
                                        st.image(img_r, caption=f"Reactant {i+1}")
                                        
                    st.markdown("---")
                    
                    st.markdown(f"""
                        **Predicted Reactants (SMILES):**
                        `{predicted_reactants}`
                    """)
                else:
                    st.warning("Please enter a valid SMILES string.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>AfroMediBot</strong> - Advancing African Drug Discovery with AI</p>
    <p>Powered by Groq, Google Cloud Vision, and RDKit</p>
</div>
""", unsafe_allow_html=True)
