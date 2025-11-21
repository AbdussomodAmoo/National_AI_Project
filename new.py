# app.py - Drug Discovery Screening App (Refactored)
import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================
try:
    from groq import Groq
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from gtts import gTTS
    PDF_AUDIO_AVAILABLE = True
except ImportError:
    PDF_AUDIO_AVAILABLE = False

# ============================================================================
# LLM INTEGRATION (GroqClient & Expert Analysis)
# ============================================================================
class GroqClient:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key) if LLM_AVAILABLE and api_key else None

    def generate_expert_analysis(self, results_df, query_text):
        if not self.client:
            return "LLM service unavailable. Please check the Groq API key."

        # Prepare context for the LLM
        top_candidates_markdown = "Name | MW (Da) | QED Score | LogP\n---|---|---|---\n"
        for _, row in results_df.head(10).iterrows():
            name = str(row.get('name', 'Unknown'))
            mw = f"{row.get('molecular_weight', 0):.1f}"
            qed = f"{row.get('qed_drug_likeliness', 0):.3f}"
            logp = f"{row.get('alogp', 0):.2f}"
            top_candidates_markdown += f"{name} | {mw} | {qed} | {logp}\n"

        system_prompt = f"""You are AfroMediBot, an AI Expert in natural products drug discovery. 
Your task is to provide a concise, professional expert analysis report based on the screening results.
The analysis should focus on the top candidates' drug-likeness and potential for the target {query_text}.

Key Data:
- Total Candidates: {len(results_df)}
- Drug-like Candidates (QED >= 0.5): {results_df.get('qed_drug_likeliness', pd.Series([0])).apply(lambda x: x >= 0.5 if pd.notna(x) else False).sum()}
- Query: {query_text}

Top 10 Candidates Data (for analysis):
{top_candidates_markdown}

Structure the response with markdown headings:
## 🔬 Expert Analysis Report
### 1. Summary of Screening Results
### 2. Physicochemical Assessment (Key trends in MW, LogP, QED)
### 3. Lead Candidate Recommendation
### 4. Next Steps (In vitro, in vivo, or synthesis recommendations)

Be concise, scientific, and highlight the most promising molecule(s).
"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate the expert analysis report for the screening: {query_text}."}
                ],
                temperature=0.3,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"LLM Error: {e}")
            return None


# ============================================================================
# UTILITIES & AGENTS (Retained from original code)
# ============================================================================
@st.cache_data
def get_groq_api_key():
    """Retrieves and displays the Groq API key in the sidebar."""
    with st.sidebar:
        st.header("⚙️ LLM Settings")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Paste your Groq API Key here",
            help="Required for 'Expert Analysis' tab.",
            key="groq_api_key_input"
        )
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            st.success("API Key Loaded!")
        return api_key

@st.cache_data
def load_prefiltered_database():
    """Load your pre-filtered database that ships with the app"""
    paths = [
        'afrodb_filtered_strict.csv',
        'data/afrodb_filtered_strict.csv',
        'filtered_compounds.csv'
    ]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
    return None

class PlantAgent:
    def __init__(self, df):
        self.df = df
        # ... (Keep _load_common_names and resolve_plant_name methods as they are) ...
        self.common_name_map = self._load_common_names()
    
    def _load_common_names(self):
        # ... (Keep your full existing common_name_map dictionary) ...
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
        return mapping
    
    def resolve_plant_name(self, plant_name):
        lower_name = plant_name.lower()
        return self.common_name_map.get(lower_name, plant_name)
    
    def search_by_plant(self, plant_name, top_n=50):
        resolved = self.resolve_plant_name(plant_name)
        
        if 'organisms' in self.df.columns:
            results = self.df[self.df['organisms'].str.contains(resolved, case=False, na=False)]
            
            if results.empty and resolved.lower() != plant_name.lower():
                results = self.df[self.df['organisms'].str.contains(plant_name, case=False, na=False)]
            
            return results.head(top_n) if not results.empty else None
        
        return None

class PredictorAgent:
    # ... (Keep all PredictorAgent methods as they are) ...
    def __init__(self):
        self.models = {}
    
    def featurize(self, smiles):
        """Your exact featurization"""
        if not RDKIT_AVAILABLE:
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        feature_dict = {
            'MolWt': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        }
        
        return feature_dict
    
    def predict_druglikeness(self, smiles):
        """Quick drug-likeness check"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
        
        return {
            'lipinski_pass': lipinski,
            'molecular_weight': mw,
            'logp': logp,
            'hbd': hbd,
            'hba': hba
        }

class FilterAgent:
    # ... (Keep all FilterAgent methods as they are) ...
    def __init__(self, df):
        self.df = df.copy()
        if RDKIT_AVAILABLE:
            self.pains_params = FilterCatalogParams()
            self.pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_catalog = FilterCatalog.FilterCatalog(self.pains_params)

    def lipinski_filter(self, row):
        try:
            mw = row['molecular_weight']
            logp = row['alogp']
            hba = row['hydrogen_bond_acceptors']
            hbd = row['hydrogen_bond_donors']
            rules = [mw <= 500, logp <= 5, hba <= 10, hbd <= 5]
            return sum(rules) >= 3
        except:
            return False

    def veber_filter(self, row):
        try:
            rotatable_bonds = row['rotatable_bond_count']
            tpsa = row['topological_polar_surface_area']
            return (rotatable_bonds <= 10) and (tpsa <= 140)
        except:
            return False

    def pains_filter(self, smiles):
        if not RDKIT_AVAILABLE:
            return True
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            return not self.pains_catalog.HasMatch(mol)
        except:
            return False

    def apply_filters(self, filter_mode='drug_like', qed_threshold=0.5, 
                      np_threshold=0.3, apply_pains=True):
        
        self.df['lipinski_pass'] = self.df.apply(self.lipinski_filter, axis=1)
        self.df['veber_pass'] = self.df.apply(self.veber_filter, axis=1)
        self.df['qed_pass'] = self.df['qed_drug_likeliness'] >= qed_threshold
        self.df['np_pass'] = self.df['np_likeness'] >= np_threshold
        
        base_filters = (
            self.df['lipinski_pass'] &
            self.df['veber_pass'] &
            self.df['qed_pass'] &
            self.df['np_pass']
        )
        
        if apply_pains:
            self.df['pains_pass'] = self.df['canonical_smiles'].apply(self.pains_filter)
            base_filters = base_filters & self.df['pains_pass']
        
        filtered_df = self.df[base_filters].copy()
        return filtered_df
    
# ============================================================================
# NEW CORE SCREENING FUNCTION
# ============================================================================
def screen_plant_compounds(plant_name, disease_target, df_db, plant_agent):
    """Executes the core search and filtering logic."""
    
    st.info(f"🌿 Screening {plant_name} for {disease_target}...")
    
    # 1. Search by Plant
    compounds = plant_agent.search_by_plant(plant_name, top_n=100)
    
    if compounds is None or compounds.empty:
        st.warning(f"❌ No compounds found in the database for **{plant_name}**. Try a different name.")
        return None, None
    
    # 2. Apply Drug-likeness Filters
    filter_agent = FilterAgent(compounds)
    results_df = filter_agent.apply_filters()
    
    if results_df.empty:
        st.warning(f"⚠️ Found {len(compounds)} compounds, but none passed strict drug-likeness filters (QED >= 0.5) for **{plant_name}**.")
        return None, None
        
    # Sort by QED for ranking
    if 'qed_drug_likeliness' in results_df.columns:
        results_df = results_df.sort_values(by='qed_drug_likeliness', ascending=False)
        
    query_text = f"Candidates from {plant_name} for {disease_target}"
    return results_df, query_text

def view_full_database(df):
    """Applies a default drug-likeness filter to the entire database for initial viewing."""
    if df is None:
        st.error("❌ No compound database loaded.")
        return None, None
    
    # Apply standard filters to the entire database
    filter_agent = FilterAgent(df)
    filtered_df = filter_agent.apply_filters()
    
    st.info(f"🌿 Database loaded. Displaying **{len(filtered_df)}** drug-like compounds (QED > 0.5) from **{len(df)}** total entries.")
    
    # Sort by QED for ranking
    if 'qed_drug_likeliness' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='qed_drug_likeliness', ascending=False)
        
    query_text = "Full Prefiltered Database View"
    return filtered_df, query_text
    
# ============================================================================
# PDF & AUDIO GENERATION (Retained from original code)
# ============================================================================
def generate_pdf_report(results_df, query):
    # ... (Keep the exact code for generate_pdf_report) ...
    if not PDF_AUDIO_AVAILABLE:
        return None
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    
    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, y, "AfroMediBot Drug Discovery Report")
    y -= 15
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30
    
    # Executive Summary Box
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(40, y-60, width-80, 60, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y-15, "EXECUTIVE SUMMARY")
    c.setFont("Helvetica", 10)
    c.drawString(50, y-30, f"Query: {query}")
    c.drawString(50, y-45, f"Total Candidates Screened: {len(results_df)}")
    
    drug_like = results_df.get('qed_drug_likeliness', pd.Series([0])).apply(lambda x: x >= 0.5 if pd.notna(x) else False).sum()
    c.drawString(50, y-60, f"Drug-like Candidates: {drug_like}")
    y -= 80
    
    # Methodology
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "METHODOLOGY")
    y -= 20
    c.setFont("Helvetica", 9)
    methodology = [
        "1. Compound Retrieval: Natural products database screening",
        "2. Drug-likeness Filtering: Lipinski's Rule of Five (MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10)",
        "3. ADMET Prediction: Machine learning-based toxicity and pharmacokinetic profiling",
        "4. Ranking: Multi-objective optimization (QED score, molecular weight, lipophilicity)"
    ]
    for line in methodology:
        c.drawString(60, y, line)
        y -= 15
    y -= 10
    
    # Top Candidates Table
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "TOP DRUG CANDIDATES")
    y -= 20
    
    # Table headers
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, y, "Rank")
    c.drawString(80, y, "Compound Name")
    c.drawString(250, y, "MW (Da)")
    c.drawString(320, y, "LogP")
    c.drawString(370, y, "QED")
    c.drawString(420, y, "HBD")
    c.drawString(460, y, "HBA")
    y -= 15
    
    # Table rows
    c.setFont("Helvetica", 8)
    for idx, row in results_df.head(15).iterrows():
        if y < 100:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 8)
        
        rank = idx + 1
        name = str(row.get('name', 'Unknown'))[:25]
        mw = f"{row.get('molecular_weight', 0):.1f}"
        logp = f"{row.get('alogp', 0):.2f}"
        qed = f"{row.get('qed_drug_likeliness', 0):.3f}"
        hbd = str(row.get('hydrogen_bond_donors', 0))
        hba = str(row.get('hydrogen_bond_acceptors', 0))
        
        c.drawString(50, y, str(rank))
        c.drawString(80, y, name)
        c.drawString(250, y, mw)
        c.drawString(320, y, logp)
        c.drawString(370, y, qed)
        c.drawString(420, y, hbd)
        c.drawString(460, y, hba)
        y -= 12
    
    # New page for detailed profiles
    c.showPage()
    y = height - 50
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "DETAILED COMPOUND PROFILES")
    y -= 30
    
    for idx, row in results_df.head(5).iterrows():
        if y < 150:
            c.showPage()
            y = height - 50
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Candidate #{idx+1}: {row.get('name', 'Unknown')}")
        y -= 18
        
        c.setFont("Helvetica", 9)
        c.drawString(60, y, f"SMILES: {row.get('canonical_smiles', 'N/A')[:80]}")
        y -= 15
        c.drawString(60, y, f"Molecular Formula: {row.get('molecular_formula', 'N/A')}")
        y -= 15
        c.drawString(60, y, f"Source: {row.get('organisms', 'Unknown')[:60]}")
        y -= 20
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(60, y, "Physicochemical Properties:")
        y -= 15
        c.setFont("Helvetica", 9)
        properties = [
            f"• Molecular Weight: {row.get('molecular_weight', 'N/A')} Da",
            f"• LogP (Lipophilicity): {row.get('alogp', 'N/A')}",
            f"• TPSA: {row.get('topological_polar_surface_area', 'N/A')} \u00C5\u00B2", # Å² (Angstroms squared)
            f"• Rotatable Bonds: {row.get('rotatable_bond_count', 'N/A')}",
            f"• H-Bond Donors: {row.get('hydrogen_bond_donors', 'N/A')}",
            f"• H-Bond Acceptors: {row.get('hydrogen_bond_acceptors', 'N/A')}",
            f"• QED Drug-likeness: {row.get('qed_drug_likeliness', 'N/A')}"
        ]
        for prop in properties:
            c.drawString(70, y, prop)
            y -= 12
        y -= 10
    
    # Conclusion
    c.showPage()
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "CONCLUSIONS & RECOMMENDATIONS")
    y -= 25
    c.setFont("Helvetica", 10)
    conclusions = [
        f"• {len(results_df)} compounds were successfully screened from the natural products database.",
        f"• {drug_like} candidates passed drug-likeness criteria (Lipinski's Rule of Five).",
        "• Top candidates show favorable ADMET profiles suitable for lead optimization.",
        "• Recommended next steps: In vitro bioactivity assays, cytotoxicity testing, and ADME studies.",
        "• Further structural optimization may improve pharmacological properties."
    ]
    for line in conclusions:
        c.drawString(60, y, line)
        y -= 20
    
    y -= 20
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Note: These predictions are computational estimates. Experimental validation is required.")
    
    c.save()
    buffer.seek(0)
    return buffer
    
def generate_audio_summary(results_df, query):
    # ... (Keep the exact code for generate_audio_summary) ...
    if not PDF_AUDIO_AVAILABLE:
        return None
    
    drug_like = results_df.get('qed_drug_likeliness', pd.Series([0])).apply(lambda x: x >= 0.5 if pd.notna(x) else False).sum()
    
    top_compound = results_df.iloc[0] if len(results_df) > 0 else None
    
    if top_compound is not None:
        summary = f"""
        AfroMediBot Drug Discovery Report.
        
        Query: {query}.
        
        We have successfully screened {len(results_df)} natural product compounds.
        {drug_like} candidates passed drug-likeness criteria based on Lipinski's Rule of Five.
        
        The top candidate is {top_compound.get('name', 'Unknown compound')}, 
        with a molecular weight of {top_compound.get('molecular_weight', 'unknown')} daltons,
        and a QED drug-likeness score of {top_compound.get('qed_drug_likeliness', 'unknown')}.
        
        This compound shows favorable physicochemical properties suitable for lead optimization.
        
        Please refer to the detailed PDF report for complete analysis and recommendations.
        """
    else:
        summary = f"Screening complete for {query}. {len(results_df)} compounds analyzed."
    
    try:
        tts = gTTS(text=summary, lang='en', slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except:
        return None

# ============================================================================
# STREAMLIT UI (Refactored)
# ============================================================================
def main():
    # Page config (from original code)
    st.set_page_config(
        page_title="AfroMediBot - AI Drug Discovery",
        page_icon="🌿",
        layout="wide"
    )
    
    # Custom CSS (from original code)
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header (from original code)
    st.markdown('<p class="main-header">🌿 AfroMediBot</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Powered Drug Discovery from African Medicinal Plants</p>', unsafe_allow_html=True)
    
    # --- Sidebar for Settings (API Key is handled here) ---
    groq_api_key = get_groq_api_key()
    
    # Initialize Agents and Database
    df_db = load_prefiltered_database()
    plant_agent = PlantAgent(df_db) if df_db is not None else None
    
    if df_db is None:
        st.error("Fatal Error: Could not load compound database. Ensure 'afrodb_filtered_strict.csv' is in the correct path.")
        return
    
    # --- Main Input Form & Database View Buttons ---
    st.header("🔬 Drug Candidate Screening")
    
    # 1. Button for "Prefiltered Database" view (Restored functionality)
    if st.button("📊 View Prefiltered Database Candidates", help="Shows all compounds in the database that pass basic drug-likeness filters.", key="view_db_btn"):
        st.session_state.analysis_report = None # Clear analysis on new run
        results_df, query_text = view_full_database(df_db)
        if results_df is not None:
            st.session_state.results_df = results_df
            st.session_state.query_text = query_text
            st.rerun() # Re-run to display results section
    
    # 2. Screening Form (New Form structure)
    with st.form("screening_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            plant_name = st.text_input(
                "Enter Plant Name (Botanical or Common):",
                placeholder="e.g., Vernonia amygdalina or bitter leaf",
                key="plant_input"
            )
        
        with col2:
            diseases = ['Cancer', 'Malaria', 'Diabetes', 'HIV', 'Tuberculosis', 'Inflammation']
            disease_target = st.selectbox(
                "Select Target Disease:",
                diseases,
                key="disease_select"
            )
        
        submit_button = st.form_submit_button("🚀 Run Plant Screening", type="primary")

    # --- Screening Execution ---
    if submit_button and plant_name:
        # Clear previous state on new search
        st.session_state.results_df = None 
        st.session_state.query_text = None
        st.session_state.analysis_report = None
        
        results_df, query_text = screen_plant_compounds(plant_name, disease_target, df_db, plant_agent)
        
        if results_df is not None:
            st.session_state.results_df = results_df
            st.session_state.query_text = query_text
            # Re-run to update the page with results
            st.rerun() 
    
    
    # --- Results Display and Actions ---
    if 'results_df' in st.session_state and st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        query_text = st.session_state.query_text

        st.markdown("---")
        st.subheader(f"Results for: **{query_text}**")

        # --- Tabbed Output ---
        tab_data, tab_analysis = st.tabs(["📊 Detailed Data", "🤖 Expert Analysis"])

        with tab_data:
            st.markdown("### Top Drug-like Candidates")
            
            # Display results in a clear table
            display_df = results_df[[
                'name', 'canonical_smiles', 'molecular_weight', 'alogp', 'qed_drug_likeliness', 'organisms'
            ]].head(20).copy()
            display_df.columns = ['Name', 'SMILES', 'MW (Da)', 'LogP', 'QED Score', 'Source Organism(s)']
            
            st.dataframe(display_df, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 💾 Output Generation")

            # --- KEY BUTTONS: PDF, Audio, CSV (Restored) ---
            pdf_buffer = generate_pdf_report(results_df, query_text)
            audio_buffer = generate_audio_summary(results_df, query_text)

            col_pdf, col_audio, col_csv = st.columns(3)

            with col_pdf:
                if pdf_buffer:
                    st.download_button(
                        label="📄 Generate PDF Report",
                        data=pdf_buffer,
                        file_name=f"{query_text.replace(' ', '_')}_report.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.button("📄 Generate PDF Report", disabled=True, help="Dependencies (reportlab) not found.")

            with col_audio:
                if audio_buffer:
                    st.download_button(
                        label="🎧 Generate Audio Summary",
                        data=audio_buffer,
                        file_name=f"{query_text.replace(' ', '_')}_summary.mp3",
                        mime="audio/mp3"
                    )
                else:
                    st.button("🎧 Generate Audio Summary", disabled=True, help="Dependencies (gTTS) not found.")
            
            with col_csv:
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data CSV",
                    data=csv,
                    file_name=f"{query_text.replace(' ', '_')}_data.csv",
                    mime="text/csv"
                )


        # --- Expert Analysis Tab (New) ---
        with tab_analysis:
            st.markdown("### 🧠 AI-Powered Expert Analysis")
            
            if not LLM_AVAILABLE or not groq_api_key:
                st.warning("⚠️ Groq LLM is unavailable. Please check the **API key in the sidebar** to enable analysis.")
            else:
                
                # Check if analysis is already generated and cached in session_state
                if 'analysis_report' not in st.session_state or st.session_state.analysis_report is None:
                    # Button to trigger LLM analysis
                    if st.button("🚀 Generate Expert Analysis Report", type="primary"):
                        client = GroqClient(groq_api_key)
                        with st.spinner("Contacting Groq LLM for analysis... This may take a moment."):
                            report = client.generate_expert_analysis(results_df, query_text)
                        
                        if report:
                            st.session_state.analysis_report = report
                            st.rerun()
                        else:
                            st.error("Could not generate AI analysis. Check API key status or Groq service.")
                
                # Display the generated report
                if 'analysis_report' in st.session_state and st.session_state.analysis_report:
                    st.markdown(st.session_state.analysis_report)
                    
                    # Download button for the analysis
                    analysis_data = st.session_state.analysis_report
                    st.download_button(
                        label="📄 Download Analysis as TXT",
                        data=analysis_data,
                        file_name=f"{query_text.replace(' ', '_')}_analysis.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    # Ensure session state is initialized for persistence
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'query_text' not in st.session_state:
        st.session_state.query_text = None
    if 'analysis_report' not in st.session_state:
        st.session_state.analysis_report = None

    main()
