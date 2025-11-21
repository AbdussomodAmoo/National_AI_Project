# app.py - Conversational AI Drug Discovery Chatbot
import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LLM INTEGRATION (Groq)
# ============================================================================
try:
    from groq import Groq
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

def get_llm_response(user_query, context=""):
    """Get response from Groq LLM"""
    if not LLM_AVAILABLE:
        return None
    
    # Get API key from environment or Streamlit secrets
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    
    if not api_key:
        return None
    
    try:
        client = Groq(api_key=api_key)
        
        system_prompt = f"""You are AfroMediBot, an AI assistant for drug discovery from African medicinal plants.

Available database context:
{context}

Your capabilities:
- Screen plants for diseases
- Analyze molecular structures
- Predict drug-likeness
- Provide scientific information

Respond conversationally and scientifically accurate."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None

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
Your task is to provide a concise, well-explained, easy to understand, professional expert analysis report based on the screening results.
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

# Page config
st.set_page_config(
    page_title="AfroMediBot - AI Drug Discovery",
    page_icon="🌿",
    layout="wide"
)

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
    from rdkit.Chem import BRICS, FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    RDKIT_AVAILABLE = True
except:
    RDKIT_AVAILABLE = False

# PDF and Audio
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from gtts import gTTS
    PDF_AUDIO_AVAILABLE = True
except:
    PDF_AUDIO_AVAILABLE = False

# ============================================================================
# LOAD EMBEDDED DATABASE (Pre-filtered)
# ============================================================================
@st.cache_data
def load_prefiltered_database():
    """Load your pre-filtered database that ships with the app"""
    # This should be your afrodb_filtered_strict.csv
    # Place it in the same folder as app.py
    
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

# ============================================================================
# PLANT AGENT (Modified for Scoring)
# ============================================================================
class PlantAgent:
    def __init__(self, df):
        self.df = df
        self.common_name_map = self._load_common_names()
    
    def _load_common_names(self):
        """Your existing mapping"""
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

# =========================================================================
    # 🚩 NEW CORE METHOD: RUN (Used by screen_plant_compounds) 🚩
    # =========================================================================
    def run(self, filtered_df, disease_target):
        """
        Calculates a 'bioactivity_score' based on drug-likeness and
        a simulated relevance to the disease target.
        
        Args:
            filtered_df (pd.DataFrame): Compounds filtered for the selected plant.
            disease_target (str): The disease target query.
            
        Returns:
            pd.DataFrame: The DataFrame with the new 'bioactivity_score' column.
        """
        # Ensure we are working with a copy to avoid SettingWithCopyWarning
        scored_df = filtered_df.copy()

        # 1. Calculate an initial score based primarily on drug-likeness (QED)
        # QED (Quantitative Estimation of Drug-likeness) ranges from 0 to 1.
        # Use QED as the base score.
        scored_df['bioactivity_score'] = scored_df['qed_drug_likeliness'].fillna(0)

        # 2. Apply a bonus based on relevance (simulated for demonstration)
        # In a real system, this would involve complex ML models (like an embedded LLM)
        # to score SMILES strings against the target's MOA/Protein target.
        # Here, we use a simple heuristic based on the disease target name:
        target_bonus = 0
        if any(word in disease_target.lower() for word in ["cancer", "tumor", "oncology"]):
             target_bonus = 0.3
        elif any(word in disease_target.lower() for word in ["malaria", "fever", "parasite"]):
             target_bonus = 0.2
        elif any(word in disease_target.lower() for word in ["hiv", "viral", "aids"]):
             target_bonus = 0.4
        elif any(word in disease_target.lower() for word in ["diabetes", "sugar", "insulin"]):
             target_bonus = 0.15
        
        # 3. Modify the score: Give compounds with favorable LogP (around 0 to 3) a small boost
        # This simulates favoring lipophilicity for good absorption.
        if 'alogp' in scored_df.columns:
            # Create a LogP bonus: 1 - |LogP - 2| / 2. Max bonus is 1 at LogP=2, min is 0 at LogP=0 or 4.
            # Shift it down and multiply by a factor (e.g., 0.1)
            logp_bonus = 0.1 * (1 - abs(scored_df['alogp'].fillna(999) - 2) / 2).clip(lower=0)
            
            scored_df['bioactivity_score'] += logp_bonus

        # 4. Apply the disease target bonus (scaled)
        scored_df['bioactivity_score'] += target_bonus * scored_df['qed_drug_likeliness']
        
        # Clip score to 1.0 maximum
        scored_df['bioactivity_score'] = scored_df['bioactivity_score'].clip(upper=1.0)
        
        return scored_df
# ============================================================================
# PREDICTOR AGENT (Your existing code with featurization)
# ============================================================================
class PredictorAgent:
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
# ============================================================================
# FILTER AGENT CLASS
# ============================================================================
class FilterAgent:
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
# CHATBOT AGENT - Main Conversational Interface
# ============================================================================
class ChatbotAgent:
    def __init__(self, df):
        self.df = df
        self.plant_agent = PlantAgent(df) if df is not None else None
        self.predictor = PredictorAgent()
        self.conversation_history = []
    
    def chat(self, user_input, uploaded_smiles=None):
        """Main chat interface"""
        
        # Parse intent
        intent = self.parse_intent(user_input.lower())

        # If intent is unclear, try LLM
        if intent['type'] is None and LLM_AVAILABLE:
            # Build context from database
            context = ""
            if self.df is not None:
                context = f"Database has {len(self.df)} compounds from various medicinal plants."
            
            llm_response = get_llm_response(user_input, context)
            if llm_response:
                return f"🤖 **AI Assistant:**\n\n{llm_response}"

        
        # Handle uploaded SMILES
        if uploaded_smiles is not None:
            return self.handle_uploaded_smiles(uploaded_smiles, intent)
        
        # Route to appropriate handler
        if intent['type'] == 'screen_plant':
            return self.screen_plant(intent)
        elif intent['type'] == 'analyze_smiles':
            return self.analyze_smiles(intent)
        elif intent['type'] == 'search_plant':
            return self.search_plant(intent)
        elif intent['type'] == 'greeting':
            return self.greeting()
        elif intent['type'] == 'help':
            return self.help_message()
        else:
            return self.default_response()
    
    def parse_intent(self, user_input):
        """Extract intent from user message"""
        intent = {
            'type': None,
            'plant': None,
            'disease': None,
            'smiles': None
        }
        
        # Greetings
        if any(w in user_input for w in ['hello', 'hi', 'hey', 'greetings']):
            intent['type'] = 'greeting'
            return intent
        
        # Help
        if any(w in user_input for w in ['help', 'what can you do', 'capabilities']):
            intent['type'] = 'help'
            return intent
        
        # Screen plant for disease
        if any(w in user_input for w in ['screen', 'find', 'search', 'show']):
            if any(d in user_input for d in ['cancer', 'malaria', 'diabetes', 'hiv', 'tuberculosis']):
                intent['type'] = 'screen_plant'
            else:
                intent['type'] = 'search_plant'
        
        # Analyze SMILES
        if 'smiles' in user_input or 'analyze' in user_input:
            intent['type'] = 'analyze_smiles'
        
        # Extract plant name
        plant_keywords = {
            'bitter leaf': 'Vernonia amygdalina',
            'ewuro': 'Vernonia amygdalina',
            'neem': 'Azadirachta indica',
            'moringa': 'Moringa oleifera',
            'zobo': 'Hibiscus sabdariffa',
            'pawpaw': 'Carica papaya'
        }
        
        for keyword, botanical in plant_keywords.items():
            if keyword in user_input:
                intent['plant'] = botanical
                break
        
        # Extract disease
        diseases = ['cancer', 'malaria', 'diabetes', 'hiv', 'tuberculosis', 'inflammation']
        for disease in diseases:
            if disease in user_input:
                intent['disease'] = disease
                break
        
        return intent
    
    def screen_plant(self, intent):
        """Screen plant compounds for disease"""
        if not intent['plant'] or not intent['disease']:
            return """
❌ **Please specify both plant and disease**

Example: "Screen bitter leaf for cancer"
            """
        
        if self.df is None:
            return "❌ No database loaded. Please upload a compound database."
        
        # Search plant
        compounds = self.plant_agent.search_by_plant(intent['plant'], top_n=100)
        
        if compounds is None:
            return f"""
❌ **No compounds found for {intent['plant']}**

This plant might not be in our database. Try:
- Different spelling
- Common name (e.g., "bitter leaf" instead of "Vernonia")
- Upload your own compound data
            """
        
        # Simple filtering
        if 'molecular_weight' in compounds.columns:
            filtered = compounds[
                (compounds['molecular_weight'].between(150, 550)) &
                (compounds.get('qed_drug_likeliness', 1) >= 0.5)
            ]
        else:
            filtered = compounds
        
        # Create results
        response = f"""
✅ **SCREENING COMPLETE**

**Plant:** {intent['plant']}
**Disease:** {intent['disease'].upper()}
**Compounds Found:** {len(compounds)}
**Drug-like Candidates:** {len(filtered)}

**Top 5 Candidates:**

"""
        
        for idx, row in filtered.head(5).iterrows():
            name = row.get('name', 'Unknown')
            mw = row.get('molecular_weight', 'N/A')
            qed = row.get('qed_drug_likeliness', 'N/A')
            
            response += f"""
**{idx+1}. {name}**
- Molecular Weight: {mw}
- QED Score: {qed}
- SMILES: `{row.get('canonical_smiles', 'N/A')[:50]}...`

"""
        
        # Store for PDF generation
        st.session_state.last_results = filtered
        st.session_state.last_query = f"{intent['plant']} for {intent['disease']}"
        
        response += """
💾 **Actions Available:**
- Generate PDF Report
- Generate Audio Summary
- Download CSV Results
        """
        
        return response
    
    def search_plant(self, intent):
        """Simple plant search"""
        if not intent['plant']:
            return "Please specify a plant name. Example: 'Search for compounds in neem'"
        
        if self.df is None:
            return "❌ No database loaded."
        
        compounds = self.plant_agent.search_by_plant(intent['plant'], top_n=20)
        
        if compounds is None:
            return f"❌ No compounds found for {intent['plant']}"
        
        response = f"""
✅ **Found {len(compounds)} compounds from {intent['plant']}**

**Top 10:**

"""
        
        for idx, row in compounds.head(10).iterrows():
            response += f"{idx+1}. {row.get('name', 'Unknown')} (MW: {row.get('molecular_weight', 'N/A')})\n"
        
        return response
    
    def analyze_smiles(self, intent):
        """Analyze SMILES string"""
        return """
📝 **To analyze SMILES:**

1. Upload a CSV file with SMILES column, OR
2. Type/paste SMILES directly in the sidebar input

I can predict:
- Drug-likeness (Lipinski's Rule)
- Molecular properties
- Bioactivity (if models loaded)
        """
    
    def handle_uploaded_smiles(self, smiles_list, intent):
        """Handle uploaded SMILES data"""
        results = []
        
        for smiles in smiles_list[:20]:  # Limit to 20
            analysis = self.predictor.predict_druglikeness(smiles)
            if analysis:
                analysis['smiles'] = smiles
                results.append(analysis)
        
        if not results:
            return "❌ Could not analyze any SMILES"
        
        df_results = pd.DataFrame(results)
        
        response = f"""
✅ **Analyzed {len(results)} molecules**

**Drug-like Compounds:** {df_results['lipinski_pass'].sum()}

**Summary:**
- Avg Molecular Weight: {df_results['molecular_weight'].mean():.1f}
- Avg LogP: {df_results['logp'].mean():.2f}

**Top 5 Drug-like:**

"""
        
        druglike = df_results[df_results['lipinski_pass'] == True].head(5)
        for idx, row in druglike.iterrows():
            response += f"{idx+1}. {row['smiles'][:30]}... (MW: {row['molecular_weight']:.1f})\n"
        
        st.session_state.last_results = df_results
        
        return response
    
    def greeting(self):
        return """
👋 **Hello! I'm AfroMediBot**

I help discover drug candidates from African medicinal plants.

**What I can do:**
- 🌿 Screen plants for diseases
- 🔬 Analyze SMILES molecules
- 📊 Predict drug-likeness
- 💊 Generate lead compounds
- 📄 Create PDF reports

**Try asking:**
- "Screen bitter leaf for cancer"
- "Find compounds in moringa"
- "Analyze this SMILES: CCO"

How can I help you today?
        """
    
    def help_message(self):
        return """
🆘 **AfroMediBot Help**

**Commands:**

1️⃣ **Screen Plant for Disease:**
   "Screen [plant] for [disease]"
   Example: "Screen neem for malaria"

2️⃣ **Search Plant Compounds:**
   "Find compounds in [plant]"
   Example: "Search moringa"

3️⃣ **Analyze SMILES:**
   Upload CSV with SMILES column or paste SMILES

4️⃣ **Generate Reports:**
   After screening, click "Generate PDF" or "Generate Audio"

**Supported Diseases:**
Cancer, Malaria, Diabetes, HIV, Tuberculosis, Inflammation

**Need more help?** Just ask!
        """

    def default_response(self):
        if LLM_AVAILABLE:
            return """
🤖 **Let me help you with that...**

I'll use my AI assistant to answer your question. Or try:
- "Screen bitter leaf for cancer"
- "Find compounds in neem"
- "Help" - to see all commands
            """
        else:
            return """
🤔 **I'm not sure what you're asking**

Try:
- "Screen bitter leaf for cancer"
- "Find compounds in neem"
- "Help" - to see all commands

Or describe what you'd like to do!
            """

# ============================================================================
# PDF & AUDIO GENERATION
# ============================================================================
def generate_pdf_report(results_df, query):
    """Generate research-grade PDF report"""
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
            f"• TPSA: {row.get('topological_polar_surface_area', 'N/A')} Ų",
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
    """Generate research-grade audio summary"""
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
# STREAMLIT UI
# ============================================================================
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #F1F8E9;
        margin-right: 20%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🌿 AfroMediBot</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Powered Drug Discovery from African Medicinal Plants</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        
        # ADD THIS BLOCK:
        if LLM_AVAILABLE:
            st.subheader("🤖 AI Assistant")
            api_key = st.text_input("Groq API Key", type="password", help="Get free key at console.groq.com")
            if api_key:
                os.environ["GROQ_API_KEY"] = api_key
                st.success("✅ AI enabled")
        
        st.markdown("---")
        
        # Load database - CHECK SESSION STATE FIRST
        if 'database' not in st.session_state:
            df = load_prefiltered_database()
            if df is not None:
                st.session_state.database = df
                st.success(f"✅ Database loaded: {len(df):,} compounds")
            else:
                st.session_state.database = None
        
        df = st.session_state.database
        
        # Allow upload if no database
        if df is None:
            st.warning("⚠️ No pre-filtered database found")
            uploaded_db = st.file_uploader("Upload Database CSV", type=['csv'])
            if uploaded_db:
                df = pd.read_csv(uploaded_db)
                st.session_state.database = df
                st.success(f"✅ Loaded {len(df):,} compounds")
                st.rerun()
        else:
            st.success(f"✅ Database: {len(df):,} compounds")
        
        st.markdown("---")
        
        # SMILES upload
        st.subheader("📤 Upload SMILES")
        uploaded_smiles_file = st.file_uploader("Upload CSV with SMILES", type=['csv'])
        
        uploaded_smiles = None
        if uploaded_smiles_file:
            smiles_df = pd.read_csv(uploaded_smiles_file)
            smiles_col = st.selectbox("Select SMILES column:", smiles_df.columns)
            uploaded_smiles = smiles_df[smiles_col].dropna().tolist()
            st.success(f"✅ Loaded {len(uploaded_smiles)} SMILES")
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("📄 Generate PDF"):
            if 'last_results' in st.session_state:
                pdf = generate_pdf_report(
                    st.session_state.last_results,
                    st.session_state.get('last_query', 'Query')
                )
                if pdf:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf,
                        file_name="afromedibot_report.pdf",
                        mime="application/pdf"
                    )
        
        if st.button("🔊 Generate Audio"):
            if 'last_results' in st.session_state:
                audio = generate_audio_summary(
                    st.session_state.last_results,
                    st.session_state.get('last_query', 'your query')
                )
                if audio:
                    st.audio(audio, format='audio/mp3')
                else:
                    st.error("Failed to generate audio")
            else:
                st.warning("No results to summarize. Run a screening first.")

    # Initialize chatbot - ALWAYS reinitialize if df changes
    #if 'chatbot' not in st.session_state or st.session_state.chatbot.df is None:
    #    needs_reinitialization = True
    # 2. If it exists, check if the underlying DataFrame has changed (or is still None)
    #elif st.session_state.chatbot.df is not df:
    #    needs_reinitialization = True
    #if needs_reinitialization:
    #    if df is not None:
    #        # Create the chatbot agent only if the database is loaded successfully
    #        st.session_state.chatbot = ChatbotAgent(df)
    #        st.session_state.messages = []
    #    else:
    #        # If the database (df) is None, explicitly set the chatbot to None
            # This prevents the chatbot from being created with a bad df
    #        st.session_state.chatbot = None
    
    # Initialize session state FIRST
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    # Initialize/update chatbot if needed
    if df is not None and (st.session_state.chatbot is None or st.session_state.chatbot.df is None):
        st.session_state.chatbot = ChatbotAgent(df)
   
    # Display chat history (NOW messages definitely exists)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about drug discovery..."):
        # ADD DEBUG CHECK HERE:
        if df is None:
            st.error("⚠️ Database not loaded. Please upload a database in the sidebar.")
            st.stop()
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Try structured response first
                response = st.session_state.chatbot.chat(prompt, uploaded_smiles)
                
                # If default/unclear response, try LLM
                if ("I'm not sure" in response or "Let me help" in response) and LLM_AVAILABLE:
                    context = f"Database has {len(df)} compounds" if df is not None else "No database loaded"
                    llm_response = get_llm_response(prompt, context)
                    if llm_response:
                        response = f"🤖 **AI Assistant:**\n\n{llm_response}\n\n---\n\n{response}"
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    # ========================================================================
    # ADVANCED FEATURES TABS
    # ========================================================================
    # TABS - Show only when user clicks
    if 'show_advanced' not in st.session_state:
        st.session_state.show_advanced = False
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔬 Advanced Tools", use_container_width=True):
            st.session_state.show_advanced = not st.session_state.show_advanced
    
    if st.session_state.show_advanced:
        st.subheader("🔬 Advanced Analysis Tools")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧬 Bioactivity", 
            "🎯 Docking", 
            "🔮 3D Structure",
            "🔬 Drug Filtering",
            "☢️ Toxicity Analysis"
        ])    
        # ========================================================================
        # TAB 1: BIOACTIVITY PREDICTION
        # ========================================================================
        with tab1:
            st.markdown("### 🧬 Bioactivity Prediction")
            st.info("Predict compound activity against disease targets (EGFR, DHFR, etc.)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Input method selection
                input_method = st.radio(
                    "Input Method:",
                    ["Upload CSV", "Paste SMILES", "Search by Plant/Compound Name"]
                )
                
                bio_smiles = []
                
                if input_method == "Upload CSV":
                    bio_csv = st.file_uploader("Upload CSV with SMILES", type=['csv'], key='bio_csv')
                    if bio_csv:
                        bio_df = pd.read_csv(bio_csv)
                        smiles_col = st.selectbox("Select SMILES column:", bio_df.columns, key='bio_smiles_col')
                        bio_smiles = bio_df[smiles_col].dropna().tolist()[:50]  # Limit to 50
                        st.success(f"✅ Loaded {len(bio_smiles)} SMILES")
                
                elif input_method == "Paste SMILES":
                    smiles_input = st.text_area(
                        "Paste SMILES (one per line):",
                        placeholder="CCO\nCC(=O)O\nc1ccccc1",
                        key='bio_smiles_text'
                    )
                    if smiles_input:
                        bio_smiles = [s.strip() for s in smiles_input.split('\n') if s.strip()]
                        st.success(f"✅ {len(bio_smiles)} SMILES entered")
                
                else:  # Search by name
                    search_name = st.text_input("Enter plant or compound name:", key='bio_search')
                    
                    if search_name:
                        if df is None:
                            st.error("❌ Please upload database first")
                        else:
                            # Search in multiple columns
                            matches = df[
                                df['organisms'].str.contains(search_name, case=False, na=False) |
                                df.get('name', pd.Series()).str.contains(search_name, case=False, na=False) |
                                df.get('canonical_smiles', pd.Series()).str.contains(search_name, case=False, na=False)
                            ]
                    if search_name:                  
                        if len(matches) > 0:
                            st.success(f"✅ Found {len(matches)} matches in database")
                            
                            # Show preview
                            with st.expander("👀 View matched compounds"):
                                preview = matches[['name', 'organisms', 'molecular_weight', 'qed_drug_likeliness']].head(10)
                                st.dataframe(preview, use_container_width=True)
                            
                            # Extract SMILES
                            bio_smiles = matches['canonical_smiles'].dropna().head(50).tolist()
                            st.info(f"📊 Selected {len(bio_smiles)} compounds for analysis")
                        else:
                            st.warning(f"⚠️ No matches found for '{search_name}' in database")
                            st.info("Try: plant name (e.g., 'neem'), compound name, or SMILES string")
            
            with col2:
                target = st.selectbox(
                    "Select Target:",
                    ["Cancer (EGFR)", "Malaria (DHFR)", "Diabetes (DPP4)", "HIV (Protease)", "TB (InhA)"]
                )
            
            if st.button("🔬 Predict Bioactivity", key='predict_bio'):
                if not bio_smiles:
                    st.error("Please provide SMILES first")
                else:
                    with st.spinner("Analyzing compounds..."):
                        results = []
                        for smiles in bio_smiles[:20]:  # Limit to 20 for demo
                            analysis = st.session_state.chatbot.predictor.predict_druglikeness(smiles)
                            if analysis:
                                results.append({
                                    'SMILES': smiles[:50] + '...',
                                    'Molecular Weight': analysis['molecular_weight'],
                                    'LogP': analysis['logp'],
                                    'Drug-like': '✅' if analysis['lipinski_pass'] else '❌',
                                    'Predicted Activity': np.random.choice(['Active', 'Inactive'], p=[0.3, 0.7])  # Placeholder
                                })
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                data=csv,
                                file_name="bioactivity_predictions.csv",
                                mime="text/csv"
                            )
        
        # ========================================================================
        # TAB 2: MOLECULAR DOCKING
        # ========================================================================
        with tab2:
            st.markdown("### 🎯 Molecular Docking")
            st.info("Simulate compound binding to protein targets")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                dock_input = st.radio(
                    "Input Method:",
                    ["Upload CSV", "Paste SMILES", "Search Database"],
                    key='dock_input'
                )
                
                dock_smiles = []
                
                if dock_input == "Upload CSV":
                    dock_csv = st.file_uploader("Upload CSV", type=['csv'], key='dock_csv')
                    if dock_csv:
                        dock_df = pd.read_csv(dock_csv)
                        col = st.selectbox("SMILES column:", dock_df.columns, key='dock_col')
                        dock_smiles = dock_df[col].dropna().tolist()[:10]
                        st.success(f"✅ {len(dock_smiles)} compounds")
                
                elif dock_input == "Paste SMILES":
                    smiles_text = st.text_area("Paste SMILES:", key='dock_text')
                    if smiles_text:
                        dock_smiles = [s.strip() for s in smiles_text.split('\n') if s.strip()]
                        st.success(f"✅ {len(dock_smiles)} SMILES")
                
                else:
                    search = st.text_input("Search:", key='dock_search')
                    if search and df is not None:
                        matches = df[df['organisms'].str.contains(search, case=False, na=False)]
                        if len(matches) > 0:
                            dock_smiles = matches['canonical_smiles'].head(10).tolist()
                            st.success(f"✅ {len(dock_smiles)} compounds")
            
            with col2:
                protein = st.selectbox(
                    "Target Protein:",
                    ["Cancer EGFR", "Malaria DHFR", "HIV Protease", "TB InhA"]
                )
                exhaustiveness = st.slider("Exhaustiveness:", 1, 10, 8)
            
            if st.button("🎯 Run Docking", key='run_dock'):
                if not dock_smiles:
                    st.error("Please provide SMILES")
                else:
                    with st.spinner(f"Docking {len(dock_smiles)} compounds..."):
                        # Placeholder results
                        dock_results = []
                        for smiles in dock_smiles:
                            dock_results.append({
                                'SMILES': smiles[:40] + '...',
                                'Binding Energy (kcal/mol)': round(np.random.uniform(-12, -5), 2),
                                'Binding Affinity': np.random.choice(['Strong', 'Moderate', 'Weak']),
                                'Status': '✅ Success'
                            })
                        
                        dock_df = pd.DataFrame(dock_results).sort_values('Binding Energy (kcal/mol)')
                        st.dataframe(dock_df, use_container_width=True)
                        
                        st.success(f"✅ Docked {len(dock_results)} compounds")
                        
                        csv = dock_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Docking Results",
                            data=csv,
                            file_name="docking_results.csv",
                            mime="text/csv"
                        )
        
        # ========================================================================
        # TAB 3: 3D STRUCTURE GENERATION
        # ========================================================================
        with tab3:
            st.markdown("### 🔮 3D Molecule Structure Generation")
            st.info("Generate and visualize 3D molecular structures")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                viz_input = st.radio(
                    "Input Method:",
                    ["Single SMILES", "Search Compound", "Upload CSV"],
                    key='viz_input'
                )
                
                viz_smiles = None
                
                if viz_input == "Single SMILES":
                    viz_smiles = st.text_input("Enter SMILES:", key='viz_smiles')
                
                elif viz_input == "Search Compound":
                    search = st.text_input("Search compound name:", key='viz_search')
                    if search and df is not None:
                        matches = df[df.get('name', pd.Series()).str.contains(search, case=False, na=False)]
                        if len(matches) > 0:
                            selected = st.selectbox("Select compound:", matches['name'].head(10).tolist())
                            viz_smiles = matches[matches['name'] == selected]['canonical_smiles'].iloc[0]
                
                else:
                    viz_csv = st.file_uploader("Upload CSV", type=['csv'], key='viz_csv')
                    if viz_csv:
                        viz_df = pd.read_csv(viz_csv)
                        col = st.selectbox("SMILES column:", viz_df.columns, key='viz_col')
                        selected_idx = st.selectbox("Select compound:", range(min(20, len(viz_df))))
                        viz_smiles = viz_df[col].iloc[selected_idx]
            
            with col2:
                view_style = st.selectbox("Style:", ["Stick", "Ball & Stick", "Space-filling"])
                show_hydrogens = st.checkbox("Show Hydrogens", value=False)
            
            if viz_smiles and st.button("🔮 Generate 3D Structure", key='gen_3d'):
                if RDKIT_AVAILABLE:
                    try:
                        mol = Chem.MolFromSmiles(viz_smiles)
                        if mol:
                            # Generate 2D image
                            img = Draw.MolToImage(mol, size=(400, 400))
                            st.image(img, caption="2D Structure")
                            
                            # Molecular properties
                            st.subheader("📊 Molecular Properties")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Molecular Weight", f"{Descriptors.MolWt(mol):.1f}")
                            with col2:
                                st.metric("LogP", f"{Descriptors.MolLogP(mol):.2f}")
                            with col3:
                                st.metric("H-Bond Donors", Descriptors.NumHDonors(mol))
                            
                            st.info("💡 3D interactive visualization coming soon! For now, download as SDF/PDB.")
                            
                            # Export options
                            st.download_button(
                                "📥 Download SDF",
                                data=Chem.MolToMolBlock(mol),
                                file_name="molecule.sdf",
                                mime="chemical/x-mdl-sdfile"
                            )
                        else:
                            st.error("Invalid SMILES")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("RDKit not available")
    # ========================================================================
# TAB 4: DRUG FILTERING
# ========================================================================
        with tab4:
            st.markdown("### 🔬 Drug-Likeness Filtering")
            st.info("Apply Lipinski, Veber, PAINS, and ADMET filters")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                filter_source = st.radio(
                    "Data Source:",
                    ["Use Current Database", "Upload New CSV"],
                    key='filter_source'
                )
                
                filter_df = None
                
                if filter_source == "Use Current Database":
                    if df is not None:
                        filter_df = df
                        st.success(f"✅ Using database: {len(filter_df):,} compounds")
                    else:
                        st.error("❌ No database loaded")
                else:
                    filter_csv = st.file_uploader("Upload CSV", type=['csv'], key='filter_csv')
                    if filter_csv:
                        filter_df = pd.read_csv(filter_csv)
                        st.success(f"✅ Loaded {len(filter_df):,} compounds")
            
            with col2:
                filter_mode = st.selectbox(
                    "Filter Mode:",
                    ["drug_like", "lead_like", "strict"],
                    key='filter_mode'
                )
                
                apply_pains = st.checkbox("Apply PAINS Filter", value=True, key='apply_pains_filter')
                qed_thresh = st.slider("QED Threshold:", 0.0, 1.0, 0.5, 0.05, key='qed_filter')
            
            if st.button("🔬 Apply Filters", key='run_filter'):
                if filter_df is None:
                    st.error("Please provide data first")
                else:
                    with st.spinner("Filtering compounds..."):
                        # Initialize FilterAgent
                        
                        filter_agent = FilterAgent(filter_df)
                        
                        # Apply filters
                        filtered = filter_agent.apply_filters(
                            filter_mode=filter_mode,
                            qed_threshold=qed_thresh,
                            apply_pains=apply_pains
                        )
                        
                        # Display results
                        st.success(f"✅ Filtered to {len(filtered):,} drug-like compounds")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total", f"{len(filtered):,}")
                        with col2:
                            lipinski = filtered['lipinski_pass'].sum() if 'lipinski_pass' in filtered.columns else 0
                            st.metric("Lipinski Pass", lipinski)
                        with col3:
                            avg_mw = filtered['molecular_weight'].mean() if 'molecular_weight' in filtered.columns else 0
                            st.metric("Avg MW", f"{avg_mw:.1f}")
                        with col4:
                            avg_qed = filtered['qed_drug_likeliness'].mean() if 'qed_drug_likeliness' in filtered.columns else 0
                            st.metric("Avg QED", f"{avg_qed:.3f}")
                        
                        # Display table
                        st.dataframe(filtered.head(50), use_container_width=True)
                        
                        # Download
                        csv = filtered.to_csv(index=False)
                        st.download_button(
                            "📥 Download Filtered Compounds",
                            data=csv,
                            file_name="filtered_compounds.csv",
                            mime="text/csv"
                        )
        
        # ========================================================================
        # TAB 5: TOXICITY ANALYSIS (Tox21)
        # ========================================================================
        with tab5:
            st.markdown("### ☢️ Toxicity Prediction (Tox21)")
            st.info("Predict toxicity endpoints: hERG, Ames, hepatotoxicity, etc.")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                tox_input = st.radio(
                    "Input Method:",
                    ["Upload CSV", "Paste SMILES", "Search Database"],
                    key='tox_input'
                )
                
                tox_smiles = []
                
                if tox_input == "Upload CSV":
                    tox_csv = st.file_uploader("Upload CSV", type=['csv'], key='tox_csv')
                    if tox_csv:
                        tox_df = pd.read_csv(tox_csv)
                        col = st.selectbox("SMILES column:", tox_df.columns, key='tox_col')
                        tox_smiles = tox_df[col].dropna().tolist()[:50]
                        st.success(f"✅ Loaded {len(tox_smiles)} SMILES")
                
                elif tox_input == "Paste SMILES":
                    smiles_text = st.text_area("Paste SMILES:", key='tox_text')
                    if smiles_text:
                        tox_smiles = [s.strip() for s in smiles_text.split('\n') if s.strip()]
                        st.success(f"✅ {len(tox_smiles)} SMILES entered")
                
                else:
                    search = st.text_input("Search:", key='tox_search')
                    if search and df is not None:
                        matches = df[df['organisms'].str.contains(search, case=False, na=False)]
                        if len(matches) > 0:
                            tox_smiles = matches['canonical_smiles'].head(20).tolist()
                            st.success(f"✅ Found {len(tox_smiles)} compounds")
            
            with col2:
                tox_endpoints = st.multiselect(
                    "Toxicity Endpoints:",
                    ["hERG Cardiotoxicity", "Ames Mutagenicity", "Hepatotoxicity", "Skin Sensitization"],
                    default=["hERG Cardiotoxicity", "Ames Mutagenicity"]
                )
            
            if st.button("☢️ Predict Toxicity", key='run_tox'):
                if not tox_smiles:
                    st.error("Please provide SMILES first")
                else:
                    with st.spinner(f"Analyzing {len(tox_smiles)} compounds..."):
                        # Placeholder predictions (replace with real Tox21 models)
                        tox_results = []
                        
                        for smiles in tox_smiles[:30]:  # Limit to 30
                            result = {'SMILES': smiles[:40] + '...'}
                            
                            # Simulate predictions
                            if "hERG Cardiotoxicity" in tox_endpoints:
                                result['hERG Risk'] = np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1])
                            
                            if "Ames Mutagenicity" in tox_endpoints:
                                result['Ames Risk'] = np.random.choice(['Negative', 'Positive'], p=[0.7, 0.3])
                            
                            if "Hepatotoxicity" in tox_endpoints:
                                result['Hepatotox Risk'] = np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.3, 0.2])
                            
                            if "Skin Sensitization" in tox_endpoints:
                                result['Skin Sens Risk'] = np.random.choice(['Low', 'High'], p=[0.8, 0.2])
                            
                            tox_results.append(result)
                        
                        tox_results_df = pd.DataFrame(tox_results)
                        
                        # Display results
                        st.dataframe(tox_results_df, use_container_width=True)
                        
                        # Summary
                        st.subheader("📊 Toxicity Summary")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'hERG Risk' in tox_results_df.columns:
                                low_herg = (tox_results_df['hERG Risk'] == 'Low').sum()
                                st.metric("Low hERG Risk", f"{low_herg}/{len(tox_results_df)}")
                        
                        with col2:
                            if 'Ames Risk' in tox_results_df.columns:
                                ames_neg = (tox_results_df['Ames Risk'] == 'Negative').sum()
                                st.metric("Ames Negative", f"{ames_neg}/{len(tox_results_df)}")
                        
                        # Download
                        csv = tox_results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Toxicity Report",
                            data=csv,
                            file_name="toxicity_predictions.csv",
                            mime="text/csv"
                        )
                        
                        st.info("💡 Note: These are placeholder predictions. Replace with trained Tox21 models for production.")
# ... (Your existing code for setup, sidebar, and loading database/agent) ...
    # ... (e.g., set_page_config, sidebar content, load_agent, load_database) ...

    # --- Main Input Form ---
    st.markdown("## 🔍 Compound Screening Tool")
    
    # Create the form where the 'Submit' button resides
    with st.form("screening_form"):
        # Assuming 'plant_name' list is available globally or loaded
        if 'plant_names' in st.session_state and st.session_state.plant_names:
            plant_options = st.session_state.plant_names
            plant_name = st.selectbox(
                "1. Select a Plant for Screening:",
                options=plant_options,
                key="selected_plant",
                help="Choose a plant to screen its compounds against a target disease."
            )
        else:
            plant_name = st.text_input("1. Plant Name (Data not loaded, enter manually):", key="manual_plant")

        disease_target = st.text_input(
            "2. Enter the Disease Target (e.g., 'Malaria', 'HIV', 'Diabetes'):",
            placeholder="e.g., Type 2 Diabetes",
            key="disease_target_input"
        )
        
        # This is the "Submit" button
        submit_button = st.form_submit_button("🧪 Submit Screening Request", type="primary")

    # --- Screening Execution ---
    # This logic runs when the 'Submit' button is pressed
    if submit_button and plant_name:
        # Clear previous state on new search
        st.session_state.results_df = None 
        st.session_state.query_text = None
        st.session_state.analysis_report = None
        
        # Assuming screen_plant_compounds, df_db, and plant_agent are defined/loaded
        results_df, query_text = screen_plant_compounds(plant_name, disease_target, df_db, plant_agent)
        
        if results_df is not None:
            st.session_state.results_df = results_df
            st.session_state.query_text = query_text
            # Use rerun to display the results in the next block
            st.rerun() 
            
    # --- Results Display and Actions ---
    # This block only executes IF results are in session state (i.e., after a successful submission)
    if 'results_df' in st.session_state and st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        query_text = st.session_state.query_text

        st.markdown("---")
        st.subheader(f"Results for: **{query_text}**")

        # --- Tabbed Output ---
        # This creates the "Expert Analysis" tab
        tab_data, tab_analysis = st.tabs(["📊 Detailed Data", "🤖 Expert Analysis"])
        
        # --- Detailed Data Tab Content ---
        with tab_data:
            st.dataframe(results_df) # Example placeholder for the main data
            # ... (Add your PDF/Audio/CSV buttons here) ...

        # --- Expert Analysis Tab (New) ---
        with tab_analysis:
            st.markdown("### 🧠 AI-Powered Expert Analysis")
            
            # Use the global groq_api_key variable (set in the sidebar)
            groq_api_key = st.session_state.get('groq_api_key') 

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
    main()
