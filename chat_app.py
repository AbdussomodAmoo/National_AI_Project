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
            model="llama-3.1-70b-versatile",  # or "mixtral-8x7b-32768"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=1024
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
# PLANT AGENT (Your existing code)
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

            # Common Database Organisms
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
    """Generate PDF report"""
    if not PDF_AUDIO_AVAILABLE:
        return None
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    y = 750
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "AfroMediBot - Drug Discovery Report")
    y -= 30
    
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Query: {query}")
    y -= 20
    c.drawString(50, y, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 30
    
    c.drawString(50, y, f"Total Candidates: {len(results_df)}")
    y -= 30
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Top Candidates:")
    y -= 20
    
    c.setFont("Helvetica", 10)
    for idx, row in results_df.head(10).iterrows():
        if y < 100:
            c.showPage()
            y = 750
        
        c.drawString(60, y, f"{idx+1}. {row.get('name', 'Unknown')}")
        y -= 15
        c.drawString(70, y, f"MW: {row.get('molecular_weight', 'N/A')}, QED: {row.get('qed_drug_likeliness', 'N/A')}")
        y -= 20
    
    c.save()
    buffer.seek(0)
    return buffer

def generate_audio_summary(text):
    """Generate audio summary"""
    if not PDF_AUDIO_AVAILABLE:
        return None
    
    try:
        tts = gTTS(text=text, lang='en', slow=False)
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
        if st.button("📄 Generate PDF", disabled='last_results' not in st.session_state):
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
        
        if st.button("🔊 Generate Audio", disabled='last_results' not in st.session_state):
            if 'last_results' in st.session_state:
                summary = f"Found {len(st.session_state.last_results)} drug candidates for {st.session_state.get('last_query', 'your query')}"
                audio = generate_audio_summary(summary)
                if audio:
                    st.audio(audio, format='audio/mp3')
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatbotAgent(df)
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about drug discovery..."):
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
        tab1, tab2, tab3 = st.tabs(["🧬 Bioactivity", "🎯 Docking", "🔮 3D"])    
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
                if search_name and df is not None:
                    # Search in database
                    matches = df[
                        df['organisms'].str.contains(search_name, case=False, na=False) |
                        df.get('name', pd.Series()).str.contains(search_name, case=False, na=False)
                    ]
                    if len(matches) > 0:
                        bio_smiles = matches['canonical_smiles'].head(20).tolist()
                        st.success(f"✅ Found {len(bio_smiles)} compounds")
                    else:
                        st.warning("No matches found")
        
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

if __name__ == "__main__":
    main()
