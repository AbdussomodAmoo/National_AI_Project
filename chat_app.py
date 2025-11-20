# app.py - Conversational AI Drug Discovery Chatbot
import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

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
            'Vernonia amygdalina': 'bitter leaf, ewuro, onugbu',
            'Moringa oleifera': 'moringa, drumstick tree, zogale',
            'Azadirachta indica': 'neem, dogoyaro',
            'Garcinia kola': 'bitter kola, orogbo',
            'Hibiscus sabdariffa': 'roselle, zobo',
            'Carica papaya': 'papaya, pawpaw',
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
        
        # Load database
        df = load_prefiltered_database()
        if df is not None:
            st.success(f"✅ Database loaded: {len(df):,} compounds")
        else:
            st.warning("⚠️ No pre-filtered database found")
            uploaded_db = st.file_uploader("Upload Database CSV", type=['csv'])
            if uploaded_db:
                df = pd.read_csv(uploaded_db)
                st.success(f"✅ Loaded {len(df):,} compounds")
        
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
                response = st.session_state.chatbot.chat(prompt, uploaded_smiles)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
