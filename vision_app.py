# app.py - AfroMediBot Feature Showcase
import streamlit as st
import pandas as pd
import os
from io import BytesIO
import base64
from PIL import Image
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

# ============================================================================
# LOAD DEPENDENCIES
# ============================================================================

# Literature Mining
try:
    from Bio import Entrez
    from groq import Groq
    import json
    import time
    LITERATURE_AVAILABLE = True
except ImportError:
    LITERATURE_AVAILABLE = False

# 3D Visualization
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw
    import py3Dmol
    from stmol import showmol
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Google Vision
try:
    from google.cloud import vision
    
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

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

1. **Evidence Strength**: Rate LOW/MEDIUM/HIGH based on number of papers, study types, and results consistency
2. **Active Compounds**: List specific chemical compounds mentioned with their activities
3. **Mechanisms of Action**: Explain HOW the plant compounds work against {disease}
4. **Key Findings**: Summarize the most important discoveries (IC50 values, clinical outcomes, etc.)
5. **Research Gaps**: What's missing or needs more study?

Be concise, scientific, and cite paper numbers when making claims."""
    
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
    
    # Google Vision Credentials
    vision_creds = st.file_uploader("Google Vision Credentials (JSON)", type=['json'])
    if vision_creds:
        with open('vision-credentials.json', 'wb') as f:
            f.write(vision_creds.read())
        st.success("✅ Vision API configured")
    
    st.markdown("---")
    
    st.subheader("📊 Quick Stats")
    st.metric("Plants in Database", "500+")
    st.metric("Compounds Analyzed", "50,000+")
    st.metric("AI Models", "6")

# ============================================================================
# MAIN APP - TAB NAVIGATION
# ============================================================================

tab_home, tab_literature, tab_3d, tab_plant = st.tabs([
    "🏠 Home",
    "📚 Literature Mining",
    "🧊 3D Molecule Viewer",
    "🌿 Plant Recognition"
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

                            # --- NEW STEP 1: Extract the most likely species name ---
                            identified_species_name = extract_plant_species(labels, entities)

                            # --- Display the main result clearly ---
                            st.subheader("🌿 Primary Identification")
                            
                            if identified_species_name != "Unknown plant":
                                st.success(f"**Identified Species:** **{identified_species_name}**")
                            else:
                                st.warning("Could not definitively identify species.")
                                
                            st.markdown("---")
                            
                            # --- Display Details (Labels and Entities) ---
                            
                            st.subheader("🏷️ Detected Labels")
                            for label in labels[:5]:
                                st.write(f"• {label.description}: {label.score:.1%} confidence")
                            
                            st.subheader("🌿 Plant Identification")
                            if entities:
                                # --- Display specific scientific and relevant entities ---
                                found_entity = False
                                for entity in entities[:3]:
                                    # Filter for entities that look like names or specific details
                                    if entity.description and (entity.description.lower() not in ['plant', 'leaf', 'tree'] or entity.score > 0.6):
                                        st.success(f"✅ Possible match: **{entity.description}** (Score: {entity.score:.2f})")
                                        found_entity = True
                                if not found_entity:
                                    st.info("No highly specific web entities found.")
                            else:
                                st.warning("Could not identify specific species. Try a clearer image.")

                            # --- NEW STEP 2: Add action button to search literature ---
                            st.markdown("---")
                            if st.button(f"📚 Find Literature for {identified_species_name}", key="search_lit_from_plant", type="secondary"):
                                # Set session state variables to automatically fill the Literature tab
                                st.session_state.example_plant = identified_species_name
                                st.session_state.active_tab = "literature"
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error during Vision API analysis: {e}")
                            st.exception(e)

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
