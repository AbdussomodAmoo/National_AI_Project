import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint
from typing import List, Dict, Optional, Union
import joblib
import warnings
import os
import json
import requests
from typing import Dict, List, Optional
import streamlit as st
from datetime import datetime
from groq import Groq
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import os
import sys
from io import BytesIO
import base64

# Set page config FIRST
st.set_page_config(
    page_title="Aframend - AI Drug Discovery",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= IMPORT YOUR AGENTS =============
# Add your script directories to path if needed
# sys.path.append('./scripts')

# Import your existing classes (adjust paths as needed)
try:
    from agents.plant_agent import PlantAgent
    from agents.filter_agents import FilterAgent  # Note: Your file is filter_agents.py (plural)
    from agents.docking_agent import SimpleDockingAgent, setup_all_targets_rapid
    from agents.ADMETMoleculeOptimizer import MoleculeOptimizationAgent
    from agents.chatbot_agent import ChatbotAgent  # If you need this
    
    st.success("✅ Agents loaded successfully!")
    #from main_pipeline import PlantAgent, FilterAgent
    #from molecular_docking import SimpleDockingAgent
    #from optimization_agent import MoleculeOptimizationAgent
    # from bioactivity_agent import BioactivityPredictor  # If you have this
    # from admet_models import ADMETPredictor  # Your ADMET class
except ImportError:
    st.error("⚠️ Could not import agent classes. Make sure all files are in the correct location.")
    st.stop()

# ============= SESSION STATE INITIALIZATION =============
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'agents_loaded' not in st.session_state:
    st.session_state.agents_loaded = False

# ============= LOAD AGENTS (ONCE) =============
@st.cache_resource
def load_all_agents():
    """Load all agents once and cache them"""
    with st.spinner("🔧 Loading AI agents..."):
        try:
            # Load data
            df = pd.read_csv('coconut_csv/coconut_csv-06-2025.csv')
            
            # Initialize agents
            plant_agent = PlantAgent(df)
            filter_agent = FilterAgent(df)
            
            # Load docking results if available
            docking_df = None
            if os.path.exists('final_docking_results.csv'):
                docking_df = pd.read_csv('final_docking_results.csv')
            
            # Load ADMET models (adjust path to your models)
            # admet_predictor = ADMETPredictor(models_dir='./models')
            
            # Load bioactivity models
            # bio_models = load_bioactivity_models()
            
            # Initialize optimizer
            # optimizer = MoleculeOptimizationAgent(admet_predictor, bio_models, df, docking_df)
            
            return {
                'plant_agent': plant_agent,
                'filter_agent': filter_agent,
                'docking_df': docking_df,
                'data': df
                # 'optimizer': optimizer,
                # 'admet': admet_predictor
            }
        except Exception as e:
            st.error(f"Error loading agents: {e}")
            return None

# ============= HELPER FUNCTIONS =============
def display_molecule_2d(smiles):
    """Display 2D molecule structure"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(300, 300))
            return img
    except:
        pass
    return None

def create_download_link(df, filename):
    """Create download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

# ============= MAIN APP =============
def main():
    # Load agents
    agents = load_all_agents()
    if agents is None:
        st.error("Failed to load agents. Please check your data files.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=AfroMediBot", width=150)
        st.title("🌿 AfroMediBot")
        st.markdown("---")
        
        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["🔍 Plant Search", "⚗️ Drug Discovery", "🎯 Molecular Docking", "📊 Analytics"]
        )
        
        st.markdown("---")
        st.caption("Powered by African Natural Products")
    
    # Main content area
    if mode == "🔍 Plant Search":
        plant_search_interface(agents)
    
    elif mode == "⚗️ Drug Discovery":
        drug_discovery_interface(agents)
    
    elif mode == "🎯 Molecular Docking":
        docking_interface(agents)
    
    elif mode == "📊 Analytics":
        analytics_interface(agents)

# ============= PLANT SEARCH INTERFACE =============
def plant_search_interface(agents):
    st.title("🔍 Plant Compound Search")
    st.markdown("Search for compounds from African medicinal plants")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        plant_query = st.text_input(
            "Enter plant name:",
            placeholder="e.g., Vernonia amygdalina, bitter leaf, moringa",
            key="plant_search"
        )
        
        top_n = st.slider("Number of results:", 5, 50, 10)
        
        if st.button("🔍 Search", type="primary"):
            if plant_query:
                with st.spinner("Searching database..."):
                    results = agents['plant_agent'].search_by_plant(plant_query, top_n=top_n)
                    
                    if results is not None and len(results) > 0:
                        st.success(f"✅ Found {len(results)} compounds!")
                        
                        # Display results
                        st.dataframe(
                            results[['name', 'canonical_smiles', 'molecular_weight', 
                                   'alogp', 'qed_drug_likeliness']],
                            use_container_width=True
                        )
                        
                        # Download button
                        st.markdown(create_download_link(results, f"{plant_query}_compounds.csv"), 
                                  unsafe_allow_html=True)
                        
                        # Show molecule structures
                        with st.expander("🧪 View Molecular Structures"):
                            cols = st.columns(3)
                            for idx, (_, row) in enumerate(results.head(6).iterrows()):
                                with cols[idx % 3]:
                                    st.caption(row.get('name', 'Unknown'))
                                    img = display_molecule_2d(row['canonical_smiles'])
                                    if img:
                                        st.image(img)
                                    st.caption(f"MW: {row['molecular_weight']:.1f}")
                    else:
                        st.warning("No compounds found. Try a different name.")
            else:
                st.warning("Please enter a plant name")
    
    with col2:
        st.info("""
        **Tips:**
        - Use botanical names (e.g., *Moringa oleifera*)
        - Use common names (e.g., bitter leaf)
        - Search by genus only (e.g., Vernonia)
        """)

# ============= DRUG DISCOVERY INTERFACE =============
def drug_discovery_interface(agents):
    st.title("⚗️ Drug Discovery Pipeline")
    st.markdown("Filter compounds and optimize for drug-likeness")
    
    tab1, tab2 = st.tabs(["🔬 Filter Compounds", "🎯 Optimize Molecule"])
    
    with tab1:
        st.subheader("Apply Drug-Likeness Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            filter_mode = st.selectbox(
                "Filter Mode:",
                ["drug_like", "lead_like", "strict"]
            )
            
            apply_pains = st.checkbox("Apply PAINS filter", value=True)
            qed_threshold = st.slider("QED Threshold:", 0.0, 1.0, 0.5, 0.05)
        
        with col2:
            np_threshold = st.slider("NP-likeness Threshold:", -5.0, 5.0, 0.3, 0.1)
            sa_threshold = st.slider("Synthetic Accessibility:", 1, 10, 6)
        
        if st.button("🔬 Apply Filters", type="primary"):
            with st.spinner("Filtering compounds..."):
                filtered_results = agents['filter_agent'].apply_filters(
                    filter_mode=filter_mode,
                    qed_threshold=qed_threshold,
                    np_threshold=np_threshold,
                    sa_threshold=sa_threshold,
                    apply_pains=apply_pains
                )
                
                st.success(f"✅ Filtered to {len(filtered_results)} drug-like compounds")
                
                # Display results
                st.dataframe(filtered_results.head(50), use_container_width=True)
                
                # Download
                st.markdown(
                    create_download_link(filtered_results, "filtered_compounds.csv"),
                    unsafe_allow_html=True
                )
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Compounds", len(filtered_results))
                with col2:
                    avg_qed = filtered_results['qed_drug_likeliness'].mean()
                    st.metric("Avg QED", f"{avg_qed:.3f}")
                with col3:
                    lipinski_pass = filtered_results['lipinski_pass'].sum()
                    st.metric("Lipinski Pass", lipinski_pass)
    
    with tab2:
        st.subheader("Optimize Molecule")
        st.info("🚧 Coming soon: Analog generation and multi-objective optimization")

# ============= DOCKING INTERFACE =============
def docking_interface(agents):
    st.title("🎯 Molecular Docking")
    
    if agents['docking_df'] is not None:
        st.success(f"✅ Loaded {len(agents['docking_df'])} docking results")
        
        # Target selection
        targets = agents['docking_df']['target'].unique()
        selected_target = st.selectbox("Select Target:", targets)
        
        # Filter by target
        target_results = agents['docking_df'][
            agents['docking_df']['target'] == selected_target
        ].sort_values('binding_energy')
        
        # Display results
        st.subheader(f"Top Binders for {selected_target}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Binding", f"{target_results['binding_energy'].min():.2f} kcal/mol")
        with col2:
            st.metric("Total Compounds", len(target_results))
        with col3:
            success_rate = (target_results['status'] == 'Success').mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Top compounds table
        st.dataframe(
            target_results.head(20)[['compound_name', 'smiles', 'binding_energy', 'status']],
            use_container_width=True
        )
        
        # Download
        st.markdown(
            create_download_link(target_results, f"{selected_target}_docking.csv"),
            unsafe_allow_html=True
        )
        
    else:
        st.warning("⚠️ No docking results found. Upload `final_docking_results.csv`")
        
        uploaded_file = st.file_uploader("Upload Docking Results CSV", type=['csv'])
        if uploaded_file:
            docking_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(docking_df)} results")
            agents['docking_df'] = docking_df

# ============= ANALYTICS INTERFACE =============
def analytics_interface(agents):
    st.title("📊 Database Analytics")
    
    df = agents['data']
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Compounds", f"{len(df):,}")
    with col2:
        unique_organisms = df['organisms'].nunique()
        st.metric("Unique Organisms", f"{unique_organisms:,}")
    with col3:
        avg_mw = df['molecular_weight'].mean()
        st.metric("Avg Molecular Weight", f"{avg_mw:.1f}")
    with col4:
        avg_qed = df['qed_drug_likeliness'].mean()
        st.metric("Avg QED", f"{avg_qed:.3f}")
    
    # Most common organisms
    st.subheader("🌱 Most Common Source Organisms")
    organism_stats = agents['plant_agent'].list_all_organisms_tabular()
    st.dataframe(organism_stats.head(20), use_container_width=True)
    
    # Property distributions
    st.subheader("📈 Property Distributions")
    
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df['molecular_weight'].value_counts().sort_index().head(50))
        st.caption("Molecular Weight Distribution")
    
    with col2:
        st.bar_chart(df['alogp'].value_counts().sort_index().head(50))
        st.caption("LogP Distribution")

# ============= RUN APP =============
if __name__ == "__main__":
    main()
