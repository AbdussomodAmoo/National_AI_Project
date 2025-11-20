# app.py - Complete AfroMediBot in One File
import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
import base64
from collections import Counter
import requests
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
    from rdkit.Chem import BRICS, FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    RDKIT_AVAILABLE = True
except ImportError:
    st.error("⚠️ RDKit not installed. Install with: pip install rdkit-pypi")
    RDKIT_AVAILABLE = False

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AfroMediBot - AI Drug Discovery",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PLANT AGENT CLASS
# ============================================================================
class PlantAgent:
    def __init__(self, df):
        self.df = df
        self.common_name_map = self._load_hardcoded_common_name_mapping()

    def _load_hardcoded_common_name_mapping(self):
        known_mappings = {
            'Vernonia amygdalina': 'bitter leaf, ewuro, onugbu, grawa',
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
        lower_plant_name = plant_name.lower()
        if lower_plant_name in self.common_name_map:
            return self.common_name_map[lower_plant_name]
        return plant_name

    def search_by_plant(self, plant_name, top_n=10):
        resolved_name = self.resolve_plant_name(plant_name)
        results = self.df[self.df['organisms'].str.contains(resolved_name, case=False, na=False)]
        
        if results.empty and resolved_name.lower() != plant_name.lower():
            results = self.df[self.df['organisms'].str.contains(plant_name, case=False, na=False)]
        
        return results.head(top_n) if not results.empty else None

    def list_all_organisms_tabular(self):
        organism_series = self.df['organisms'].dropna()
        all_organisms = []
        
        for entry in organism_series:
            parts = [o.strip() for o in entry.split('|') if o.strip()]
            all_organisms.extend(parts)
        
        organism_counts = Counter(all_organisms)
        return pd.DataFrame(
            organism_counts.items(), 
            columns=['Organism', 'Count']
        ).sort_values(by='Count', ascending=False)

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
# DOCKING AGENT CLASS (Simplified)
# ============================================================================
class SimpleDockingAgent:
    def __init__(self):
        self.results = []
    
    def load_docking_results(self, csv_path):
        """Load pre-computed docking results"""
        if os.path.exists(csv_path):
            self.results = pd.read_csv(csv_path)
            return True
        return False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_data
def load_database(file_path):
    """Load the main compound database"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

def display_molecule_2d(smiles):
    """Display 2D molecule structure"""
    if not RDKIT_AVAILABLE:
        return None
    try:
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
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

# ============================================================================
# SESSION STATE
# ============================================================================
if 'agents_loaded' not in st.session_state:
    st.session_state.agents_loaded = False
if 'plant_agent' not in st.session_state:
    st.session_state.plant_agent = None
if 'filter_agent' not in st.session_state:
    st.session_state.filter_agent = None

# ============================================================================
# LOAD AGENTS
# ============================================================================
@st.cache_resource
def initialize_agents(df):
    """Initialize all agents"""
    plant_agent = PlantAgent(df)
    filter_agent = FilterAgent(df)
    docking_agent = SimpleDockingAgent()
    
    # Try to load docking results
    docking_paths = ['docking_results.csv', 'data/docking_results.csv', 
                     'final_docking_results.csv', 'afrodb_docking_results.csv']
    
    docking_loaded = False
    for path in docking_paths:
        if docking_agent.load_docking_results(path):
            docking_loaded = True
            break
    
    return {
        'plant_agent': plant_agent,
        'filter_agent': filter_agent,
        'docking_agent': docking_agent,
        'docking_loaded': docking_loaded
    }

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.title("🌿 AfroMediBot")
        st.markdown("### AI Drug Discovery Platform")
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload COCONUT Database CSV",
            type=['csv'],
            help="Upload the coconut_csv-06-2025.csv file"
        )
        
        mode = st.radio(
            "Select Mode:",
            ["🔍 Plant Search", "⚗️ Drug Filtering", "🎯 Docking Results", "📊 Analytics"]
        )
        
        st.markdown("---")
        st.caption("© 2024 AfroMediBot")
    
    # Load database
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} compounds")
    else:
        # Try to find database in common locations
        db_paths = [
            'coconut_csv-06-2025.csv',
            'data/coconut_csv-06-2025.csv',
            'coconut_csv/coconut_csv-06-2025.csv'
        ]
        
        df = None
        for path in db_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Found database: {len(df):,} compounds")
                break
        
        if df is None:
            st.warning("⚠️ Please upload the COCONUT database CSV file")
            st.info("Expected file: `coconut_csv-06-2025.csv`")
            st.stop()
    
    # Initialize agents
    agents = initialize_agents(df)
    
    # Route to selected mode
    if mode == "🔍 Plant Search":
        plant_search_interface(agents, df)
    elif mode == "⚗️ Drug Filtering":
        drug_filtering_interface(agents, df)
    elif mode == "🎯 Docking Results":
        docking_interface(agents)
    elif mode == "📊 Analytics":
        analytics_interface(agents, df)

# ============================================================================
# PLANT SEARCH INTERFACE
# ============================================================================
def plant_search_interface(agents, df):
    st.title("🔍 Plant Compound Search")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        plant_query = st.text_input(
            "Enter plant name:",
            placeholder="e.g., Vernonia amygdalina, bitter leaf, moringa"
        )
        
        top_n = st.slider("Number of results:", 5, 50, 10)
        
        if st.button("🔍 Search", type="primary"):
            if plant_query:
                with st.spinner("Searching..."):
                    results = agents['plant_agent'].search_by_plant(plant_query, top_n)
                    
                    if results is not None:
                        st.success(f"✅ Found {len(results)} compounds!")
                        
                        # Display key columns
                        display_cols = ['name', 'canonical_smiles', 'molecular_weight', 
                                      'alogp', 'qed_drug_likeliness']
                        available_cols = [col for col in display_cols if col in results.columns]
                        
                        st.dataframe(results[available_cols], use_container_width=True)
                        
                        # Download
                        st.markdown(
                            create_download_link(results, f"{plant_query}_compounds.csv"),
                            unsafe_allow_html=True
                        )
                        
                        # Visualize molecules
                        if RDKIT_AVAILABLE and 'canonical_smiles' in results.columns:
                            with st.expander("🧪 View Structures"):
                                cols = st.columns(3)
                                for idx, (_, row) in enumerate(results.head(6).iterrows()):
                                    with cols[idx % 3]:
                                        img = display_molecule_2d(row['canonical_smiles'])
                                        if img:
                                            st.image(img, use_column_width=True)
                                        st.caption(f"MW: {row.get('molecular_weight', 'N/A')}")
                    else:
                        st.warning("No results found. Try a different name.")
    
    with col2:
        st.info("""
        **Search Tips:**
        - Botanical names: *Moringa oleifera*
        - Common names: bitter leaf
        - Genus only: Vernonia
        """)

# ============================================================================
# DRUG FILTERING INTERFACE
# ============================================================================
def drug_filtering_interface(agents, df):
    st.title("⚗️ Drug-Likeness Filtering")
    
    st.markdown("Apply filters to identify drug-like compounds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        filter_mode = st.selectbox("Filter Mode:", ["drug_like", "lead_like", "strict"])
        apply_pains = st.checkbox("Apply PAINS filter", value=True)
    
    with col2:
        qed_threshold = st.slider("QED Threshold:", 0.0, 1.0, 0.5, 0.05)
        np_threshold = st.slider("NP-likeness:", -5.0, 5.0, 0.3, 0.1)
    
    if st.button("🔬 Apply Filters", type="primary"):
        with st.spinner("Filtering compounds..."):
            filtered_df = agents['filter_agent'].apply_filters(
                filter_mode=filter_mode,
                qed_threshold=qed_threshold,
                np_threshold=np_threshold,
                apply_pains=apply_pains
            )
            
            st.success(f"✅ Filtered to {len(filtered_df):,} compounds")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", f"{len(filtered_df):,}")
            with col2:
                if 'lipinski_pass' in filtered_df.columns:
                    st.metric("Lipinski Pass", f"{filtered_df['lipinski_pass'].sum():,}")
            with col3:
                if 'qed_drug_likeliness' in filtered_df.columns:
                    avg_qed = filtered_df['qed_drug_likeliness'].mean()
                    st.metric("Avg QED", f"{avg_qed:.3f}")
            
            # Display results
            st.dataframe(filtered_df.head(50), use_container_width=True)
            
            # Download
            st.markdown(
                create_download_link(filtered_df, "filtered_compounds.csv"),
                unsafe_allow_html=True
            )

# ============================================================================
# DOCKING INTERFACE
# ============================================================================
def docking_interface(agents):
    st.title("🎯 Molecular Docking Results")
    
    if agents['docking_loaded']:
        docking_df = agents['docking_agent'].results
        st.success(f"✅ Loaded {len(docking_df):,} docking results")
        
        # Target selection
        if 'target' in docking_df.columns:
            targets = docking_df['target'].unique()
            selected_target = st.selectbox("Select Target:", targets)
            
            target_results = docking_df[docking_df['target'] == selected_target]
            target_results = target_results.sort_values('binding_energy')
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                best_energy = target_results['binding_energy'].min()
                st.metric("Best Binding", f"{best_energy:.2f} kcal/mol")
            with col2:
                st.metric("Total Compounds", len(target_results))
            with col3:
                if 'status' in target_results.columns:
                    success = (target_results['status'] == 'Success').sum()
                    st.metric("Successful", success)
            
            # Results table
            st.dataframe(target_results.head(20), use_container_width=True)
            
            # Download
            st.markdown(
                create_download_link(target_results, f"{selected_target}_results.csv"),
                unsafe_allow_html=True
            )
        else:
            st.dataframe(docking_df.head(50), use_container_width=True)
    else:
        st.warning("⚠️ No docking results found")
        st.info("Upload a docking results CSV file")
        
        uploaded = st.file_uploader("Upload Docking Results", type=['csv'])
        if uploaded:
            docking_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(docking_df):,} results")
            agents['docking_agent'].results = docking_df
            st.rerun()

# ============================================================================
# ANALYTICS INTERFACE
# ============================================================================
def analytics_interface(agents, df):
    st.title("📊 Database Analytics")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Compounds", f"{len(df):,}")
    with col2:
        if 'organisms' in df.columns:
            unique_org = df['organisms'].nunique()
            st.metric("Unique Organisms", f"{unique_org:,}")
    with col3:
        if 'molecular_weight' in df.columns:
            avg_mw = df['molecular_weight'].mean()
            st.metric("Avg MW", f"{avg_mw:.1f}")
    with col4:
        if 'qed_drug_likeliness' in df.columns:
            avg_qed = df['qed_drug_likeliness'].mean()
            st.metric("Avg QED", f"{avg_qed:.3f}")
    
    # Top organisms
    st.subheader("🌱 Most Common Source Organisms")
    organism_stats = agents['plant_agent'].list_all_organisms_tabular()
    st.dataframe(organism_stats.head(20), use_container_width=True)
    
    # Property distributions
    if 'molecular_weight' in df.columns:
        st.subheader("📈 Property Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(df['molecular_weight'].value_counts().sort_index().head(50))
            st.caption("Molecular Weight")
        
        with col2:
            if 'alogp' in df.columns:
                st.bar_chart(df['alogp'].value_counts().sort_index().head(50))
                st.caption("LogP")

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
