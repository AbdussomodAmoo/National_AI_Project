# app.py - AfroMediBot Feature Showcase
import streamlit as st
import pandas as pd
import tabulate
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
import joblib
import numpy as np
from rdkit.Chem import rdMolDescriptors, AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
import tempfile
import subprocess


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
# SESSION STATE INITIALIZATION (Add new variables here)
# ============================================================================

if 'active_compounds' not in st.session_state:
    # This will hold the compounds (DataFrame) selected for analysis/docking
    st.session_state['active_compounds'] = pd.DataFrame() 
if 'analysis_report' not in st.session_state:
    # This holds the LLM's generated report for the Bioactivity tab
    st.session_state['analysis_report'] = "" 
if 'groq_api_key_input' not in st.session_state:
    # Ensures the API key input is tracked
    st.session_state['groq_api_key_input'] = ''
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
# ----------------------------------------------------------------------------
# STANDALONE PREDICTION FUNCTION (Extracts logic from old Chatbot.predictor)
# ----------------------------------------------------------------------------

def predict_druglikeness_properties(smiles):
    """
    Predicts drug-likeness properties for a single SMILES string.
    NOTE: This is the logic previously housed in the PredictorAgent class.
    """
    if not RDKIT_AVAILABLE:
        return {'lipinski_pass': False, 'molecular_weight': 0, 'logp': 0, 'hbd': 0, 'hba': 0}
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        # Lipinski's Rule of Five check (Max 1 violation) 
        lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
        
        return {
            'lipinski_pass': lipinski,
            'molecular_weight': mw,
            'logp': logp,
            'hbd': hbd,
            'hba': hba
        }
    except Exception as e:
        print(f"Prediction Error for {smiles}: {e}")
        return None

# ============================================================================
# BIOACTIVITY PREDICTION FUNCTIONS
# ============================================================================
BIOACTIVITY_TARGETS = {
    'Cancer (EGFR)': 'egfr',
    'Malaria (DHFR)': 'dhfr',
    'Diabetes (DPP4)': 'dpp4',
    'HIV (Protease)': 'hiv_protease',
    'TB (InhA)': 'tb_inha'
}


# Global variable to cache loaded models
@st.cache_resource
def load_bioactivity_models():
    """Load all trained bioactivity models"""
    models = {}
    missing_models = []
    model_dir = "models/bioactivity"

    # ✅ CHECK IF DIRECTORY EXISTS
    if not os.path.exists(model_dir):
        st.error(f"❌ Directory not found: {model_dir}")
        st.info(f"💡 Current working directory: {os.getcwd()}")
        st.info(f"💡 Files in current directory: {os.listdir('.')}")
        return models
        
    model_files = {
        
        'Cancer (EGFR)': 'cancer_EGFR',
        #'Malaria (DHFR)': 'dhfr',
        'Diabetes (DPP4)': 'diabetes_DPP4',
        'HIV (Protease)': 'hiv_HIV_Protease',
        #'TB (InhA)': 'tb_inha'
    }
    
    for display_name, file_prefix in model_files.items():
        # Try both classification and regression
        class_path = f"{model_dir}/{file_prefix}_classification_model.joblib"
        reg_path = f"{model_dir}/{file_prefix}_regression.joblib"

        loaded = False
        error_msg = None # Stores error message

        # Try classification
        if os.path.exists(class_path):
            try:
                st.write(f"🔄 Loading {class_path}...")  # ✅ DEBUG
                models[display_name] = {
                    'model': joblib.load(class_path),
                    'type': 'classification'
                }
                loaded = True
                st.write(f"✅ Loaded {display_name} (classification)")  # ✅ DEBUG
            except Exception as e:
                error_msg = str(e) # Save error
                st.error(f"❌ Failed to load {class_path}: {e}")

        # Try regression if classification failed
        if not loaded and os.path.exists(reg_path):
            try:
                st.write(f"🔄 Loading {reg_path}...")  # ✅ DEBUG
                models[display_name] = {
                    'model': joblib.load(reg_path),
                    'type': 'regression'
                }
                loaded = True
                st.write(f"✅ Loaded {display_name} (regression)")  # ✅ DEBUG
            except Exception as e:
                error_msg = str(e)  # ✅ Save error
                st.warning(f"❌ Failed to load {reg_path}: {e}")
                           
        if not loaded:
            missing_models.append(display_name)
            st.warning(f"⚠️ No model found for {display_name}")
            

    # Display status
    if models:
        st.success(f"✅ Loaded {len(models)} bioactivity models: {', '.join(models.keys())}")
    
    if missing_models:
        st.warning(f"⚠️ Missing models: {', '.join(missing_models)}")
        st.info("💡 The app will work with available models only.")
    
    return models

def featurize(smiles):
    """
    Extract comprehensive molecular features from SMILES.
    Must match your training featurization exactly.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    feature_dict = {
        # --- 2D Descriptors ---
        'MolWt': Descriptors.MolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'RingCount': Descriptors.RingCount(mol),
        'FractionCsp3': Descriptors.FractionCSP3(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
    }

    # --- 3D descriptors ---
    try:
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDG()
        params.randomSeed = 42

        if AllChem.EmbedMolecule(mol_3d, params) == 0:
            AllChem.UFFOptimizeMolecule(mol_3d)

            feature_dict.update({
                'Asphericity': rdMolDescriptors.CalcAsphericity(mol_3d),
                'Eccentricity': rdMolDescriptors.CalcEccentricity(mol_3d),
                'InertialShapeFactor': rdMolDescriptors.CalcInertialShapeFactor(mol_3d),
                'RadiusOfGyration': rdMolDescriptors.CalcRadiusOfGyration(mol_3d),
                'SpherocityIndex': rdMolDescriptors.CalcSpherocityIndex(mol_3d),
                'PMI1': rdMolDescriptors.CalcPMI1(mol_3d),
                'PMI2': rdMolDescriptors.CalcPMI2(mol_3d),
                'PMI3': rdMolDescriptors.CalcPMI3(mol_3d),
            })
        else:
            raise Exception("3D embedding failed")
    except:
        # Fill with 0 if 3D fails
        feature_dict.update({
            'Asphericity': 0, 'Eccentricity': 0, 'InertialShapeFactor': 0,
            'RadiusOfGyration': 0, 'SpherocityIndex': 0,
            'PMI1': 0, 'PMI2': 0, 'PMI3': 0,
        })

    # --- Morgan Fingerprints ---
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    fp_array = np.array(fp)

    for j, bit in enumerate(fp_array):
        feature_dict[f'fp_{j}'] = bit

    return feature_dict

def predict_bioactivity(smiles, target_name, models_dict):
    """Predict bioactivity using trained models"""
    
    if target_name not in models_dict:
        return None
    
    # Featurize
    features = featurize(smiles)
    if features is None:
        return None
    
    # Convert to DataFrame
    X = pd.DataFrame([features]).fillna(0)
    
    # Get model
    model_info = models_dict[target_name]
    model = model_info['model']
    model_type = model_info['type']
    
    # Predict
    if model_type == 'classification':
        pred_class = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        
        return {
            'prediction': 'Active' if pred_class == 1 else 'Inactive',
            'confidence': max(pred_proba),
            'activity_probability': pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
        }
    else:  # regression
        pred_log_ic50 = model.predict(X)[0]
        pred_ic50 = 10 ** pred_log_ic50
        
        # Classify based on IC50 threshold
        activity = 'Active' if pred_ic50 < 10 else 'Inactive'
        
        return {
            'prediction': activity,
            'ic50_um': pred_ic50,
            'confidence': 1.0 if pred_ic50 < 10 else 0.5
        }

# ============================================================================
# MOLECULAR DOCKING
# ============================================================================
# Global dictionary to store configured docker instances
#all_dockers = {}
def perform_docking_for_target(smiles, target_name, debug=False):
    """
    Executes a molecular docking simulation for a single compound against a specific target.

    Parameters:
    -----------
    smiles : str
        The SMILES string of the compound to dock.
    target_name : str
        The name of the target protein (e.g., 'cancer_EGFR').
    debug : bool
        If True, prints detailed debug messages during the workflow.

    Returns:
    --------
    dict: {'target': str, 'smiles': str, 'binding_energy': float or None, 'status': str}
    """
    if target_name not in st.session_state.all_dockers:
        return {
            'target': target_name,
            'smiles': smiles,
            'binding_energy': None,
            'status': f"Error: Target '{target_name}' not found/prepared."
        }

    docker = st.session_state.all_dockers[target_name]

    if debug:
        st.write(f"DEBUG: Starting docking for SMILES: {smiles} on target: {target_name}")

    # Execute the docking workflow
    energy = docker.dock_compound(smiles)

    if energy is not None:
        status = 'Success'
        if debug:
            st.write(f"DEBUG: Docking successful. Energy: {energy:.2f} kcal/mol.")
    else:
        status = 'Failed'
        if debug:
            st.write(f"DEBUG: Docking failed.")

    return {
        'target': target_name,
        'smiles': smiles,
        'binding_energy': energy,
        'status': status
    }
    
def initialize_docking_agents():
    """
    Initialize all docking agents from .pdbqt files
    """
    

    protein_dir = 'protein_structure' # Protein's folder
    
    if 'all_dockers' not in st.session_state:
        st.session_state.all_dockers = {} # <- Ensure initialization if called directly
        
    #if not os.path.exists(protein_dir):
    #    st.error(f"❌ Protein directory not found: {protein_dir}")
    #    return 0
    
    loaded_count = 0
    
    for target_key, config in DOCKING_TARGETS.items():
        pdbqt_path = f"{protein_dir}/{target_key}.pdbqt"
        
        if os.path.exists(pdbqt_path):
            try:
                # Store directly into the session state dictionary
                st.session_state.all_dockers[target_key] = SimpleDockingAgent(pdbqt_path, config['binding_site'])
                st.success(f"✅ Loaded docking target: {target_key}")
                loaded_count += 1
            except Exception as e:
                st.warning(f"⚠️ Failed to load {target_key}: {e}")
        else:
            st.warning(f"⚠️ PDBQT file not found: {pdbqt_path}")
    
    return loaded_count

# Initialize your SimpleDockingAgent here
class SimpleDockingAgent:
    """
    Minimal AutoDock Vina wrapper with robust error reporting.
    """

    def __init__(self, protein_pdbqt, binding_site):
        """
        Parameters:
        -----------
        protein_pdbqt : str
            Path to prepared protein PDBQT file
        binding_site : dict
            {'center': (x, y, z), 'size': (x, y, z)}
        """
        self.protein_pdbqt = protein_pdbqt
        self.center = binding_site['center']
        self.size = binding_site['size']

    def smiles_to_3d(self, smiles):
        """
        Convert SMILES to 3D molecule
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            params = AllChem.ETKDG()
            params.randomSeed = 42

            if AllChem.EmbedMolecule(mol, params) != 0:
                return None

            # Optimize geometry
            AllChem.UFFOptimizeMolecule(mol)

            return mol

        except Exception as e:
            print(f"3D generation failed: {e}")
            return None

    def mol_to_pdbqt(self, mol):
        """
        Convert RDKit molecule to PDBQT string using Meeko
        """
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        # Note: error_msg is not used here but is returned by Meeko
        pdbqt_string, is_ok, _ = PDBQTWriterLegacy.write_string(mol_setups[0])

        return pdbqt_string if is_ok else None

    def run_vina(self, ligand_pdbqt_content):
        """
        Run AutoDock Vina docking with built-in error reporting.
        (FIXED: Removed unsupported '--log' and parse stdout instead)
        """
        ligand_file, output_file = None, None

        try:
            # 1. Write ligand PDBQT to temp file
            with tempfile.NamedTemporaryFile(suffix='_ligand.pdbqt', delete=False, mode='w') as f:
                f.write(ligand_pdbqt_content)
                ligand_file = f.name

            # 2. Define output files
            output_file = ligand_file.replace('_ligand.pdbqt', '_out.pdbqt')

            # 3. Build Vina command (Removed '--log' and log_file)
            cmd = [
                'vina',
                '--receptor', self.protein_pdbqt,
                '--ligand', ligand_file,
                '--out', output_file,
                '--center_x', str(self.center[0]),
                '--center_y', str(self.center[1]),
                '--center_z', str(self.center[2]),
                '--size_x', str(self.size[0]),
                '--size_y', str(self.size[1]),
                '--size_z', str(self.size[2]),
                '--exhaustiveness', '8',
                '--num_modes', '1',
                '--energy_range', '3'
            ]

            # 4. Run Vina
            # Capture stdout to get the scores (since --log is not supported)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # --- CRITICAL ERROR CHECK ---
            if result.returncode != 0:
                print(f"\n--- VINA EXECUTION FAILED! Return Code: {result.returncode} ---")
                print("Command: " + " ".join(cmd))
                print("\n--- VINA STDERR (Error Output) ---")
                print(result.stderr)
                return None

            # 5. Parse binding energy from STDOUT
            binding_energy = None

            # Vina prints the result table to stdout. We look for the first line starting with '1'.
            output_lines = result.stdout.splitlines()

            for line in output_lines:
                # The line format is: 1 | -8.5 | 0.000 / 0.000
                if line.strip().startswith('1'):
                    # Use a split on whitespace to extract columns
                    parts = line.split()

                    # Check if the first part is exactly '1' and there's a second part (the score)
                    if len(parts) >= 2 and parts[0] == '1':
                        try:
                            # The binding energy is the second column
                            binding_energy = float(parts[1])
                            break
                        except ValueError:
                            # Handle cases where the second part might not be a float
                            continue

            return binding_energy

        except subprocess.TimeoutExpired:
            print("Vina timeout (>2 min)")
            return None
        except Exception as e:
            print(f"Vina execution failed: {e}")
            return None
        finally:
            # 6. Cleanup (only ligand and output pdbqt files)
            for f in [ligand_file, output_file]:
                if f and os.path.exists(f):
                    os.remove(f)

    def dock_compound(self, smiles):
        """
        Complete docking workflow for a SMILES string
        """
        try:
            # Step 1: Convert SMILES to 3D
            mol = self.smiles_to_3d(smiles)
            if mol is None:
                return None

            # Step 2: Convert to PDBQT format
            pdbqt_content = self.mol_to_pdbqt(mol)
            if pdbqt_content is None:
                return None

            # Step 3: Run Vina docking
            binding_energy = self.run_vina(pdbqt_content)

            return binding_energy

        except Exception as e:
            print(f"Docking failed for {smiles[:20]}...: {e}")
            return None

    def dock_batch(self, smiles_list, compound_names=None):
        """
        Dock multiple compounds
        """
        results = []

        for idx, smiles in enumerate(smiles_list):
            name = compound_names[idx] if compound_names else f"Compound_{idx+1}"

            print(f"Docking {name} ({idx+1}/{len(smiles_list)})...", end=' ')

            energy = self.dock_compound(smiles)

            if energy is not None:
                print(f"✅ {energy:.2f} kcal/mol")
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'binding_energy': energy,
                    'status': 'Success'
                })
            else:
                print(f"❌ Failed")
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'binding_energy': None,
                    'status': 'Failed'
                })

        return pd.DataFrame(results)

DOCKING_TARGETS = {
    # CANCER TARGETS
    'cancer_EGFR': {
        'pdb_id': '1M17',
        'binding_site': {'center': (25.0, 12.5, 40.3), 'size': (20, 20, 20)},
        'models': ['cancer_EGFR_regression', 'cancer_EGFR_classification']
    },
    'cancer_BCR_ABL': {
        'pdb_id': '2HYY',
        'binding_site': {'center': (15.3, 22.1, 18.7), 'size': (20, 20, 20)},
        'models': ['cancer_BCR_ABL_regression', 'cancer_BCR_ABL_classification']
    },
    'cancer_HER2': {
        'pdb_id': '3PP0',
        'binding_site': {'center': (18.5, 14.2, 32.1), 'size': (20, 20, 20)},
        'models': ['cancer_HER2_regression', 'cancer_HER2_classification']
    },

    # HIV TARGETS
    'hiv_RT': {
        'pdb_id': '1RTD',
        'binding_site': {'center': (15.2, 18.7, 22.1), 'size': (20, 20, 20)},
        'models': ['hiv_HIV_RT_regression', 'hiv_HIV_RT_classification']
    },
    'hiv_Protease': {
        'pdb_id': '1HXB',
        'binding_site': {'center': (0.5, 1.2, -2.3), 'size': (20, 20, 20)},
        'models': ['hiv_HIV_Protease_regression', 'hiv_HIV_Protease_classification']
    },
    'hiv_Integrase': {
        'pdb_id': '1QS4',
        'binding_site': {'center': (12.3, 8.5, 15.2), 'size': (20, 20, 20)},
        'models': ['hiv_HIV_Integrase_regression', 'hiv_HIV_Integrase_classification']
    },

    # DIABETES TARGETS
    'diabetes_DPP4': {
        'pdb_id': '1X70',
        'binding_site': {'center': (25.1, 15.3, 10.8), 'size': (20, 20, 20)},
        'models': ['diabetes_DPP4_regression', 'diabetes_DPP4_classification']
    },

    # HYPERTENSION
    'hypertension_ACE': {
        'pdb_id': '1O86',
        'binding_site': {'center': (30.2, 28.5, 42.1), 'size': (20, 20, 20)},
        'models': ['hypertension_ACE_regression', 'hypertension_ACE_classification']
    },

    # INFLAMMATION
    'inflammation_COX2': {
        'pdb_id': '5KIR',
        'binding_site': {'center': (28.5, 22.3, 15.8), 'size': (20, 20, 20)},
        'models': ['inflammation_COX2_regression', 'inflammation_COX2_classification']
    },
    'cancer_CDK': {
        'pdb_id': '1HCK',
        'binding_site': {'center': (15.5, 22.3, 18.8), 'size': (20, 20, 20)},
        'models': ['cancer_CDK_regression', 'cancer_CDK_classification']
    },
    'tuberculosis': {
        'pdb_id': '4TZK',
        'binding_site': {'center': (12.8, 18.5, 22.3), 'size': (20, 20, 20)},
        'models': ['tuberculosis_classification']
    },
    'inflammation_LOX5': {
        'pdb_id': '3O8Y',
        'binding_site': {'center': (20.5, 15.3, 25.1), 'size': (20, 20, 20)},
        'models': ['inflammation_LOX5_regression', 'inflammation_LOX5_classification']
    },
    'tuberculosis_InhA': {
        'pdb_id': '4TZK',
        'binding_site': {'center': (12.8, 18.5, 22.3), 'size': (20, 20, 20)},
        'models': ['tuberculosis_InhA_regression', 'tuberculosis_classification']
    },
    'cancer_VEGFR2': {
        'pdb_id': '3WZE',
        'binding_site': {'center': (30.2, 18.5, 22.7), 'size': (20, 20, 20)},
        'models': ['cancer_VEGFR2_regression', 'cancer_VEGFR2_classification']
    },
    'cancer_Topo_II': {
        'pdb_id': '3QX3',
        'binding_site': {'center': (25.8, 32.1, 15.5), 'size': (20, 20, 20)},
        'models': ['cancer_Topo_II_classification']
    },
    'diabetes_Alpha_Glucosidase': {
        'pdb_id': '3L4Y',
        'binding_site': {'center': (22.5, 28.3, 35.2), 'size': (20, 20, 20)},
        'models': ['diabetes_Alpha_Glucosidase_regression', 'diabetes_Alpha_Glucosidase_classification']
    },
    'diabetes_PPAR_gamma': {
        'pdb_id': '2PRG',
        'binding_site': {'center': (18.3, 12.7, 20.8), 'size': (20, 20, 20)},
        'models': ['diabetes_PPAR_gamma_classification']
    }
     
}



    

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
# ADMET CONSTANTS
# ============================================================================

ADMET_MODEL_DIR = "models/admet"

# Define the models and their associated files
ADMET_MODEL_CONFIG = {
    # Key: Display Name | Value: File prefix for model and scaler
    "Lipophilicity (logP)": "my_admet_models_logp",
    "Aqueous Solubility": "my_admet_models_solubility",
    "hERG Inhibition": "my_admet_models_herg",
    "Ames Mutagenicity": "my_admet_models_ames",
}

# ============================================================================
# ADMET PREDICTION FUNCTIONS
# ============================================================================

# Global variable to cache loaded models
@st.cache_resource
def load_admet_models():
    """Load all trained ADMET models and their scalers."""
    models = {}

    if not os.path.exists(ADMET_MODEL_DIR):
        st.error(f"❌ ADMET Directory not found: {ADMET_MODEL_DIR}")
        return models

    for display_name, file_prefix in ADMET_MODEL_CONFIG.items():
        model_path = f"{ADMET_MODEL_DIR}/{file_prefix}_model.pkl"
        scaler_path = f"{ADMET_MODEL_DIR}/{file_prefix}_scaler.pkl"

        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

                models[display_name] = {
                    'model': model,
                    'scaler': scaler,
                    'type': 'regression' if 'logp' in file_prefix or 'solubility' in file_prefix else 'classification'
                }
                # st.write(f"✅ Loaded {display_name}") # Uncomment for debug
            except Exception as e:
                st.warning(f"⚠️ Failed to load {display_name} models: {e}")
        else:
            st.warning(f"⚠️ Model file not found for {display_name}: {model_path}")

    return models

# Reusing the featurize(smiles) function from your bioactivity section.
# Since the featurization features match what the ADMET models were trained on!

def predict_admet(smiles, admet_models_dict):
    """Predict all available ADMET properties for a single compound."""
    
    # 1. Featurize the molecule (reuse your existing featurize function)
    features = featurize(smiles) 
    if features is None:
        return {'SMILES': smiles, 'Status': '❌ Invalid SMILES'}

    # Convert to DataFrame (must match column names used during training)
    X = pd.DataFrame([features]).fillna(0)

    results = {'SMILES': smiles, 'Status': '✅ Success'}
    
    for display_name, info in admet_models_dict.items():
        model = info['model']
        scaler = info['scaler']
        model_type = info['type']
        
        # 2. Scale the features if a scaler is available
        X_scaled = scaler.transform(X) if scaler else X.copy()
        
        # 3. Predict
        try:
            if model_type == 'classification':
                # Assuming 1 = Active/Mutagenic/Inhibitor, 0 = Inactive/Non-mutagenic/Non-inhibitor
                pred_class = model.predict(X_scaled)[0]
                pred_label = 'Positive' if pred_class == 1 else 'Negative'
                
                if 'hERG' in display_name:
                    results[display_name] = f"{pred_label} (Inhibitor)"
                elif 'Ames' in display_name:
                    results[display_name] = f"{pred_label} (Mutagenic)"
                else:
                    results[display_name] = pred_label

            else:  # Regression (logP, Solubility)
                prediction = model.predict(X_scaled)[0]
                results[display_name] = f"{prediction:.2f}"
                
        except Exception as e:
            results[display_name] = f"Error: {e}"

    return results





# ============================================================================
# CORE CLASSES (LLM Client Setup)
# ============================================================================
class GroqClient:
    """Handles Groq LLM interactions for analysis."""
    # Note: The Groq module is imported at the top of vision_app.py
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "mixtral-8x7b-32768"

    def generate_expert_analysis(self, compound_df: pd.DataFrame, target_disease: str) -> str:
        """Generates a summary of compounds for a target disease using LLM."""
        if compound_df.empty:
            return "**Analysis Failed:** No compounds were provided for analysis."
            
        compound_info = compound_df[['Compound_Name', 'canonical_smiles', 'molecular_weight']].head(5).to_markdown(index=False)
        
        system_prompt = f"""You are AfroMediBot, an expert cheminformatics and medicinal chemistry analyst. 
        Your task is to analyze the provided plant compounds and generate a concise, expert report 
        (in Markdown format) on their potential against {target_disease} based on their structures and 
        physicochemical properties (which you must infer using RDKit principles like Lipinski's Rule of Five). 
        
        Focus on: 1. Drug-likeness assessment. 2. Potential mechanism of action based on structural motifs. 
        3. A simple recommendation (High/Medium/Low priority).
        """
        
        user_query = f"""
        Generate an expert analysis report for the following compounds identified as relevant to '{target_disease}'.
        
        Compound Data:\n{compound_info}
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                model=self.model,
                temperature=0.2
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq API Error: {e}")
            return f"**LLM Error:** Could not generate analysis report due to API issue. Details: {e}"
    def generate_docking_analysis(self, docking_df: pd.DataFrame, target_protein: str) -> str:
            """Generates a professional interpretation of molecular docking simulation results."""
            if docking_df.empty:
                return "**Analysis Failed:** No docking results found."
    
            # Sort by binding energy (lower/more negative is better)
            best_compounds = docking_df.sort_values(
                'Binding Energy (kcal/mol)', ascending=True
            ).head(3)
            
            # Format results for the prompt
            docking_info = best_compounds[['SMILES', 'Binding Energy (kcal/mol)', 'Binding Affinity']].to_markdown(index=False)
            
            system_prompt = f"""You are AfroMediBot, a structural biology and molecular modeling expert. 
            Analyze the provided virtual docking results against the target protein: {target_protein}.
    
            Focus on: 
            1. **Summary of Affinities**: Identify the range and average binding energy.
            2. **Top Candidate Rationale**: Why is the highest affinity compound a good lead? (Relate energy to binding strength).
            3. **Simulation Caveats**: Mention the limitations of in silico (virtual) docking studies.
            
            Provide the answer in concise Markdown format.
            """
            
            user_query = f"""
            Generate a report analyzing the following virtual docking simulation results:
            
            Target Protein: {target_protein}
            
            Top Compounds Docking Results:\n{docking_info}
            """
            
            try:
                client = Groq(api_key=self.client.api_key)
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                    ],
                    model=self.model,
                    temperature=0.2
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Groq API Error: {e}")
                return f"**LLM Error:** Could not interpret docking results. Details: {e}"
    def generate_bioactivity_analysis(self, results_df: pd.DataFrame, target_query: str) -> str:
        """
        NEW METHOD: Generates expert analysis for bioactivity prediction results.
        """
        if results_df.empty:
            return "**Analysis Failed:** No bioactivity results to analyze."
        
        # Prepare top candidates markdown table
        top_candidates_markdown = "SMILES | MW (Da) | LogP | Activity | Confidence\n---|---|---|---|---\n"
        
        for _, row in results_df.head(10).iterrows():
            smiles = str(row.get('SMILES', 'Unknown'))[:40] + "..."
            mw = f"{row.get('Molecular Weight', 0):.1f}"
            logp = f"{row.get('LogP', 0):.2f}"
            activity = row.get('Predicted Activity', 'Unknown')
            confidence = row.get('Confidence', 'N/A')
            top_candidates_markdown += f"{smiles} | {mw} | {logp} | {activity} | {confidence}\n"

        system_prompt = f"""You are AfroMediBot, an AI Expert in natural products drug discovery and bioactivity prediction.
Your task is to provide a concise, professional expert analysis report based on the ML-predicted bioactivity screening results.
Focus on the top candidates' drug-likeness and predicted activity for the target: {target_query}.

Key Data:
- Total Candidates Screened: {len(results_df)}
- Active Candidates: {(results_df.get('Predicted Activity', pd.Series()) == 'Active').sum()}
- Target: {target_query}

Top 10 Candidates Data:
{top_candidates_markdown}

Structure your response with markdown headings:
## 🔬 Bioactivity Expert Analysis Report
### 1. Summary of Screening Results
### 2. Physicochemical Assessment (Key trends in MW, LogP)
### 3. Lead Candidate Recommendation (Highlight top 2-3 active compounds)
### 4. Next Steps (In vitro validation, optimization suggestions)

Be concise, scientific, and highlight the most promising molecule(s)."""

        user_query = f"Generate the expert bioactivity analysis report for the screening: {target_query}."
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API Error: {e}")
            return f"**LLM Error:** Could not generate bioactivity analysis. Details: {e}"

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
        
        if 'organisms' not in self.df.columns:
            return None
        
        # First try: resolved name
        results = self.df[self.df['organisms'].str.contains(resolved, case=False, na=False)]
        
        # Second try: original input if resolved failed
        if len(results) == 0 and resolved.lower() != plant_name.lower():
            results = self.df[self.df['organisms'].str.contains(plant_name, case=False, na=False)]
        
        # Return results or None
        return results.head(top_n) if len(results) > 0 else None

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/200x80/2E7D32/FFFFFF?text=AfroMediBot", width='stretch')
    st.markdown("---")
    
    st.subheader("⚙️ API Configuration")
    
    # Groq API Key
    groq_api_key_input = st.text_input("Groq API Key", type="password", help="Get free key at console.groq.com")
    
    # Capture the key into session state for global access
    if groq_api_key_input:
        st.session_state['groq_api_key'] = groq_api_key_input
        # Also set environment variable for other services (like Entrez/PubMed)
        os.environ["GROQ_API_KEY"] = groq_api_key_input
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
    st.sidebar.markdown("### 💾 Compounds Database")
    
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
                st.session_state['compounds_df'] = df_new # for button compatibility
                st.sidebar.success(f"Database Loaded: {len(df_new)} compounds.")
            else:
                st.sidebar.error("CSV must contain an 'organisms' column.")
                st.session_state['database'] = pd.DataFrame()
                st.session_state['compounds_df'] = None
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
            st.session_state['database'] = pd.DataFrame()
            st.session_state['database'] = None
    else:
        # Initialize the database as an empty DataFrame or check existing state
        if 'database' not in st.session_state:
            st.session_state['database'] = pd.DataFrame()
        
        if st.session_state['database'].empty:
            st.sidebar.info("Upload a CSV to enable compound searching.")
            st.session_state['compounds_df'] = None
        else:
            st.sidebar.success(f"✅ Active Database: {len(st.session_state['database'])} compounds.")
            st.session_state['compounds_df'] = st.session_state['database']
    
    
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

# Initialize active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
tab_home, tab_literature, tab_3d, tab_plant, tab_bio, tab_dock, tab_admet, tab_synthesis = st.tabs([
    "🏠 Home",
    "📚 Literature Mining",
    "🧊 3D Molecule Viewer",
    "🌿 Plant Recognition",
    "🔬Bioactivity Analysis",
    "🔗Molecular Docking",
    "ADMET PREDICTION",
    "🖥️🧪 Retrosynthesis"
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
            st.session_state.active_tab = "1" # Switch to Literature tab
            st.rerun()
    
    with col2:
        if st.button("🧊 View 3D Molecules", width='stretch', type="primary", key="home_view_3d"):
            st.session_state.active_tab = "2"
            st.rerun()
    
    with col3:
        if st.button("🌿 Identify Plant", width='stretch', type="primary", key="home_identify_plant"):
            st.session_state.active_tab = "3"
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
                # --- START OF PIPELINE INTEGRATION ---
                # 1. Save the successfully generated SMILES to session state for other tabs
                st.session_state['smiles_from_generator'] = [smiles_input]
                st.info(f"✨ **Pipeline Ready:** Compound '{mol_name if mol_name else smiles_input[:20]+'...'}' is now available in the Bioactivity, Docking, and ADMET tabs.")
                # --- END OF PIPELINE INTEGRATION ---
                
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
    # Add tabs for different search methods
    search_tab1, search_tab2 = st.tabs(["📷 Image Recognition", "✏️ Manual Search"])
    
    with search_tab1:
        st.info("Upload a plant image to identify species using AI")
        
        uploaded_image = st.file_uploader(
            "Upload Plant Image",
            type=['jpg', 'jpeg', 'png'],
            help="Take a clear photo of the plant (leaf, flower, or whole plant)",
            key="plant_image_uploader"
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
                                uploaded_image.seek(0)
                                labels, entities = identify_plant_google_vision(uploaded_image, 'vision-credentials.json')
                                identified_species_name = extract_plant_species(labels, entities)
                                st.session_state.identified_species = identified_species_name
                                st.session_state.vision_labels = labels
                                st.session_state.vision_entities = entities
                            except Exception as e:
                                st.error(f"Error during Vision API analysis: {e}")
                                st.exception(e)
            
            # Display identification results
            if 'identified_species' in st.session_state:
                identified_species_name = st.session_state.identified_species
                labels = st.session_state.get('vision_labels', [])
                entities = st.session_state.get('vision_entities', [])
                
                st.markdown("---")
                st.subheader("🌿 Primary Identification")
                
                if identified_species_name != "Unknown plant":
                    st.success(f"**Identified Species:** **{identified_species_name}**")
                else:
                    st.warning("Could not definitively identify species.")
                    
                st.markdown("---")
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
    
                st.markdown("---")
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    has_database = False
                    if 'compounds_df' in st.session_state and st.session_state.compounds_df is not None:
                        has_database = not st.session_state.compounds_df.empty
                    elif 'database' in st.session_state:
                        has_database = not st.session_state['database'].empty
                    
                    has_species = 'identified_species' in st.session_state
                    map_disabled = not (has_database and has_species)
                    
                    if st.button(
                        "🗺️ Map Plant Name", 
                        key="map_plant_name", 
                        type="secondary",
                        use_container_width=True,
                        disabled=map_disabled,
                        help="Map common name to scientific name and find compounds" if not map_disabled else "Upload database first"
                    ):
                        if 'compounds_df' not in st.session_state or st.session_state.compounds_df is None:
                            if 'database' not in st.session_state or st.session_state['database'].empty:
                                st.error("⚠️ Please upload a compounds database in the sidebar first")
                            else:
                                try:
                                    identified_species_name = st.session_state.identified_species
                                    db_df = st.session_state.get('compounds_df') or st.session_state.get('database')
                                    # DEBUG
                                    st.write(f"DEBUG - Input: {identified_species_name}")
                                    st.write(f"DEBUG - Database columns: {db_df.columns.tolist()}")
                                    st.write(f"DEBUG - First organism entry: {db_df['organisms'].iloc[0] if 'organisms' in db_df.columns else 'NO ORGANISMS COLUMN'}")

                                    if db_df is None or db_df.empty:
                                        st.error("⚠️ Database is empty or not loaded properly")
                                    else:
                                        plant_agent = PlantAgent(db_df)
                                        resolved_name = plant_agent.resolve_plant_name(identified_species_name)
                                        st.write(f"DEBUG - Resolved to: {resolved_name}")  # Check mapping
                                        st.session_state.resolved_plant_name = resolved_name
                                        
                                        st.subheader("🔍 Mapping Results")
                                        if resolved_name.lower() != identified_species_name.lower():
                                            st.success(f"**Common Name:** {identified_species_name}")
                                            st.success(f"**Scientific Name:** {resolved_name}")
                                        else:
                                            st.info(f"**Name:** {resolved_name}")
                                        
                                        plant_compounds = plant_agent.search_by_plant(resolved_name, top_n=50)
                                        
                                        if plant_compounds is not None and not plant_compounds.empty:
                                            st.subheader(f"💊 Found {len(plant_compounds)} Compounds")
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
                                            
                                            st.session_state.mapped_plant = resolved_name
                                            st.session_state.plant_compounds = plant_compounds
                                        else:
                                            st.warning(f"No compounds found for {resolved_name} in database")
                                except ValueError as ve:
                                    st.error(f"❌ Error: {ve}")
                                except Exception as e:
                                    st.error(f"❌ Unexpected error during mapping: {e}")
                                    st.exception(e)
                        else:
                            try:
                                identified_species_name = st.session_state.identified_species
                                db_df = st.session_state.get('compounds_df') or st.session_state.get('database')
                                
                                if db_df is None or db_df.empty:
                                    st.error("⚠️ Database is empty or not loaded properly")
                                else:
                                    plant_agent = PlantAgent(db_df)
                                    resolved_name = plant_agent.resolve_plant_name(identified_species_name)
                                    st.session_state.resolved_plant_name = resolved_name
                                    
                                    st.subheader("🔍 Mapping Results")
                                    if resolved_name.lower() != identified_species_name.lower():
                                        st.success(f"**Common Name:** {identified_species_name}")
                                        st.success(f"**Scientific Name:** {resolved_name}")
                                    else:
                                        st.info(f"**Name:** {resolved_name}")
                                    
                                    plant_compounds = plant_agent.search_by_plant(resolved_name, top_n=50)
                                    
                                    if plant_compounds is not None and not plant_compounds.empty:
                                        st.subheader(f"💊 Found {len(plant_compounds)} Compounds")
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
                                        
                                        st.session_state.mapped_plant = resolved_name
                                        st.session_state.plant_compounds = plant_compounds
                                    else:
                                        st.warning(f"No compounds found for {resolved_name} in database")
                            except ValueError as ve:
                                st.error(f"❌ Error: {ve}")
                            except Exception as e:
                                st.error(f"❌ Unexpected error during mapping: {e}")
                                st.exception(e)
                
                if 'resolved_plant_name' in st.session_state and 'mapped_plant' not in st.session_state:
                    st.info(f"📋 Last mapped: **{st.session_state.resolved_plant_name}** - Click 'Map Plant Name' again to search compounds")
                
                with col_btn2:
                    filter_disabled = ('plant_compounds' not in st.session_state or 
                                      st.session_state.plant_compounds is None or
                                      st.session_state.plant_compounds.empty)
                    
                    if st.button(
                        "🔬 Filter Compounds", 
                        key="filter_plant_compounds", 
                        type="primary",
                        use_container_width=True,
                        disabled=filter_disabled,
                        help="Filter and analyze found compounds" if not filter_disabled else "Map plant name first"
                    ):
                        if filter_disabled:
                            st.warning("⚠️ Please map the plant name first to find compounds")
                        else:
                            st.session_state.show_filters = True
                
                if (st.session_state.get('show_filters', False) and 
                    'plant_compounds' in st.session_state and 
                    st.session_state.plant_compounds is not None and
                    not st.session_state.plant_compounds.empty):
                    
                    st.markdown("---")
                    st.subheader("🎯 Compound Filtering")
                    
                    try:
                        compounds = st.session_state.plant_compounds.copy()
                        mapped_plant = st.session_state.get('mapped_plant', 
                                                            st.session_state.get('identified_species', 'Unknown'))
                        
                        st.write(f"**Source Plant:** {mapped_plant}")
                        st.write(f"**Total Compounds:** {len(compounds)}")
                        
                        filter_col1, filter_col2 = st.columns(2)
                        
                        with filter_col1:
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
                        
                        st.write(f"**Filtered Results:** {len(compounds)} compounds")
                        st.dataframe(compounds)
                        
                        csv = compounds.to_csv(index=False)
                        st.download_button(
                            "📥 Download Filtered Compounds",
                            data=csv,
                            file_name=f"filtered_compounds_{mapped_plant}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"❌ Error during filtering: {e}")
                        st.session_state.show_filters = False
    
    # Manual Search Tab (same indentation level as search_tab1)
    with search_tab2:
        st.info("Enter a plant name directly to search the compounds database")
        manual_plant_name = st.text_input(
            "Plant Name (Common or Scientific)",
            placeholder="e.g., Moringa oleifera, Bitter leaf, Neem",
            key="manual_plant_search"
        )
        
        if manual_plant_name.strip():
            st.session_state.identified_species = manual_plant_name.strip()
            st.success(f"✅ Selected plant: **{manual_plant_name.strip()}**")
            
            st.markdown("---")
            col_btn1, col_btn2 = st.columns(2)
            
            # Add the same button logic here for manual search
            # (You can copy the button code from search_tab1)
            with col_btn1:
                has_database = False
                if 'compounds_df' in st.session_state and st.session_state.compounds_df is not None:
                    has_database = not st.session_state.compounds_df.empty
                elif 'database' in st.session_state:
                    has_database = not st.session_state['database'].empty
                
                has_species = 'identified_species' in st.session_state
                map_disabled = not (has_database and has_species)
                
                if st.button(
                    "🗺️ Map Plant Name", 
                    key="map_plant_name_manual",  # ← DIFFERENT KEY!
                    type="secondary",
                    use_container_width=True,
                    disabled=map_disabled,
                    help="Map common name to scientific name and find compounds" if not map_disabled else "Upload database first"
                ):
                    if not has_database:
                        st.error("⚠️ Please upload a compounds database in the sidebar first")
                    else:
                        try:
                            identified_species_name = st.session_state.identified_species
                            db_df = st.session_state.get('compounds_df') or st.session_state.get('database')
                            
                            if db_df is None or db_df.empty:
                                st.error("⚠️ Database is empty or not loaded properly")
                            else:
                                plant_agent = PlantAgent(db_df)
                                resolved_name = plant_agent.resolve_plant_name(identified_species_name)
                                st.session_state.resolved_plant_name = resolved_name
                                
                                st.subheader("🔍 Mapping Results")
                                if resolved_name.lower() != identified_species_name.lower():
                                    st.success(f"**Common Name:** {identified_species_name}")
                                    st.success(f"**Scientific Name:** {resolved_name}")
                                else:
                                    st.info(f"**Name:** {resolved_name}")
                                
                                plant_compounds = plant_agent.search_by_plant(resolved_name, top_n=50)
                                
                                if plant_compounds is not None and not plant_compounds.empty:
                                    st.subheader(f"💊 Found {len(plant_compounds)} Compounds")
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
                                    
                                    st.session_state.mapped_plant = resolved_name
                                    st.session_state.plant_compounds = plant_compounds
                                else:
                                    st.warning(f"No compounds found for {resolved_name} in database")
                        except ValueError as ve:
                            st.error(f"❌ Error: {ve}")
                        except Exception as e:
                            st.error(f"❌ Unexpected error during mapping: {e}")
                            st.exception(e)
            
            # Display previous mapping results if they exist
            if 'resolved_plant_name' in st.session_state and 'mapped_plant' not in st.session_state:
                st.info(f"📋 Last mapped: **{st.session_state.resolved_plant_name}** - Click 'Map Plant Name' again to search compounds")
            
            with col_btn2:
                filter_disabled = ('plant_compounds' not in st.session_state or 
                                  st.session_state.plant_compounds is None or
                                  st.session_state.plant_compounds.empty)
                
                if st.button(
                    "🔬 Filter Compounds", 
                    key="filter_plant_compounds_manual",  # ← DIFFERENT KEY!
                    type="primary",
                    use_container_width=True,
                    disabled=filter_disabled,
                    help="Filter and analyze found compounds" if not filter_disabled else "Map plant name first"
                ):
                    if filter_disabled:
                        st.warning("⚠️ Please map the plant name first to find compounds")
                    else:
                        st.session_state.show_filters = True
            
            # Display filter UI (same as search_tab1)
            if (st.session_state.get('show_filters', False) and 
                'plant_compounds' in st.session_state and 
                st.session_state.plant_compounds is not None and
                not st.session_state.plant_compounds.empty):
                
                st.markdown("---")
                st.subheader("🎯 Compound Filtering")
                
                try:
                    compounds = st.session_state.plant_compounds.copy()
                    mapped_plant = st.session_state.get('mapped_plant', 
                                                        st.session_state.get('identified_species', 'Unknown'))
                    
                    st.write(f"**Source Plant:** {mapped_plant}")
                    st.write(f"**Total Compounds:** {len(compounds)}")
                    
                    filter_col1, filter_col2 = st.columns(2)
                    
                    with filter_col1:
                        if 'molecular_weight' in compounds.columns:
                            mw_min = float(compounds['molecular_weight'].min())
                            mw_max = float(compounds['molecular_weight'].max())
                            mw_range = st.slider(
                                "Molecular Weight Range",
                                mw_min,
                                mw_max,
                                (mw_min, mw_max),
                                key="mw_slider_manual"  # ← DIFFERENT KEY!
                            )
                            compounds = compounds[
                                (compounds['molecular_weight'] >= mw_range[0]) & 
                                (compounds['molecular_weight'] <= mw_range[1])
                            ]
                    
                    with filter_col2:
                        if 'activity_type' in compounds.columns:
                            activities = compounds['activity_type'].unique().tolist()
                            selected_activities = st.multiselect(
                                "Filter by Activity",
                                activities,
                                default=activities[:3] if len(activities) > 3 else activities,
                                key="activity_filter_manual"  # ← DIFFERENT KEY!
                            )
                            if selected_activities:
                                compounds = compounds[compounds['activity_type'].isin(selected_activities)]
                    
                    st.write(f"**Filtered Results:** {len(compounds)} compounds")
                    st.dataframe(compounds)
                    
                    csv = compounds.to_csv(index=False)
                    st.download_button(
                        "📥 Download Filtered Compounds",
                        data=csv,
                        file_name=f"filtered_compounds_{mapped_plant}.csv",
                        mime="text/csv",
                        key="download_manual"  # ← DIFFERENT KEY!
                    )
                except Exception as e:
                    st.error(f"❌ Error during filtering: {e}")
                    st.session_state.show_filters = False           

# ========================================================================
# BIOACTIVITY PREDICTION TAB (CONSOLIDATED)
# ========================================================================
with tab_bio:
    st.header("🧬 Bioactivity Prediction")
    st.info("Predict compound activity against disease targets using trained ML models")
    
    # Load models
    models_dict = load_bioactivity_models()
    st.success(f"✅ Loaded {len(models_dict)} bioactivity models")

    if not models_dict:
        st.error("❌ No bioactivity models found in 'main/models/bioactivity/' folder.")
        st.info("Please ensure your model files are in: models/bioactivity/egfr_model.joblib, etc.")
        st.stop()

    # Target selection - ONLY show available models
    with st.expander("🎯 Available Models", expanded=False):
        for target, info in models_dict.items():
            st.write(f"✅ {target} ({info['type']})")
    
    col1, col2 = st.columns([2, 1])
    
    # Get database
    df = st.session_state.get('database')
    groq_api_key = st.session_state.get('groq_api_key')
        
    with col1:
        # Input method
        input_method = st.radio(
            "Input Method:",
            ["Upload CSV", "Paste SMILES", "Search by Plant/Compound Name"],
            key='bio_input'
        )
        
        bio_smiles = []
        
        if input_method == "Upload CSV":
            bio_csv = st.file_uploader("Upload CSV with SMILES", type=['csv'], key='bio_csv')
            if bio_csv:
                bio_df = pd.read_csv(bio_csv)
                smiles_col = st.selectbox("Select SMILES column:", bio_df.columns, key='bio_smiles_col')
                bio_smiles = bio_df[smiles_col].dropna().tolist()[:50]
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
                matches = df[
                    df.get('organisms', pd.Series()).astype(str).str.contains(search_name, case=False, na=False) |
                    df.get('name', pd.Series()).astype(str).str.contains(search_name, case=False, na=False)
                ]
                
                if len(matches) > 0:
                    st.success(f"✅ Found {len(matches)} matches")
                    with st.expander("👀 View compounds"):
                        st.dataframe(matches[['name', 'organisms', 'molecular_weight']].head(10))
                    
                    if 'canonical_smiles' in matches.columns:
                        bio_smiles = matches['canonical_smiles'].dropna().tolist()[:50]
                    elif 'SMILES' in matches.columns:
                        bio_smiles = matches['SMILES'].dropna().tolist()[:50]
                    else:
                        st.error("No SMILES column found in matches")
                        bio_smiles = []
                    st.info(f"📊 Selected {len(bio_smiles)} compounds")
    
    with col2:
        # Target selection - dynamically populated
        target = st.selectbox(
            "Select Target:",
            list(models_dict.keys()) if models_dict else ["No models loaded"],
            key='bio_target'
        )
    
    # Predict button
    if st.button("🔬 Predict Bioactivity", key='predict_bio'):
        st.write(f"**Debug:** Processing {len(bio_smiles)} SMILES")
        st.write(f"**First 3 SMILES:** {bio_smiles[:3]}")
        if not bio_smiles:
            st.error("Please provide SMILES first")
        elif not models_dict:
            st.error("No models loaded")
        else:
            with st.spinner(f"Predicting {target} activity..."):
                results = []
                
                for smiles in bio_smiles[:20]:  # Limit to 20
                    # Predict with real model
                    prediction = predict_bioactivity(smiles, target, models_dict)
                    
                    if prediction:
                        # Also calculate basic properties
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            # Get model type
                            model_type = models_dict[target]['type']
                            
                            # Build result row based on model type
                            result_row = {
                                'SMILES': smiles[:50] + '...',
                                'Molecular Weight': round(Descriptors.MolWt(mol), 1),
                                'LogP': round(Descriptors.MolLogP(mol), 2),
                                'Predicted Activity': prediction['prediction'],
                                'Model Type': model_type.title()
                            }
                            
                            # Add type-specific columns
                            if model_type == 'classification':
                                result_row['Confidence'] = f"{prediction['confidence']:.1%}"
                                result_row['Activity Probability'] = f"{prediction['activity_probability']:.1%}"
                            else:  # regression
                                result_row['IC50 (μM)'] = f"{prediction['ic50_um']:.2f}"
                                result_row['Confidence'] = f"{prediction['confidence']:.1%}"
                            
                            results.append(result_row)
                
                if results:
                    st.session_state['bio_results_df'] = pd.DataFrame(results)
                    st.session_state['bio_target_query'] = target
                    st.session_state['bio_llm_analysis_report'] = ""
                    st.session_state['bio_model_type'] = models_dict[target]['type']
                    st.rerun()
    
    # Display results
    if 'bio_results_df' in st.session_state and not st.session_state['bio_results_df'].empty:
        results_df = st.session_state['bio_results_df']
        target_query = st.session_state['bio_target_query']
        model_type = st.session_state.get('bio_model_type', 'unknown')
        
        st.subheader("📊 Prediction Results")
        # Display model type indicator
        st.info(f"**Model Type:** {model_type.title()} | **Target:** {target_query}")

        # show dataframe
        st.dataframe(results_df, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tested", len(results_df))
        with col2:
            active_count = (results_df['Predicted Activity'] == 'Active').sum()
            st.metric("Active Compounds", active_count)
        with col3:
            if model_type == 'regression':
                # Show average IC50 for regression models
                avg_ic50 = results_df['IC50 (μM)'].str.replace(' μM', '').astype(float).mean()
                st.metric("Avg IC50 (μM)", f"{avg_ic50:.2f}")
            else:
                # Show hit rate for classification models
                hit_rate = (active_count / len(results_df)) * 100
                st.metric("Hit Rate", f"{hit_rate:.1f}%")
        
        # LLM Analysis
        st.markdown("---")
        st.subheader("🤖 Expert Interpretation")
        
        if not groq_api_key:
            st.warning("Enter Groq API Key in sidebar to generate AI analysis")
        elif st.button("🚀 Generate Expert Analysis", type="secondary", key='run_llm_bio'):
            client = GroqClient(groq_api_key)
            with st.spinner(f"Analyzing results for {target_query}..."):
                report = client.generate_expert_analysis(results_df, target_query)
            
            st.session_state['llm_analysis_report'] = report
            st.rerun()
        
        # Display report
        if 'llm_analysis_report' in st.session_state and st.session_state['llm_analysis_report']:
            st.markdown(st.session_state['llm_analysis_report'])
        
        # Download
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results",
            data=csv,
            file_name=f"bioactivity_{target_query.replace(' ', '_')}.csv",
            mime="text/csv"
        )
# ========================================================================
# MOLECULAR DOCKING TAB (MODIFIED for LLM Analysis)
# ========================================================================
with tab_dock:
    st.markdown("### 🎯 Molecular Docking")
    st.info("Simulate compound binding to protein targets")

    # [INSERTION POINT A] - Initialize the session state variable
    if 'all_dockers' not in st.session_state:
        st.session_state.all_dockers = {}

    
    # Get key from state
    #groq_api_key = st.session_state.get('groq_api_key')
    
    # Initialize docking agents (run once)
    if 'docking_initialized' not in st.session_state:
        with st.spinner("Initializing docking targets..."):
            num_loaded = initialize_docking_agents()
            st.session_state.docking_initialized = True
            if num_loaded > 0:
                st.success(f"✅ Loaded {num_loaded} docking targets")
            else:
                st.error("❌ No docking targets loaded. Check protein files.")
   
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dock_input = st.radio(
            "Input Method:",
            ["Upload CSV", "Paste SMILES", "Search Database"],
            key='dock_input'
        )
        
        dock_smiles = []

        # --- DOCKING INPUT LOGIC ---
        if dock_input == "Upload CSV":
            dock_csv = st.file_uploader("Upload CSV", type=['csv'], key='dock_csv')
            if dock_csv:
                dock_df = pd.read_csv(dock_csv)
                col = st.selectbox("SMILES column:", dock_df.columns, key='dock_col')
                dock_smiles = dock_df[col].dropna().tolist()[:10]
                st.success(f"✅ Loaded {len(dock_smiles)} SMILES")
        elif dock_input == "Paste SMILES":
            smiles_text = st.text_area("Paste SMILES:", key='dock_text')
            if smiles_text:
                dock_smiles = [s.strip() for s in smiles_text.split('\n') if s.strip()]
                st.success(f"✅ {len(dock_smiles)} SMILES")
        else: # Search Database
            search = st.text_input("Search Plant or Compound Name:", key='dock_search')
            if search and df is not None and not df.empty:
                # Assuming 'organisms' and 'canonical_smiles' columns exist in the database
                matches = df[df.get('organisms', pd.Series()).astype(str).str.contains(search, case=False, na=False)]
                
                if not matches.empty:
                    dock_smiles = matches['canonical_smiles'].dropna().head(10).tolist()
                    st.success(f"✅ Found {len(dock_smiles)} compounds from search.")
                else:
                    st.warning("No matches found in the database.")
            elif search:
                st.error("❌ Database not available for searching.")
    with col2:
        # ✅ Check session_state for loaded targets
        if not st.session_state.all_dockers:
            st.error("❌ No docking targets available")
        #    st.info("Check that .pdbqt files exist in 'proteins/' folder")
        #    st.stop()
        
        # Create display names from loaded targets
        target_display_names = {
            'cancer_EGFR': 'Cancer (EGFR)',
            'cancer_BCR_ABL': 'Cancer (BCR ABL)',
            'cancer_CDK': 'Cancer (CDK)',
            'cancer_VEGFR2': 'Cancer (VEGFR2)',
            'hiv_Protease': 'HIV (Protease)',
            'diabetes_DPP4': 'Diabetes (DPP4)',
            'tuberculosis_InhA': 'TB (InhA)',
            'hypertension_ACE': 'Hypertension (ACE)',
            'inflammation_COX2': 'Inflammation (COX2)',
        }
        
        # ✅ Build dropdown options ONLY from loaded targets
        available_options = {
            target_display_names.get(key, key): key 
            for key in st.session_state.all_dockers.keys()
        }
        
        st.info(f"📊 {len(available_options)} targets loaded")
        
        protein_display = st.selectbox(
            "Target Protein:",
            list(available_options.keys())
        )
        
        protein = available_options[protein_display]
    
        exhaustiveness = st.slider("Exhaustiveness:", 1, 10, 8)

    # --- Run Docking Simulation & Interpretation ---
    if st.button("🎯 Run Docking & Analysis", key='run_dock', type="primary"):
        if not dock_smiles:
            st.error("Please provide SMILES first")
        elif not groq_api_key:
            st.error("Please enter your Groq API Key in the sidebar to run the simulation and analysis.")
        else:
            with st.spinner(f"Docking {len(dock_smiles)} compounds and generating report..."):
                # 1. RUN SIMULATION (Placeholder results)
                dock_results = []
                for smiles in dock_smiles:
                    docking_result = perform_docking_for_target(smiles, protein, debug=False)
                    
                    if docking_result['binding_energy'] and docking_result['status'] == 'Success':
                        binding_energy = docking_result['binding_energy']
                        
                        # Classify affinity based on energy
                        if binding_energy < -8.0:
                            affinity = 'Strong'
                        elif binding_energy < -6.0:
                            affinity = 'Moderate'
                        else:
                            affinity = 'Weak'
                        
                        dock_results.append({
                            'SMILES': smiles[:40] + '...',
                            'Binding Energy (kcal/mol)': binding_energy,
                            'Binding Affinity': affinity,
                            'Status': '✅ Success'
                        })
                    else:
                        dock_results.append({
                            'SMILES': smiles[:40] + '...',
                            'Binding Energy (kcal/mol)': 'N/A',
                            'Binding Affinity': 'Failed',
                            'Status': '❌ Docking failed'
                        })
                dock_df = pd.DataFrame(dock_results).sort_values('Binding Energy (kcal/mol)')

                if len(dock_df) > 0:
                    # Sort by binding energy (lower is better)
                    dock_df_sorted = dock_df[dock_df['Binding Energy (kcal/mol)'] != 'N/A'].copy()
                    if len(dock_df_sorted) > 0:
                        dock_df_sorted = dock_df_sorted.sort_values('Binding Energy (kcal/mol)')
                        st.dataframe(dock_df_sorted, use_container_width=True)
                    else:
                        st.error("All docking attempts failed")
                    
                    st.success(f"✅ Docked {len(dock_results)} compounds")
                    
                    csv = dock_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Docking Results",
                        data=csv,
                        file_name="docking_results.csv",
                        mime="text/csv"
                    )

                
                # 2. RUN LLM INTERPRETATION
                client = GroqClient(groq_api_key)
                report = client.generate_docking_analysis(dock_df, f"{protein} (Simulated Target)")
                
                st.session_state['docking_results_df'] = dock_df
                st.session_state['docking_report'] = report
                st.rerun() # Rerun to display results below

    # --- Display Results ---
    if 'docking_results_df' in st.session_state and not st.session_state['docking_results_df'].empty:
        dock_df = st.session_state['docking_results_df']
        
        st.subheader("📊 Docking Results Table")
        st.dataframe(dock_df, use_container_width=True)
        
        st.success(f"✅ Docked {len(dock_df)} compounds successfully.")
        
        st.markdown("---")
        st.subheader("🤖 LLM Structural Analysis")
        st.markdown(st.session_state['docking_report'])

        # Download button
        csv = dock_df.to_csv(index=False)
        st.download_button(
            "📥 Download Docking Results",
            data=csv,
            file_name="docking_results.csv",
            mime="text/csv"
        )

# ============================================================================
# TAB 5: ADMET PREDICTION)
# ============================================================================

with tab_admet:
    st.header("⚖️ ADMET Prediction")
    st.info("Predict Absorption, Distribution, Metabolism, Excretion, and Toxicity properties")

    # Load ADMET Models
    admet_models = load_admet_models()

    if not admet_models:
        st.error("❌ No ADMET models found. Check 'models/admet/' directory.")
        st.stop()

    st.success(f"✅ Loaded {len(admet_models)} ADMET models: {', '.join(admet_models.keys())}")

    col1, col2 = st.columns([2, 1])

    # --- INPUT METHOD ---
    with col1:
        # Input method
        input_method = st.radio(
            "Input Method:",
            ["Upload CSV", "Paste SMILES", "Search Database", "Use Generator Output"],
            key='admet_input'
        )

        admet_smiles = []
        df = st.session_state.get('database') # Assuming database is loaded into session state

        if input_method == "Upload CSV":
            admet_csv = st.file_uploader("Upload CSV with SMILES", type=['csv'], key='admet_csv')
            if admet_csv:
                admet_df_in = pd.read_csv(admet_csv)
                smiles_col = st.selectbox("Select SMILES column:", admet_df_in.columns, key='admet_smiles_col')
                admet_smiles = admet_df_in[smiles_col].dropna().tolist()[:50]
                st.success(f"✅ Loaded {len(admet_smiles)} SMILES")

        elif input_method == "Paste SMILES":
            smiles_input = st.text_area(
                "Paste SMILES (one per line):",
                placeholder="CCO\nCC(=O)O",
                key='admet_smiles_text'
            )
            if smiles_input:
                admet_smiles = [s.strip() for s in smiles_input.split('\n') if s.strip()]
                st.success(f"✅ {len(admet_smiles)} SMILES entered")

        elif input_method == "Search Database":
            # Reuse search logic from bioactivity
            search_name = st.text_input("Enter plant or compound name:", key='admet_search')
            # ... (implement search logic like in bioactivity tab)
            # Placeholder:
            if search_name and df is not None:
                st.info("💡 Search logic needs to be implemented here, similar to the bioactivity tab.")
                # For demonstration, use a placeholder list
                admet_smiles = []

        elif input_method == "Use Generator Output":
            if 'smiles_from_generator' in st.session_state:
                admet_smiles = st.session_state['smiles_from_generator']
                st.info(f"✅ Using {len(admet_smiles)} SMILES from the Molecule Generator.")
            else:
                st.warning("No SMILES found in the Molecule Generator output.")


    with col2:
        # Show loaded models list
        st.markdown("##### Models to be run:")
        for name in admet_models.keys():
            st.markdown(f"* {name}")

    # --- RUN PREDICTION ---
    if st.button("📈 Run ADMET Profiling", key='run_admet'):
        if not admet_smiles:
            st.error("Please provide SMILES first.")
        else:
            all_admet_results = []
            with st.spinner(f"Predicting ADMET properties for {len(admet_smiles)} compounds..."):

                for smiles in admet_smiles[:50]:
                    result_row = predict_admet(smiles, admet_models)
                    all_admet_results.append(result_row)

            if all_admet_results:
                admet_df = pd.DataFrame(all_admet_results)
                st.session_state['admet_results_df'] = admet_df
                st.rerun()

    # --- DISPLAY RESULTS ---
    if 'admet_results_df' in st.session_state and not st.session_state['admet_results_df'].empty:
        results_df = st.session_state['admet_results_df']

        st.subheader("Complete ADMET Profile")
        st.dataframe(results_df, use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 Expert Interpretation")
        st.info("💡 Implement LLM analysis here, similar to Bioactivity/Docking.")

        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download ADMET Results",
            data=csv,
            file_name="admet_profile.csv",
            mime="text/csv"
                )


# ============================================================================
# TAB 6: RETROSYNTHESIS
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
