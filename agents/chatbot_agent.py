# ============================================================================
# PREDICTOR AGENT - Unified ML Predictions
# ============================================================================

class PredictorAgent:
    """Handles all ML bioactivity predictions"""
    
    def __init__(self, model_folder=''):
        self.models = self.load_all_models(model_folder)
        self.disease_map = {
            'cancer': ['cancer_EGFR', 'cancer_BCR_ABL', 'cancer_HER2'],
            'hiv': ['hiv_HIV_RT', 'hiv_HIV_Protease'],
            'diabetes': ['diabetes_DPP4', 'diabetes_Alpha_Glucosidase'],
            'malaria': ['malaria_PfDHFR'],
            'tuberculosis': ['tuberculosis_InhA'],
            'inflammation': ['inflammation_COX2']
        }
    
    def load_all_models(self, folder):
        import glob, joblib
        models = {}
        for f in glob.glob(f'{folder}*_regression*.joblib'):
            name = f.split('/')[-1].replace('_regression_model.joblib', '').replace('_regression.joblib', '')
            models[name] = joblib.load(f)
        print(f"✅ Loaded {len(models)} models")
        return models
    
    def featurize(self, smiles):
        """Same featurization as training"""
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
        import numpy as np
        
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
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            'RingCount': Descriptors.RingCount(mol),
            'FractionCsp3': Descriptors.FractionCSP3(mol),
            'BertzCT': Descriptors.BertzCT(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        }
        
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
                raise Exception()
        except:
            feature_dict.update({
                'Asphericity': 0, 'Eccentricity': 0, 'InertialShapeFactor': 0,
                'RadiusOfGyration': 0, 'SpherocityIndex': 0,
                'PMI1': 0, 'PMI2': 0, 'PMI3': 0,
            })
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
        for j, bit in enumerate(np.array(fp)):
            feature_dict[f'fp_{j}'] = bit
        
        return feature_dict
    
    def predict(self, compounds_df, disease):
        """Predict bioactivity for disease"""
        targets = self.disease_map.get(disease, [])
        predictions = []
        
        for target in targets:
            if target not in self.models:
                continue
            
            model = self.models[target]
            
            for idx, row in compounds_df.head(50).iterrows():
                feat = self.featurize(row['canonical_smiles'])
                if feat:
                    X = pd.DataFrame([feat]).fillna(0)
                    ic50 = 10 ** model.predict(X)[0]
                    
                    if ic50 < 10:
                        predictions.append({
                            'name': row.get('name', 'Unknown'),
                            'smiles': row['canonical_smiles'],
                            'ic50_predicted': ic50,
                            'target': target,
                            'mw': row['molecular_weight'],
                            'qed': row['qed_drug_likeliness']
                        })
        
        return pd.DataFrame(predictions).sort_values('ic50_predicted') if predictions else None

# ============================================================================
# CHATBOT AGENT - Agentic Interface
# ============================================================================

class ChatbotAgent:
    """Main agentic chatbot for drug discovery"""
    
    def __init__(self, df, dockers=None):
        self.plant_agent = PlantAgent(df)
        self.filter_agent = FilterAgent(df)
        self.predictor = PredictorAgent()
        self.dockers = dockers  # Your docking agents
    
    def chat(self, user_input):
        """Main chat interface"""
        intent = self.parse_intent(user_input)
        
        if intent['type'] == 'screen':
            return self.workflow_screen(intent)
        elif intent['type'] == 'info':
            return f"ℹ️ {intent['plant']} information coming soon..."
        else:
            return "Please ask me to screen a plant for a disease. Example: 'Screen bitter leaf for cancer'"
    
    def parse_intent(self, user_input):
        """Extract plant, disease, action"""
        user_lower = user_input.lower()
        
        intent = {'type': None, 'plant': None, 'disease': None}
        
        if any(w in user_lower for w in ['screen', 'find', 'show', 'search']):
            intent['type'] = 'screen'
        elif any(w in user_lower for w in ['tell', 'what', 'info']):
            intent['type'] = 'info'
        
        # Extract plant (use common names)
        plant_map = {
            'bitter leaf': 'Vernonia amygdalina',
            'ewuro': 'Vernonia amygdalina',
            'neem': 'Azadirachta indica',
            'moringa': 'Moringa oleifera'
        }
        for common, botanical in plant_map.items():
            if common in user_lower:
                intent['plant'] = botanical
                break
        
        # Extract disease
        for disease in ['cancer', 'malaria', 'hiv', 'diabetes', 'tuberculosis']:
            if disease in user_lower:
                intent['disease'] = disease
                break
        
        return intent
    
    def workflow_screen(self, intent):
        """Complete screening workflow"""
        plant = intent['plant']
        disease = intent['disease']
        
        if not plant or not disease:
            return "❌ Please specify both plant AND disease.\nExample: 'Screen bitter leaf for cancer'"
        
        print(f"\n{'='*80}")
        print(f"🤖 Screening {plant} for {disease}")
        print(f"{'='*80}")
        
        # 1. Retrieve
        print("\n1️⃣ Retrieving compounds...")
        compounds = self.plant_agent.search_by_plant(plant, 200)
        if compounds is None:
            return f"❌ No compounds found for {plant}"
        
        # 2. Filter
        print("\n2️⃣ Filtering...")
        filtered = compounds[
            (compounds['molecular_weight'].between(150, 550)) &
            (compounds['alogp'].between(-2, 6)) &
            (compounds['qed_drug_likeliness'] >= 0.5)
        ]
        print(f"   Passed: {len(filtered)}")
        
        # 3. Predict
        print("\n3️⃣ Predicting bioactivity...")
        results = self.predictor.predict(filtered, disease)
        if results is None:
            return f"❌ No active compounds predicted"
        
        print(f"   Active: {len(results)}")
        
        # 4. Dock (top 5)
        if self.dockers and len(results) > 0:
            print("\n4️⃣ Docking top 5...")
            # Get appropriate docker
            target_map = {
                'cancer': 'cancer_EGFR',
                'tuberculosis': 'tuberculosis_InhA',
                # Add more
            }
            target_key = target_map.get(disease)
            
            if target_key and target_key in self.dockers:
                docker = self.dockers[target_key]
                energies = []
                
                for smiles in results.head(5)['smiles']:
                    e = docker.dock_compound(smiles)
                    energies.append(e if e else 0)
                
                results.loc[results.head(5).index, 'binding_energy'] = energies
        
        # 5. Final score
        results['score'] = (
            (1 / (1 + results['ic50_predicted'])) * 10 +
            abs(results.get('binding_energy', 0)) * 5 +
            results['qed'] * 10
        )
        
        final = results.sort_values('score', ascending=False).head(10)
        
        # Save
        filename = f"{plant.replace(' ', '_')}_{disease}_leads.csv"
        final.to_csv(filename, index=False)
        
        # Format response
        response = f"""
✅ **SCREENING COMPLETE**

Found **{len(final)}** lead compounds from {plant} for {disease}

**Top 5 Candidates:**
{final[['name', 'ic50_predicted', 'binding_energy', 'score']].head(5).to_string(index=False)}

💾 Full results saved to: `{filename}`
        """
        
        return response

# ============================================================================
# TEST CHATBOT
# ============================================================================

# Initialize chatbot (assumes dockers are setup)
chatbot = ChatbotAgent(df, dockers=all_dockers if 'all_dockers' in globals() else None)

# Test queries
print(chatbot.chat("Screen bitter leaf for cancer"))
print(chatbot.chat("Find compounds from neem for diabetes"))
