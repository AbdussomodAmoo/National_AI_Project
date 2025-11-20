class FilterAgent:
    def __init__(self, df):
        self.df = df.copy()
        self.pains_smarts = self._load_pains_patterns()

    # ========================================================================
    # A. LIPINSKI'S RULE OF FIVE (Original - Enhanced)
    # ========================================================================
    def lipinski_filter(self, row):
        """Lipinski's Rule of Five for drug-likeness"""
        try:
            mw = row['molecular_weight']
            logp = row['alogp']
            hba = row['hydrogen_bond_acceptors']
            hbd = row['hydrogen_bond_donors']

            rules = [
                mw <= 500,
                logp <= 5,
                hba <= 10,
                hbd <= 5
            ]

            # Strict: all 4 rules must pass
            # Relaxed: at least 3 rules must pass
            return sum(rules) >= 3  # You can change to 4 for strict
        except:
            return False

    # ========================================================================
    # B. VEBER'S RULES (Oral Bioavailability)
    # ========================================================================
    def veber_filter(self, row):
        """Veber's rules for oral bioavailability"""
        try:
            rotatable_bonds = row['rotatable_bond_count']
            tpsa = row['topological_polar_surface_area']

            return (rotatable_bonds <= 10) and (tpsa <= 140)
        except:
            return False

    # ========================================================================
    # C. PAINS FILTERS (Pan-Assay Interference)
    # ========================================================================
    def _load_pains_patterns(self):
        """
        Load PAINS (Pan-Assay Interference Compounds) SMARTS patterns
        These are problematic substructures that give false positives in assays
        """
        # Common PAINS patterns in SMARTS notation
        pains = {
            'catechol': 'c1c(O)c(O)ccc1',
            'quinone': 'C1=CC(=O)C=CC1=O',
            'rhodanine': 'C1C(=O)NC(=S)S1',
            'hydroxyphenyl_hydrazone': 'c1ccc(O)cc1N=N',
            'phenol_sulfate': 'c1ccc(OS(=O)(=O)O)cc1',
            'michael_acceptor': 'C=CC(=O)',
            'alkyl_halide': '[CX4][Cl,Br,I]',
            'catechol_extended': '[OH]c1ccccc1[OH]',
            'azo_compounds': 'N=N',
            'nitro_aromatics': 'c[N+](=O)[O-]',
            'thiols': '[SH]',
            'peroxide': 'OO',
        }
        return pains

    def pains_filter(self, smiles):
        """
        Check if compound contains PAINS substructures
        Returns True if CLEAN (no PAINS), False if contains PAINS
        """
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Check against all PAINS patterns
            for pattern_name, smarts in self.pains_smarts.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    return False  # Contains PAINS - REJECT

            return True  # Clean - ACCEPT
        except:
            return False

    # ========================================================================
    # D. SYNTHETIC ACCESSIBILITY
    # ========================================================================
    def calculate_sa_score(self, smiles):
        """
        Calculate Synthetic Accessibility Score (1-10)
        1 = very easy to synthesize
        10 = very difficult to synthesize
        Target: SA < 6 (reasonably synthesizable)
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 10  # Penalize invalid structures

            # Simple SA score approximation based on complexity
            # Real SA score requires additional RDKit module (sascorer.py)

            # You can download sascorer.py from:
            # https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score

            # For now, use a simplified proxy:
            num_rings = Descriptors.RingCount(mol)
            num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            num_heavy_atoms = mol.GetNumHeavyAtoms()

            # Simple heuristic (not perfect, but reasonable)
            sa_score = (num_rings * 0.5) + (num_stereo * 0.8) + (num_heavy_atoms * 0.05)
            sa_score = min(sa_score, 10)  # Cap at 10

            return sa_score
        except:
            return 10

    def sa_filter(self, smiles, threshold=6):
        """SA score filter (lower is better, threshold typically 6)"""
        sa_score = self.calculate_sa_score(smiles)
        return sa_score <= threshold

    # ========================================================================
    # E. LEAD-LIKENESS (Oprea's Rules)
    # ========================================================================
    def lead_likeness_filter(self, row):
        """
        Oprea's lead-likeness rules
        More stringent than Lipinski - for compounds to be optimized
        """
        try:
            mw = row['molecular_weight']
            logp = row['alogp']

            return (200 <= mw <= 350) and (1 <= logp <= 3)
        except:
            return False

    def stricter_bounds_filter(self, row):
        """
        Additional bounds to remove extreme outliers
        that passed Lipinski but are still problematic
        """
        try:
            mw = row['molecular_weight']
            logp = row['alogp']

            # Stricter bounds
            return (
                150 <= mw <= 550 and      # Remove too small/large
                -2 <= logp <= 6           # Reasonable lipophilicity range
            )
        except:
            return False
    # ========================================================================
    # F. ADMET PRE-FILTERS
    # ========================================================================
    def predict_bbb_permeability(self, row):
        """
        Blood-Brain Barrier permeability prediction
        Simple rule-based (for complex prediction, use ML models)

        BBB+ (permeable): TPSA < 90, MW < 450, HBD < 5
        """
        try:
            tpsa = row['topological_polar_surface_area']
            mw = row['molecular_weight']
            hbd = row['hydrogen_bond_donors']

            return (tpsa < 90) and (mw < 450) and (hbd < 5)
        except:
            return False

    def predict_caco2_permeability(self, row):
        """
        Caco-2 permeability (intestinal absorption)
        Simple rule: LogP > 0 and TPSA < 140

        High permeability: LogP > 0, TPSA < 140
        """
        try:
            logp = row['alogp']
            tpsa = row['topological_polar_surface_area']

            return (logp > 0) and (tpsa < 140)
        except:
            return False

    def predict_pgp_substrate(self, smiles):
        """
        P-glycoprotein substrate prediction (efflux pump)
        Simple rule-based. For accurate prediction, use ML models.

        High MW and high TPSA often indicate Pgp substrates (not desirable)
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return True  # Penalize

            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)

            # If MW > 400 and TPSA > 120, likely Pgp substrate
            is_substrate = (mw > 400) and (tpsa > 120)
            return not is_substrate  # Return True if NOT a substrate (desirable)
        except:
            return False

    # ========================================================================
    # MASTER FILTER APPLICATION
    # ========================================================================
    def apply_filters(self,
                     filter_mode='drug_like',
                     qed_threshold=0.5,
                     np_threshold=0.3,
                     sa_threshold=6,
                     apply_pains=True,
                     apply_admet=False,
                     bbb_required=False):
        """
        Apply comprehensive filtering pipeline

        Parameters:
        -----------
        filter_mode : str
            'drug_like' - Standard drug-likeness (Lipinski + Veber + QED)
            'lead_like' - Lead-likeness (Oprea's rules)
            'strict' - All filters including PAINS and SA
            'custom' - Specify individual filters

        qed_threshold : float (0-1)
            QED drug-likeness threshold (default 0.5)

        np_threshold : float
            Natural product likeness threshold (default 0.3)

        sa_threshold : float (1-10)
            Synthetic accessibility threshold (default 6)

        apply_pains : bool
            Apply PAINS filters (default True)

        apply_admet : bool
            Apply ADMET filters (default False)

        bbb_required : bool
            Require BBB permeability (only for CNS drugs, default False)
        """

        print("\n" + "="*70)
        print("🔬 APPLYING COMPOUND FILTERS")
        print("="*70)
        print(f"Starting compounds: {len(self.df)}")
        print(f"Filter mode: {filter_mode.upper()}\n")

        # Step 1: Lipinski's Rule
        print("📋 Step 1: Lipinski's Rule of Five...")
        self.df['lipinski_pass'] = self.df.apply(self.lipinski_filter, axis=1)
        lipinski_pass = self.df['lipinski_pass'].sum()
        print(f"   ✅ Passed: {lipinski_pass}/{len(self.df)}")

        # Step 2: Veber's Rules
        print("\n📋 Step 2: Veber's Rules (Oral Bioavailability)...")
        self.df['veber_pass'] = self.df.apply(self.veber_filter, axis=1)
        veber_pass = self.df['veber_pass'].sum()
        print(f"   ✅ Passed: {veber_pass}/{len(self.df)}")

        # Step 3: QED and NP-likeness
        print("\n📋 Step 3: QED & Natural Product Likeness...")
        self.df['qed_pass'] = self.df['qed_drug_likeliness'] >= qed_threshold
        self.df['np_pass'] = self.df['np_likeness'] >= np_threshold
        qed_pass = self.df['qed_pass'].sum()
        np_pass = self.df['np_pass'].sum()
        print(f"   ✅ QED passed: {qed_pass}/{len(self.df)}")
        print(f"   ✅ NP-likeness passed: {np_pass}/{len(self.df)}")

        # Base filters for all modes
        base_filters = (
            self.df['lipinski_pass'] &
            self.df['veber_pass'] &
            self.df['qed_pass'] &
            self.df['np_pass']
        )

        # Step 4: Lead-likeness (if specified)
        if filter_mode == 'lead_like':
            print("\n📋 Step 4: Lead-likeness (Oprea's Rules)...")
            self.df['lead_like_pass'] = self.df.apply(self.lead_likeness_filter, axis=1)
            lead_pass = self.df['lead_like_pass'].sum()
            print(f"   ✅ Passed: {lead_pass}/{len(self.df)}")
            base_filters = base_filters & self.df['lead_like_pass']

        # Step 5: PAINS filters
        if apply_pains:
            print("\n📋 Step 5: PAINS Filters (removing interference compounds)...")
            self.df['pains_pass'] = self.df['canonical_smiles'].apply(self.pains_filter)
            pains_pass = self.df['pains_pass'].sum()
            pains_fail = len(self.df) - pains_pass
            print(f"   ✅ Clean compounds: {pains_pass}/{len(self.df)}")
            print(f"   ❌ PAINS detected: {pains_fail}")
            base_filters = base_filters & self.df['pains_pass']

        # Step 6: Synthetic Accessibility
        if filter_mode in ['strict', 'custom']:
            print("\n📋 Step 6: Synthetic Accessibility...")
            self.df['sa_score'] = self.df['canonical_smiles'].apply(self.calculate_sa_score)
            self.df['sa_pass'] = self.df['sa_score'] <= sa_threshold
            sa_pass = self.df['sa_pass'].sum()
            print(f"   ✅ Synthesizable (SA ≤ {sa_threshold}): {sa_pass}/{len(self.df)}")
            print(f"   📊 Average SA score: {self.df['sa_score'].mean():.2f}")
            base_filters = base_filters & self.df['sa_pass']

        # Step 7: ADMET filters
        if apply_admet:
            print("\n📋 Step 7: ADMET Predictions...")

            # Caco-2 (intestinal absorption)
            self.df['caco2_permeable'] = self.df.apply(self.predict_caco2_permeability, axis=1)
            caco2_pass = self.df['caco2_permeable'].sum()
            print(f"   ✅ Good intestinal absorption: {caco2_pass}/{len(self.df)}")

            # Pgp substrate
            self.df['not_pgp_substrate'] = self.df['canonical_smiles'].apply(self.predict_pgp_substrate)
            pgp_pass = self.df['not_pgp_substrate'].sum()
            print(f"   ✅ Not Pgp substrate: {pgp_pass}/{len(self.df)}")

            base_filters = base_filters & self.df['caco2_permeable'] & self.df['not_pgp_substrate']

            # BBB permeability (optional, only for CNS drugs)
            if bbb_required:
                self.df['bbb_permeable'] = self.df.apply(self.predict_bbb_permeability, axis=1)
                bbb_pass = self.df['bbb_permeable'].sum()
                print(f"   ✅ BBB permeable: {bbb_pass}/{len(self.df)}")
                base_filters = base_filters & self.df['bbb_permeable']
        # Step 8: Stricter Bounds (if requested)
        if filter_mode in ['strict', 'drug_like']:
            print("\n📋 Step 8: Stricter Molecular Property Bounds...")
            self.df['stricter_bounds'] = self.df.apply(self.stricter_bounds_filter, axis=1)
            bounds_pass = self.df['stricter_bounds'].sum()
            print(f"   ✅ Passed stricter bounds: {bounds_pass}/{len(self.df)}")
            print(f"   ℹ️  Removes: MW outliers (<150 or >550 Da), LogP outliers (<-2 or >6)")
            base_filters = base_filters & self.df['stricter_bounds']


        # Apply all filters
        filtered_df = self.df[base_filters].copy()

        # Summary
        print("\n" + "="*70)
        print("📊 FILTERING SUMMARY")
        print("="*70)
        print(f"✅ Compounds passed all filters: {len(filtered_df)}/{len(self.df)}")
        print(f"📉 Rejection rate: {(1 - len(filtered_df)/len(self.df))*100:.1f}%")

        if len(filtered_df) > 0:
            print(f"\n📈 Filtered Compound Statistics:")
            print(f"   • MW range: {filtered_df['molecular_weight'].min():.1f} - {filtered_df['molecular_weight'].max():.1f} Da")
            print(f"   • LogP range: {filtered_df['alogp'].min():.2f} - {filtered_df['alogp'].max():.2f}")
            print(f"   • Avg QED: {filtered_df['qed_drug_likeliness'].mean():.3f}")

        print("="*70 + "\n")

        return filtered_df

    # ========================================================================
    # CONVENIENCE METHODS FOR SPECIFIC USE CASES
    # ========================================================================
    def get_drug_like_compounds(self):
        """Quick filter for standard drug-like compounds"""
        return self.apply_filters(filter_mode='drug_like', apply_pains=True)

    def get_lead_like_compounds(self):
        """Quick filter for lead optimization"""
        return self.apply_filters(filter_mode='lead_like', apply_pains=True)

    def get_cns_drug_candidates(self):
        """Filter for CNS (brain) drug candidates"""
        return self.apply_filters(
            filter_mode='drug_like',
            apply_pains=True,
            apply_admet=True,
            bbb_required=True
        )

    def get_oral_drug_candidates(self):
        """Filter for oral bioavailability"""
        return self.apply_filters(
            filter_mode='drug_like',
            apply_pains=True,
            apply_admet=True,
            bbb_required=False
        )
