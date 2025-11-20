import pandas as pd
from collections import Counter
import requests

class PlantAgent:
    def __init__(self, df):
        self.df = df
        # Directly use the known_mappings for demonstration
        self.common_name_map = self._load_hardcoded_common_name_mapping()

    def _load_hardcoded_common_name_mapping(self):
        """Loads a hardcoded set of common name mappings for demonstration."""
        known_mappings = {
            # African Medicinal Plants (with local names) - ensure Vernonia amygdalina is here
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
                    mapping[name] = botanical # common_name (key) -> botanical_name (value)
        return mapping

    def search_by_plant(self, plant_name, top_n=10):
        # Resolve common name to botanical name
        resolved_name = self.resolve_plant_name(plant_name)

        # Search database with the RESOLVED botanical name
        print(f"🔎 Searching database for: {resolved_name}")
        results = self.df[self.df['organisms'].str.contains(resolved_name, case=False, na=False)]

        # If nothing found with resolved name, try original input as fallback
        if results.empty and resolved_name.lower() != plant_name.lower(): # Added .lower() for robust comparison
            print(f"🔄 No results for resolved name. Trying original input: {plant_name}")
            results = self.df[self.df['organisms'].str.contains(plant_name, case=False, na=False)]

        if results.empty:
            print(f"❌ No compounds found for: {plant_name}")
            return None

        print(f"✅ Found {len(results)} compounds!")
        return results.head(top_n)  # Returns ALL columns

    def list_all_organisms_tabular(self):
        """Returns organisms with occurrence counts in tabular format"""
        organism_series = self.df['organisms'].dropna()
        all_organisms = []

        for entry in organism_series:
            parts = [o.strip() for o in entry.split('|') if o.strip()]
            all_organisms.extend(parts)

        organism_counts = Counter(all_organisms)
        return pd.DataFrame(organism_counts.items(), columns=['Organism', 'Count']).sort_values(by='Count', ascending=False)


    def resolve_plant_name(self, plant_name):
        """Convert common name to botanical name using internal mapping first, then APIs"""
        # Step 1: Check internal common name mapping
        lower_plant_name = plant_name.lower()
        if lower_plant_name in self.common_name_map:
            resolved_botanical_name = self.common_name_map[lower_plant_name]
            print(f"🔍 Internal mapping: '{plant_name}' → '{resolved_botanical_name}'")
            return resolved_botanical_name

        # Try 1: GBIF Search
        try:
            url = "https://api.gbif.org/v1/species/search"
            params = {'q': plant_name, 'limit': 5}
            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    scientific_name = results[0].get('scientificName', '')
                    if scientific_name:
                        print(f"🔍 GBIF: '{plant_name}' → '{scientific_name}'")
                        return scientific_name
        except:
            pass

        # Try 2: OpenTreeOfLife API
        try:
            url = "https://api.opentreeoflife.org/v3/tnrs/match_names"
            payload = {"names": [plant_name]}
            response = requests.post(url, json=payload, timeout=5)

            if response.status_code == 200:
                results = response.json().get('results', [])
                if results and len(results) > 0:
                    matches = results[0].get('matches', [])
                    if matches:
                        scientific_name = matches[0].get('taxon', {}).get('unique_name', '')
                        if scientific_name:
                            print(f"🔍 OpenTree: '{plant_name}' → '{scientific_name}'")
                            return scientific_name
        except:
            pass

        # Try 3: Wikipedia/Wikidata (common names are well documented)
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': plant_name + ' plant',
                'limit': 5,
                'format': 'json'
            }
            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                results = response.json()
                if len(results) > 1 and results[1]:
                    # Get first result title (often the scientific name)
                    title = results[1][0]
                    print(f"🔍 Wikipedia: '{plant_name}' → '{title}'")
                    return title
        except:
            pass

        print(f"⚠️ Could not resolve '{plant_name}' via any API, using as-is")
        return plant_name
