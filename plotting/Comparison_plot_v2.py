# can run by ipynb
# cell 1 =======================================================
import awkward as ak
import json
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import vector
import glob
import pyarrow.parquet as pq

preEE_luminosity = 7.9804 * 1000
postEE_luminosity = 26.6717 * 1000
preBPix_luminosity = 18.063 * 1000
postBPix_luminosity = 9.693 * 1000

# cell 2 =============================================================
# Load cross sections
with open('/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/cross_sections_for_Ntuples_v3.json', 'r') as f:
    cross_sections = json.load(f)

# Initialize sum_genWeight_dict - will be populated during file loading
sum_genWeight_dict = {}

# Optional: Load from existing JSON files if you want to use pre-calculated values
# Uncomment below if you prefer to load from JSON instead of extracting from parquet metadata
"""
with open('/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/2022preEE_total_sum_genWeight_using_Ntuples_v3.json', 'r') as f:
    # preEE_sum_genWeight = json.load(f) # DYto2L: 1211348605824.0 + 1619411512637.0 = 2830760118461.0
    sum_genWeight_dict.update(json.load(f))
with open('/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/2022postEE_total_sum_genWeight_using_Ntuples_v3.json', 'r') as f:
    # postEE_sum_genWeight = json.load(f) # DYto2L: 2449564418560.0 + 5523156684001.5 = 7972721102561.5
    sum_genWeight_dict.update(json.load(f))
with open('/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/2023preBPix_total_sum_genWeight_using_Ntuples_v3.json', 'r') as f:
    # preBPix_sum_genWeight = json.load(f)
    sum_genWeight_dict.update(json.load(f))
with open('/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/2023postBPix_total_sum_genWeight_using_Ntuples_v3.json', 'r') as f:
    # postBPix_sum_genWeight = json.load(f)
    sum_genWeight_dict.update(json.load(f))
"""

# cell 3 =============================================================================
def get_weight(ntuple_name, ntuple):
    """
    Calculate event weights for MC samples using cross-section, luminosity, and gen weights.
    
    Args:
        ntuple_name: Name of the ntuple (e.g., "DYto2L_preEE", "QCD_postEE")
        ntuple: Awkward array containing the events
    
    Returns:
        Array of event weights or None if cross-section not found
    """
    ntuple_name_without_suffix = ntuple_name

    # Determine era and corresponding luminosity
    if "_preEE" in ntuple_name:
        ntuple_name_without_suffix = ntuple_name.replace("_preEE", "")
        luminosity = preEE_luminosity
    elif "_postEE" in ntuple_name:
        ntuple_name_without_suffix = ntuple_name.replace("_postEE", "")
        luminosity = postEE_luminosity
    elif "_preBPix" in ntuple_name:
        ntuple_name_without_suffix = ntuple_name.replace("_preBPix", "")
        luminosity = preBPix_luminosity
    elif "_postBPix" in ntuple_name:
        ntuple_name_without_suffix = ntuple_name.replace("_postBPix", "")
        luminosity = postBPix_luminosity
    else:
        luminosity = 1
    
    # Get sum of gen weights from unified dictionary
    # Use original name with era suffix as key
    if ntuple_name in sum_genWeight_dict:
        total_sum_genWeight = sum_genWeight_dict[ntuple_name]
    else:
        print(f"WARNING: {ntuple_name} not found in sum_genWeight_dict, using default value of 1")
        total_sum_genWeight = 1

    # Calculate weight if cross-section is available
    if ntuple_name_without_suffix in cross_sections:
        if "M_125" in ntuple_name_without_suffix:
            xsec = cross_sections[ntuple_name_without_suffix] * 0.0025 # 0.0025 is the branching ratio of H -> gamma gamma.
        else:
            xsec = cross_sections[ntuple_name_without_suffix]
        return luminosity * xsec * (ntuple.genWeight / total_sum_genWeight)
    else:
        return None

# cell 4 =================================================================================================
################################################################################################################################
########## Load the Ntuples of 2022 preEE and postEE data ##########
################################################################################################################################
ntuples_dict = {}
# sum_genWeight_dict is now initialized in cell 2 above - we'll populate it here
basepath = {
    "all_selected" : "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/NTuples_test01"
    #"preEE" : "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/SelectedNTuples",
    #"postEE" : "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/SelectedNTuples"
}
excluded_Ntuples = [
    "ggH", "ggH_powheg", "ttH", "VBF", "VH_bkg",
    "WH_preEE_with_Event_weights", "WH_postEE_with_Event_weights", "WH_signal_filtered", "ZH_Zto2L_signal", "ZH_Zto2Nu_signal",
    "ext1" # Exclude any postEE files if basepath is later modified
]
excluded_dir = ["merged", "root"]

for tag, path in basepath.items(): # tag: all_selected; path: /Ntuples/
    for category in os.listdir(basepath[tag]): # category: DataC_2022, ZZ_preEE
        if not any(excluded in category for excluded in excluded_Ntuples): # excluded: preEE, 2023
            category_path = os.path.join(basepath[tag], category) # category_path: /Ntuples/DataC_2022, /Ntuples/ZZ_preEE
            
            # Check if it's a parquet file
            if category.endswith('.parquet'):
                ntuple_name = category.replace("-", "_").replace(".parquet", "")
                ntuples_dict[ntuple_name] = ak.from_parquet(category_path)
                print(f"[INFO] {ntuple_name} ntuples are loaded from {category_path}.")
                
                # Extract sum_genWeight from metadata for MC samples
                if "Data" not in category:
                    try:
                        metadata = pq.read_table(category_path).schema.metadata
                        sum_genWeight_dict[ntuple_name] = float(metadata[b'sum_genw_presel'])
                        print(f"[INFO] {ntuple_name}: sum_genWeight = {sum_genWeight_dict[ntuple_name]}")
                    except KeyError:
                        print(f"[WARNING] {ntuple_name}: sum_genw_presel not found in metadata")
            
            # Check if it's a directory
            elif os.path.isdir(category_path):
                print(f"[INFO] {category_path} is loaded.")
                for process in os.listdir(category_path): # process: nominal, up, down
                    if process not in excluded_dir and os.path.isdir(os.path.join(category_path, process)) is True:
                        ntuple_path = os.path.join(category_path, process) # ntuple_path: /Ntuples/ZZ_preEE/nominal
                        ntuple_name = category.replace("-", "_") # ntuple_name: ZZ_preEE
                        
                        # Load the parquet files
                        ntuples_dict[ntuple_name] = ak.from_parquet(ntuple_path)
                        print(f"[INFO] {ntuple_name} ntuples are loaded.")
                        
                        # Extract sum_genWeight from metadata for MC samples
                        if "Data" not in category:
                            try:
                                parquet_files = glob.glob(f"{ntuple_path}/*.parquet")
                                sum_genw = 0
                                for pq_file in parquet_files:
                                    metadata = pq.read_table(pq_file).schema.metadata
                                    sum_genw += float(metadata[b'sum_genw_presel'])
                                sum_genWeight_dict[ntuple_name] = sum_genw
                                print(f"[INFO] {ntuple_name}: sum_genWeight = {sum_genw}")
                            except KeyError as e:
                                print(f"[WARNING] {ntuple_name}: sum_genw_presel not found in metadata - {e}")
                    else:
                        print(f"[INFO] {process} is excluded.")
                        pass
        else:
            print(f"[INFO] {category} is excluded.")
            pass

print("[INFO] 2022 preEE and postEE background Ntuples are loaded.")
print("\n")
print(f"[INFO] ntuples_dict: \n {ntuples_dict.keys()}")
print("\n")

# cell 5 =============================================================================
################################################################################################################################
########## Apply WH-leptonic selections on the 2022 data and MC ntuples ##########
################################################################################################################################
data_ntuples_dict = {}
data_driven_ntuples_dict = {}
bkg_MC_ntuples_dict = {}
sig_MC_ntuples_dict = {}
GJets_test_MC_ntuples = {}
QCD_test_MC_ntuples = {}

for name, ntuple in ntuples_dict.items():
    if "ZH_Hto2G_Zto2Nu_M_125" in name:
        # Use vector library for proper 4-momentum addition
        """
        electrons0 = vector.zip({"pt": ntuple["electron0_pt"], "eta": ntuple["electron0_eta"], "phi": ntuple["electron0_phi"], "mass": ntuple["electron0_mass"]})
        electrons0_mask = electrons0.pt > 0
        electrons1 = vector.zip({"pt": ntuple["electron1_pt"], "eta": ntuple["electron1_eta"], "phi": ntuple["electron1_phi"], "mass": ntuple["electron1_mass"]})
        electrons1_mask = electrons1.pt > 0
        electron_4momentum = electrons0 + electrons1
        ntuple["electron_pt"] = electron_4momentum.pt
        ntuple["electron_eta"] = electron_4momentum.eta
        ntuple["electron_phi"] = electron_4momentum.phi
        ntuple["electron_mass"] = electron_4momentum.mass
        muons0 = vector.zip({"pt": ntuple["muon0_pt"], "eta": ntuple["muon0_eta"], "phi": ntuple["muon0_phi"], "mass": ntuple["muon0_mass"]})
        muons0_mask = muons0.pt > 0
        muons1 = vector.zip({"pt": ntuple["muon1_pt"], "eta": ntuple["muon1_eta"], "phi": ntuple["muon1_phi"], "mass": ntuple["muon1_mass"]})
        muons1_mask = muons1.pt > 0
        muon_4momentum = muons0 + muons1
        ntuple["muon_pt"] = muon_4momentum.pt
        ntuple["muon_eta"] = muon_4momentum.eta
        ntuple["muon_phi"] = muon_4momentum.phi
        ntuple["muon_mass"] = muon_4momentum.mass
        leptons0 = vector.zip({"pt": ak.where(electrons0_mask, electrons0.pt, ak.where(muons0_mask, muons0.pt, 0)),
                               "eta": ak.where(electrons0_mask, electrons0.eta, ak.where(muons0_mask, muons0.eta, 0)),
                               "phi": ak.where(electrons0_mask, electrons0.phi, ak.where(muons0_mask, muons0.phi, 0)),
                               "mass": ak.where(electrons0_mask, electrons0.mass, ak.where(muons0_mask, muons0.mass, 0))})
        leptons1 = vector.zip({"pt": ak.where(electrons1_mask, electrons1.pt, ak.where(muons1_mask, muons1.pt, 0)),
                               "eta": ak.where(electrons1_mask, electrons1.eta, ak.where(muons1_mask, muons1.eta, 0)),
                               "phi": ak.where(electrons1_mask, electrons1.phi, ak.where(muons1_mask, muons1.phi, 0)),
                               "mass": ak.where(electrons1_mask, electrons1.mass, ak.where(muons1_mask, muons1.mass, 0))})
        lepton_4momentum = leptons0 + leptons1
        ntuple["leptons0_pt"] = leptons0.pt
        ntuple["leptons1_pt"] = leptons1.pt
        ntuple["leptons0_eta"] = leptons0.eta
        ntuple["leptons1_eta"] = leptons1.eta
        ntuple["leptons0_phi"] = leptons0.phi
        ntuple["leptons1_phi"] = leptons1.phi
        ntuple["leptons_mass"] = lepton_4momentum.mass
        """
        ntuple["delta_phi"] = abs(ntuple.phi - ntuple.MET_phi)
        for i in range(4): # Assuming there are at most 4 jets to consider for delta_phi calculation
            ntuple[f"delta_phi_{i}"] = abs(ntuple["MET_phi"] - ntuple[f"Jet{i}_phi"])
        ntuple["min_delta_phi"] = np.minimum(np.minimum(ntuple["delta_phi_0"], ntuple["delta_phi_1"]), np.minimum(ntuple["delta_phi_2"], ntuple["delta_phi_3"]))
        # ntuple = ntuple[(ntuple.mass > 100) & (ntuple.mass < 180)] # Diphoton mass range.
        ntuple = ntuple[(ntuple["event_category"] == 0)] # Zero-lepton cut.
        ntuple = ntuple[ntuple["MET_pt"] > 50] # MET cut
        # ntuple = ntuple[((ntuple["electron_mass"] >60) & (ntuple["electron_mass"] <120)) | ((ntuple["muon_mass"] > 60) & (ntuple["muon_mass"] < 120))] # Z mass window veto
        ntuple = ntuple[(ntuple.lead_mvaID > -0.9) & (ntuple.sublead_mvaID > -0.9)] # Photon id cut.
        ntuple = ntuple[ntuple["delta_phi"] > 2.0] # MET cut
        # ntuple = ntuple[ntuple.muon0_pt > 15] # Additional muon pt cut for WH-leptonic category.
        sig_MC_ntuples_dict[name] = ntuple
    else:
        # Use vector library for proper 4-momentum addition
        """
        electrons0 = vector.zip({"pt": ntuple["electron0_pt"], "eta": ntuple["electron0_eta"], "phi": ntuple["electron0_phi"], "mass": ntuple["electron0_mass"]})
        electrons0_mask = electrons0.pt > 0
        electrons1 = vector.zip({"pt": ntuple["electron1_pt"], "eta": ntuple["electron1_eta"], "phi": ntuple["electron1_phi"], "mass": ntuple["electron1_mass"]})
        electrons1_mask = electrons1.pt > 0
        electron_4momentum = electrons0 + electrons1
        ntuple["electron_pt"] = electron_4momentum.pt
        ntuple["electron_eta"] = electron_4momentum.eta
        ntuple["electron_phi"] = electron_4momentum.phi
        ntuple["electron_mass"] = electron_4momentum.mass
        muons0 = vector.zip({"pt": ntuple["muon0_pt"], "eta": ntuple["muon0_eta"], "phi": ntuple["muon0_phi"], "mass": ntuple["muon0_mass"]})
        muons0_mask = muons0.pt > 0
        muons1 = vector.zip({"pt": ntuple["muon1_pt"], "eta": ntuple["muon1_eta"], "phi": ntuple["muon1_phi"], "mass": ntuple["muon1_mass"]})
        muons1_mask = muons1.pt > 0
        muon_4momentum = muons0 + muons1
        ntuple["muon_pt"] = muon_4momentum.pt
        ntuple["muon_eta"] = muon_4momentum.eta
        ntuple["muon_phi"] = muon_4momentum.phi
        ntuple["muon_mass"] = muon_4momentum.mass
        leptons0 = vector.zip({"pt": ak.where(electrons0_mask, electrons0.pt, ak.where(muons0_mask, muons0.pt, 0)),
                               "eta": ak.where(electrons0_mask, electrons0.eta, ak.where(muons0_mask, muons0.eta, 0)),
                               "phi": ak.where(electrons0_mask, electrons0.phi, ak.where(muons0_mask, muons0.phi, 0)),
                               "mass": ak.where(electrons0_mask, electrons0.mass, ak.where(muons0_mask, muons0.mass, 0))})
        leptons1 = vector.zip({"pt": ak.where(electrons1_mask, electrons1.pt, ak.where(muons1_mask, muons1.pt, 0)),
                               "eta": ak.where(electrons1_mask, electrons1.eta, ak.where(muons1_mask, muons1.eta, 0)),
                               "phi": ak.where(electrons1_mask, electrons1.phi, ak.where(muons1_mask, muons1.phi, 0)),
                               "mass": ak.where(electrons1_mask, electrons1.mass, ak.where(muons1_mask, muons1.mass, 0))})
        lepton_4momentum = leptons0 + leptons1
        ntuple["leptons0_pt"] = leptons0.pt
        ntuple["leptons1_pt"] = leptons1.pt
        ntuple["leptons0_eta"] = leptons0.eta
        ntuple["leptons1_eta"] = leptons1.eta
        ntuple["leptons0_phi"] = leptons0.phi
        ntuple["leptons1_phi"] = leptons1.phi
        ntuple["leptons_mass"] = lepton_4momentum.mass
        """
        ntuple["delta_phi"] = abs(ntuple.phi - ntuple.MET_phi)
        for i in range(4): # Assuming there are at most 4 jets to consider for delta_phi calculation
            ntuple[f"delta_phi_{i}"] = abs(ntuple["MET_phi"] - ntuple[f"Jet{i}_phi"])
        ntuple["min_delta_phi"] = np.minimum(np.minimum(ntuple["delta_phi_0"], ntuple["delta_phi_1"]), np.minimum(ntuple["delta_phi_2"], ntuple["delta_phi_3"]))
        # ntuple = ntuple[(ntuple.mass > 100) & (ntuple.mass < 180)] # Diphoton mass range.
        ntuple = ntuple[(ntuple.mass < 115) | (ntuple.mass > 135)] # Blind the Higgs signal region except VH_MET_signal.
        ntuple = ntuple[(ntuple["event_category"] == 0)] # Zero-lepton cut.
        ntuple = ntuple[ntuple["MET_pt"] > 50] # MET cut
        ntuple = ntuple[ntuple["delta_phi"] > 2.0] # MET cut
        # ntuple = ntuple[((ntuple["electron_mass"] >60) & (ntuple["electron_mass"] <120)) | ((ntuple["muon_mass"] > 60) & (ntuple["muon_mass"] < 120))] # Z mass window vet
        ntuple = ntuple[(ntuple.lead_mvaID > -0.7) & (ntuple.sublead_mvaID > -0.7)] # Photon id cut.
        # ntuple = ntuple[ntuple.muon0_pt > 15] # Additional muon pt cut for WH-leptonic category.
        if "Data" in name:
            data_ntuples_dict[name] = ntuple
        else:
            bkg_MC_ntuples_dict[name] = ntuple

print("===========================================================================")
print("[INFO] data_ntuples_dict:\n", data_ntuples_dict.keys())
print("===========================================================================")
print("[INFO] signal_MC_ntuples_dict:\n", sig_MC_ntuples_dict.keys())
print("===========================================================================")
print("[INFO] background_MC_ntuples_dict:\n", bkg_MC_ntuples_dict.keys())
print("===========================================================================")
print("[INFO] sum_genWeight_dict:\n", sum_genWeight_dict.keys())
print("\n")

# cell 6 =============================================================================
################################################################################################################################
########## Categorize each background sample based on processes involved ##########
################################################################################################################################
Diphotons_ntuples_dict = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if "GG_Box_3Jets" in name}
DYJets_ntuples_dict    = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if "DYto2L_2Jets" in name}
GJets_ntuples_dict     = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if "GJet" in name}
QCD_ntuples_dict       = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if "QCD" in name}
Top_ntuples_dict       = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if ("TGJets" in name or "TTG_1Jets" in name or "TTGG_0Jets" in name or "TTto" in name)}
Diboson_ntuples_dict   = {name : ntuple for name, ntuple in bkg_MC_ntuples_dict.items() if ("WW" in name or "WZ" in name or "ZZ" in name or "WGtoLNuG" in name or "ZGto" in name or "DYGto2LG_1Jets" in name)}
ZH_signal_ntuples_dict = {name : ntuple for name, ntuple in sig_MC_ntuples_dict.items()}
print("===========================================================================")
print("[INFO] Categorization of background samples completed.")
print("===========================================================================")
print("\n")

# cell 7 =============================================================================
################################################################################################################################
########## Concatenate the MC weights of the same process ##########
################################################################################################################################
Diphotons_weight = np.concatenate([get_weight(name, nt) for name, nt in Diphotons_ntuples_dict.items()])
DYJets_weight    = np.concatenate([get_weight(name, nt) for name, nt in DYJets_ntuples_dict.items()])
GJets_weight     = np.concatenate([get_weight(name, nt) for name, nt in GJets_ntuples_dict.items()])
QCD_weight       = np.concatenate([get_weight(name, nt) for name, nt in QCD_ntuples_dict.items()]) 
Top_weight       = np.concatenate([get_weight(name, nt) for name, nt in Top_ntuples_dict.items()])
Diboson_weight   = np.concatenate([get_weight(name, nt) for name, nt in Diboson_ntuples_dict.items()])
ZH_signal_weight = np.concatenate([get_weight(name, nt) for name, nt in ZH_signal_ntuples_dict.items()])
print("===========================================================================")
print("[INFO] MC weights concatenation completed.")
print("===========================================================================")
print("\n")

# cell 8 =============================================================================
################################################################################################################################
########## Max and Min gamma mvaID ##########
################################################################################################################################
Data_max_gamma_ID      = ak.to_numpy(np.concatenate([np.maximum(data_ntuples_dict[name].lead_mvaID, data_ntuples_dict[name].sublead_mvaID) for name in data_ntuples_dict.keys()]))
Data_min_gamma_ID      = ak.to_numpy(np.concatenate([np.minimum(data_ntuples_dict[name].lead_mvaID, data_ntuples_dict[name].sublead_mvaID) for name in data_ntuples_dict.keys()]))

Diphotons_max_gamma_ID = ak.to_numpy(np.concatenate([np.maximum(Diphotons_ntuples_dict[name].lead_mvaID, Diphotons_ntuples_dict[name].sublead_mvaID) for name in Diphotons_ntuples_dict.keys()]))
Diphotons_min_gamma_ID = ak.to_numpy(np.concatenate([np.minimum(Diphotons_ntuples_dict[name].lead_mvaID, Diphotons_ntuples_dict[name].sublead_mvaID) for name in Diphotons_ntuples_dict.keys()]))

DYJets_max_gamma_ID    = ak.to_numpy(np.concatenate([np.maximum(DYJets_ntuples_dict[name].lead_mvaID, DYJets_ntuples_dict[name].sublead_mvaID) for name in DYJets_ntuples_dict.keys()]))
DYJets_min_gamma_ID    = ak.to_numpy(np.concatenate([np.minimum(DYJets_ntuples_dict[name].lead_mvaID, DYJets_ntuples_dict[name].sublead_mvaID) for name in DYJets_ntuples_dict.keys()]))

GJets_max_gamma_ID     = ak.to_numpy(np.concatenate([np.maximum(GJets_ntuples_dict[name].lead_mvaID, GJets_ntuples_dict[name].sublead_mvaID) for name in GJets_ntuples_dict.keys()]))
GJets_min_gamma_ID     = ak.to_numpy(np.concatenate([np.minimum(GJets_ntuples_dict[name].lead_mvaID, GJets_ntuples_dict[name].sublead_mvaID) for name in GJets_ntuples_dict.keys()]))

QCD_max_gamma_ID       = ak.to_numpy(np.concatenate([np.maximum(QCD_ntuples_dict[name].lead_mvaID, QCD_ntuples_dict[name].sublead_mvaID) for name in QCD_ntuples_dict.keys()]))
QCD_min_gamma_ID       = ak.to_numpy(np.concatenate([np.minimum(QCD_ntuples_dict[name].lead_mvaID, QCD_ntuples_dict[name].sublead_mvaID) for name in QCD_ntuples_dict.keys()]))

Top_max_gamma_ID       = ak.to_numpy(np.concatenate([np.maximum(Top_ntuples_dict[name].lead_mvaID, Top_ntuples_dict[name].sublead_mvaID) for name in Top_ntuples_dict.keys()]))
Top_min_gamma_ID       = ak.to_numpy(np.concatenate([np.minimum(Top_ntuples_dict[name].lead_mvaID, Top_ntuples_dict[name].sublead_mvaID) for name in Top_ntuples_dict.keys()]))

Diboson_max_gamma_ID   = ak.to_numpy(np.concatenate([np.maximum(Diboson_ntuples_dict[name].lead_mvaID, Diboson_ntuples_dict[name].sublead_mvaID) for name in Diboson_ntuples_dict.keys()]))
Diboson_min_gamma_ID   = ak.to_numpy(np.concatenate([np.minimum(Diboson_ntuples_dict[name].lead_mvaID, Diboson_ntuples_dict[name].sublead_mvaID) for name in Diboson_ntuples_dict.keys()]))

ZH_signal_max_gamma_ID = ak.to_numpy(np.concatenate([np.maximum(ZH_signal_ntuples_dict[name].lead_mvaID, ZH_signal_ntuples_dict[name].sublead_mvaID) for name in ZH_signal_ntuples_dict.keys()]))
ZH_signal_min_gamma_ID = ak.to_numpy(np.concatenate([np.minimum(ZH_signal_ntuples_dict[name].lead_mvaID, ZH_signal_ntuples_dict[name].sublead_mvaID) for name in ZH_signal_ntuples_dict.keys()]))
print("===========================================================================")
print("[INFO] Max and Min gamma mvaID calculation completed.")
print("===========================================================================")

# cell 9 =============================================================================
################################################################################################################################
########## Labels for each MC sample ##########
################################################################################################################################
labels = ["Diphoton", r"$VV/V+\it{\gamma}$", "DYJets", r"$\it{\gamma}+jets$", "QCD", "Top"]
################################################################################################################################
########## Plotting ##########
################################################################################################################################
hep.style.use("CMS")
suffix = "20250621"

# cell 10 ==============================================================================
###############################################################################################
########## Diphoton lead_pt/mass (With data/MC ratio plot & data in errorbar style.) ##########
###############################################################################################
Data_lead_pt_mass = ak.to_numpy(np.concatenate([(data_ntuples_dict[name].lead_pt / data_ntuples_dict[name].mass) for name in data_ntuples_dict.keys()]))
Data_lead_pt_mass_hist, bins = np.histogram(Data_lead_pt_mass, range = (0.4, 1.2), bins = 30)
bin_center = (bins[1:] + bins[:-1]) / 2

Diphotons_lead_pt_mass = np.concatenate([(Diphotons_ntuples_dict[name].lead_pt / Diphotons_ntuples_dict[name].mass) for name in Diphotons_ntuples_dict.keys()])
DYJets_lead_pt_mass    = np.concatenate([(DYJets_ntuples_dict[name].lead_pt / DYJets_ntuples_dict[name].mass) for name in DYJets_ntuples_dict.keys()])
GJets_lead_pt_mass     = np.concatenate([(GJets_ntuples_dict[name].lead_pt / GJets_ntuples_dict[name].mass) for name in GJets_ntuples_dict.keys()])
QCD_lead_pt_mass       = np.concatenate([(QCD_ntuples_dict[name].lead_pt / QCD_ntuples_dict[name].mass) for name in QCD_ntuples_dict.keys()])
Top_lead_pt_mass       = np.concatenate([(Top_ntuples_dict[name].lead_pt / Top_ntuples_dict[name].mass) for name in Top_ntuples_dict.keys()])
Diboson_lead_pt_mass   = np.concatenate([(Diboson_ntuples_dict[name].lead_pt / Diboson_ntuples_dict[name].mass) for name in Diboson_ntuples_dict.keys()])

ZH_signal_lead_pt_mass = np.concatenate([(ZH_signal_ntuples_dict[name].lead_pt / ZH_signal_ntuples_dict[name].mass) for name in ZH_signal_ntuples_dict.keys()])

MC_samples = [Diphotons_lead_pt_mass, Diboson_lead_pt_mass, DYJets_lead_pt_mass, GJets_lead_pt_mass, QCD_lead_pt_mass, Top_lead_pt_mass]
MC_weights = [Diphotons_weight, Diboson_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight]

MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

# MC uncertainty
MC_sumw2, _ = np.histogram(
    np.concatenate(MC_samples),
    bins=bins,
    weights=np.concatenate(MC_weights)**2
)
mc_stat_error = np.sqrt(MC_sumw2)

ZH_signal_hist, _ = np.histogram(
    ZH_signal_lead_pt_mass,
    bins = bins,
    weights = ZH_signal_weight
)
fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex = True, figsize = (10, 12))

# Upper plot: Data and MC comparison
# When you measure the number of events N, each event is independent and has a constant probability per unit time or per measurement. 
# Under these conditions, the probability distribution that describes the observed number of events is the Poisson distribution.
# Since the mean equals to variance in Poisson distribution, the statistical uncertainty is given by the square root of N.
axs[0].errorbar(
    x = bin_center, y = Data_lead_pt_mass_hist,
    yerr = np.sqrt(Data_lead_pt_mass_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#ffd700", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff"]
)
axs[0].hist(
    ZH_signal_lead_pt_mass, bins = bins, weights = ZH_signal_weight * 200,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$Z(\\to \\ell\\ell)H(\\to \\gamma\\gamma) \\times 200$"
)
axs[0].set_xlim((0.4, 1.2))
# axs[0].set_yscale("log")
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 62.456, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend()

# Lower plot: Data/MC ratio
# Avoid division by zero
# Convert MC_hist to numpy array if it's an awkward array
MC_hist_np = ak.to_numpy(MC_hist) if hasattr(MC_hist, '__array_ufunc__') else MC_hist
# MC uncertainty
axs[0].bar(
    bin_center, 2 * mc_stat_error,
    bottom = MC_hist_np - mc_stat_error,
    width = np.diff(bins),
    color = 'gray', alpha = 0.4, hatch = '///', linewidth = 0,
    label = 'MC stat. unc.'
)
ratio = np.divide(
    Data_lead_pt_mass_hist, MC_hist_np,
    out = np.zeros_like(Data_lead_pt_mass_hist, dtype=float),
    where = MC_hist_np != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(np.maximum(Data_lead_pt_mass_hist, 0))
# mc_errors = np.sqrt(np.maximum(MC_hist_np, 0))
# MC uncertainty
mc_rel_err = np.where(MC_hist_np > 0, mc_stat_error / MC_hist_np, 0)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_lead_pt_mass_hist > 0) & (MC_hist_np > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_lead_pt_mass_hist[nonzero_mask])**2 +
    (mc_stat_error[nonzero_mask] / MC_hist_np[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_lead_pt_mass_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_lead_pt_mass_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
# MC uncertainty
axs[1].bar(
    bin_center, 2 * mc_rel_err,
    bottom = 1 - mc_rel_err,
    width = np.diff(bins),
    color = 'gray', alpha = 0.4, hatch = '///', linewidth = 0
)
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$p^{lead \ \gamma}_{T} / m_{\gamma\gamma}$")

# Save the plot
# plt.savefig(f"lead_photon_pt_to_mass_ratio_{suffix}.png")

plt.show()

# cell 11 ==============================================================================
######################################################################################
########## Max gamma ID (With data/MC ratio plot & data in errorbar style.) ##########
######################################################################################
Data_max_gamma_ID_hist, bins = np.histogram(Data_max_gamma_ID, range = (-0.4, 1), bins = 30)
bin_center = (bins[1:] + bins[:-1]) / 2

MC_samples = [Diphotons_max_gamma_ID, Diboson_max_gamma_ID, DYJets_max_gamma_ID, GJets_max_gamma_ID, QCD_max_gamma_ID, Top_max_gamma_ID]
MC_weights = [Diphotons_weight, Diboson_weight, DYJets_weight, GJets_weight, QCD_weight, Top_weight]

MC_hist, _ = np.histogram(
    np.concatenate(MC_samples),
    bins = bins,
    weights = np.concatenate(MC_weights)
)

MC_sumw2, _ = np.histogram(
    np.concatenate(MC_samples),
    bins=bins,
    weights=np.concatenate(MC_weights)**2
)
mc_stat_error = np.sqrt(MC_sumw2)

ZH_signal_hist, _ = np.histogram(
    ZH_signal_max_gamma_ID,
    bins = bins,
    weights = ZH_signal_weight
)

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex = True, figsize = (10, 12))

# Upper plot: Data and MC comparison
axs[0].errorbar(
    x = bin_center, y = Data_max_gamma_ID_hist,
    yerr = np.sqrt(Data_max_gamma_ID_hist), fmt = "ko", label = "Data"
)
axs[0].hist(
    MC_samples, bins = bins, weights = MC_weights,
    histtype = "stepfilled", stacked = True,
    label = labels, color = ["#0000ff", "#ffd700", "#7b68ee", "#7cfc00", "#ffa500", "#00bfff"]
)
axs[0].hist(
    ZH_signal_max_gamma_ID, bins = bins, weights = ZH_signal_weight * 20,
    histtype = "step", linestyle = "-", linewidth = 3, color = "red", label = "$Z(\\to \\ell\\ell)H(\\to \\gamma\\gamma) \\times 20$"
)
axs[0].set_xlim((-0.4, 1))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 62.456, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(loc = "upper left", ncol = 2)

# Lower plot: Data/MC ratio
# Avoid division by zero
# Convert MC_hist to numpy array if it's an awkward array
MC_hist_np = ak.to_numpy(MC_hist) if hasattr(MC_hist, '__array_ufunc__') else MC_hist
# MC uncertainty
axs[0].bar(
    bin_center, 2 * mc_stat_error,
    bottom = MC_hist_np - mc_stat_error,
    width = np.diff(bins),
    color = 'gray', alpha = 0.4, hatch = '///', linewidth = 0,
    label = 'MC stat. unc.'
)
ratio = np.divide(
    Data_max_gamma_ID_hist, MC_hist_np,
    out = np.zeros_like(Data_max_gamma_ID_hist, dtype=float),
    where = MC_hist_np != 0
)

# Calculate errors for ratio
data_errors = np.sqrt(np.maximum(Data_max_gamma_ID_hist, 0))
# mc_errors = np.sqrt(np.maximum(MC_hist_np, 0))
# MC uncertainty
mc_rel_err = np.where(MC_hist_np > 0, mc_stat_error / MC_hist_np, 0)
ratio_errors = np.zeros_like(ratio)
nonzero_mask = (Data_max_gamma_ID_hist > 0) & (MC_hist_np > 0)
ratio_errors[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
    (data_errors[nonzero_mask] / Data_max_gamma_ID_hist[nonzero_mask])**2 +
    (mc_stat_error[nonzero_mask] / MC_hist_np[nonzero_mask])**2
)

# data_errors = np.sqrt(Data_max_gamma_ID_hist)
# mc_errors = np.sqrt(MC_hist)
# ratio_errors = ratio * np.sqrt(
#     (data_errors / Data_max_gamma_ID_hist)**2 + (mc_errors / MC_hist)**2
# )

axs[1].errorbar(x = bin_center, y = ratio, yerr = ratio_errors, fmt = "ko")
# MC uncertainty
axs[1].bar(
    bin_center, 2 * mc_rel_err,
    bottom = 1 - mc_rel_err,
    width = np.diff(bins),
    color = 'gray', alpha = 0.4, hatch = '///', linewidth = 0
)
axs[1].axhline(1, color = "red", linestyle = "--", linewidth = 1)  # Reference line
axs[1].set_ylim(0, 2.0)
axs[1].set_ylabel("Data/MC", loc = "center")
axs[1].set_xlabel("$Max {\ } {\gamma} {\ } ID$")

# Save the plot
# plt.savefig(f"max_gamma_ID_{suffix}.png")
plt.show()

# cell 12 ==============================================================================
# can plot other kinametics or variables ....
