####################################################################
### can first test all of the procedure of this program in ipynb ###
####################################################################
import awkward as ak
import json
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import vector
import glob
import pyarrow as pa
import pyarrow.parquet as pq

from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from scipy.stats import norm, truncnorm

preEE_luminosity = 7.9804 * 1000
postEE_luminosity = 26.6717 * 1000
preBPix_luminosity = 18.063 * 1000
postBPix_luminosity = 9.693 * 1000

"""load cross section"""
# Load cross sections
with open('/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/cross_sections_for_Ntuples_v3.json', 'r') as f:
    cross_sections = json.load(f)

# Initialize sum_genWeight_dict - will be populated during file loading
sum_genWeight_dict = {}

"""function of get weight"""
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

################################################################################################################################
########## Load the Ntuples of 2022, 2023 data ##########
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

print("[INFO] 2022, 2023 background Ntuples are loaded.")
print("\n")
print(f"[INFO] ntuples_dict: \n {ntuples_dict.keys()}")
print("\n")

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
        if "GJet" in name:
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
            GJets_test_MC_ntuples[name] = ntuple
        if "QCD" in name:
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
            QCD_test_MC_ntuples[name] = ntuple
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
        # ntuple = ntuple[((ntuple["electron_mass"] >60) & (ntuple["electron_mass"] <120)) | ((ntuple["muon_mass"] > 60) & (ntuple["muon_mass"] < 120))] # Z mass window veto
        if "Data" in name:
            ntuple = ntuple[(ntuple.lead_mvaID > -0.9) & (ntuple.sublead_mvaID > -0.9)] # Photon id cut for data-driven background estimation.
            data_driven_ntuples_dict[name] = ntuple # For data-driven background estimation
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
print("[INFO] GJets_test_MC_ntuples:\n", GJets_test_MC_ntuples.keys())
print("===========================================================================")
print("[INFO] QCD_test_MC_ntuples:\n", QCD_test_MC_ntuples.keys())
print("===========================================================================")
print("[INFO] sum_genWeight_dict:\n", sum_genWeight_dict.keys())
print("\n")

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

################################################################################################################################
########## Concatenate the MC weights of the same process ##########
################################################################################################################################
Diphotons_weight = np.concatenate([get_weight(name, nt) for name, nt in Diphotons_ntuples_dict.items()])
DYJets_weight    = np.concatenate([get_weight(name, nt) for name, nt in DYJets_ntuples_dict.items()])
GJets_weight     = np.concatenate([get_weight(name, nt) for name, nt in GJets_ntuples_dict.items()])
GJets_test_weight = np.concatenate([get_weight(name, nt) for name, nt in GJets_test_MC_ntuples.items()])
QCD_weight       = np.concatenate([get_weight(name, nt) for name, nt in QCD_ntuples_dict.items()])
QCD_test_weight  = np.concatenate([get_weight(name, nt) for name, nt in QCD_test_MC_ntuples.items()])   
Top_weight       = np.concatenate([get_weight(name, nt) for name, nt in Top_ntuples_dict.items()])
Diboson_weight   = np.concatenate([get_weight(name, nt) for name, nt in Diboson_ntuples_dict.items()])
ZH_signal_weight = np.concatenate([get_weight(name, nt) for name, nt in ZH_signal_ntuples_dict.items()])
print("===========================================================================")
print("[INFO] MC weights concatenation completed.")
print("===========================================================================")
print("\n")

################################################################################################################################
########## Max and Min gamma mvaID ##########
################################################################################################################################
Data_max_gamma_ID      = ak.to_numpy(np.concatenate([np.maximum(data_ntuples_dict[name].lead_mvaID, data_ntuples_dict[name].sublead_mvaID) for name in data_ntuples_dict.keys()]))
Data_min_gamma_ID      = ak.to_numpy(np.concatenate([np.minimum(data_ntuples_dict[name].lead_mvaID, data_ntuples_dict[name].sublead_mvaID) for name in data_ntuples_dict.keys()]))

Data_driven_max_gamma_ID = ak.to_numpy(np.concatenate([np.maximum(data_driven_ntuples_dict[name].lead_mvaID, data_driven_ntuples_dict[name].sublead_mvaID) for name in data_driven_ntuples_dict.keys()]))
Data_driven_min_gamma_ID = ak.to_numpy(np.concatenate([np.minimum(data_driven_ntuples_dict[name].lead_mvaID, data_driven_ntuples_dict[name].sublead_mvaID) for name in data_driven_ntuples_dict.keys()]))

Diphotons_max_gamma_ID = ak.to_numpy(np.concatenate([np.maximum(Diphotons_ntuples_dict[name].lead_mvaID, Diphotons_ntuples_dict[name].sublead_mvaID) for name in Diphotons_ntuples_dict.keys()]))
Diphotons_min_gamma_ID = ak.to_numpy(np.concatenate([np.minimum(Diphotons_ntuples_dict[name].lead_mvaID, Diphotons_ntuples_dict[name].sublead_mvaID) for name in Diphotons_ntuples_dict.keys()]))

DYJets_max_gamma_ID    = ak.to_numpy(np.concatenate([np.maximum(DYJets_ntuples_dict[name].lead_mvaID, DYJets_ntuples_dict[name].sublead_mvaID) for name in DYJets_ntuples_dict.keys()]))
DYJets_min_gamma_ID    = ak.to_numpy(np.concatenate([np.minimum(DYJets_ntuples_dict[name].lead_mvaID, DYJets_ntuples_dict[name].sublead_mvaID) for name in DYJets_ntuples_dict.keys()]))

GJets_max_gamma_ID     = ak.to_numpy(np.concatenate([np.maximum(GJets_ntuples_dict[name].lead_mvaID, GJets_ntuples_dict[name].sublead_mvaID) for name in GJets_ntuples_dict.keys()]))
GJets_min_gamma_ID     = ak.to_numpy(np.concatenate([np.minimum(GJets_ntuples_dict[name].lead_mvaID, GJets_ntuples_dict[name].sublead_mvaID) for name in GJets_ntuples_dict.keys()]))

GJets_test_max_gamma_ID     = ak.to_numpy(np.concatenate([np.maximum(GJets_test_MC_ntuples[name].lead_mvaID, GJets_test_MC_ntuples[name].sublead_mvaID) for name in GJets_test_MC_ntuples.keys()]))
GJets_test_min_gamma_ID     = ak.to_numpy(np.concatenate([np.minimum(GJets_test_MC_ntuples[name].lead_mvaID, GJets_test_MC_ntuples[name].sublead_mvaID) for name in GJets_test_MC_ntuples.keys()]))

QCD_test_max_gamma_ID       = ak.to_numpy(np.concatenate([np.maximum(QCD_test_MC_ntuples[name].lead_mvaID, QCD_test_MC_ntuples[name].sublead_mvaID) for name in QCD_test_MC_ntuples.keys()]))
QCD_test_min_gamma_ID       = ak.to_numpy(np.concatenate([np.minimum(QCD_test_MC_ntuples[name].lead_mvaID, QCD_test_MC_ntuples[name].sublead_mvaID) for name in QCD_test_MC_ntuples.keys()]))

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

################################################################################################################################
########## Labels for each MC sample ##########
################################################################################################################################
labels = ["Diphoton", r"$VV/V+\it{\gamma}$", "DYJets", r"$\it{\gamma}+jets$", "QCD", "Top"]
################################################################################################################################
########## Plotting ##########
################################################################################################################################
hep.style.use("CMS")
suffix = "20250621"

"""extract data from sideband region"""
# Extract sideband region from data
data_driven_ntuples = np.concatenate([ntuple for name, ntuple in data_driven_ntuples_dict.items()])
print("number of events in data_driven_ntuples before photon ID cut:", len(data_driven_ntuples))
print("pt of leading photon in data_driven_ntuples before photon ID cut:", data_driven_ntuples.lead_pt)
data_driven_ntuples = data_driven_ntuples[(Data_driven_max_gamma_ID > -0.7) & ((Data_driven_min_gamma_ID > -0.9) & (Data_driven_min_gamma_ID < -0.7))] # Apply photon ID cut to data-driven ntuples for fair comparison with MC.
print("number of events in data_driven_ntuples after photon ID cut:", len(data_driven_ntuples))
print("pt of leading photon in data_driven_ntuples after photon ID cut:", data_driven_ntuples.lead_pt)

"""check correlation between photon id and kinematic"""
import pandas as pd
# Create samples
GJets_lead_pt_mass = np.concatenate([(GJets_test_MC_ntuples[name].lead_pt / GJets_test_MC_ntuples[name].mass) for name in GJets_test_MC_ntuples.keys()])
GJets_sublead_pt_mass = np.concatenate([(GJets_test_MC_ntuples[name].sublead_pt / GJets_test_MC_ntuples[name].mass) for name in GJets_test_MC_ntuples.keys()])
# Create a DataFrame with variables of interest
df = pd.DataFrame({
    'max_mvaID': ak.to_numpy(GJets_test_max_gamma_ID),
    'min_mvaID': ak.to_numpy(GJets_test_min_gamma_ID),
    'lead_pt_mass': ak.to_numpy(GJets_lead_pt_mass),
    'sublead_pt_mass': ak.to_numpy(GJets_sublead_pt_mass)
})

# Calculate correlation matrix
correlation_matrix = df.corr(method='pearson')
print(correlation_matrix)

# Access specific correlations
print(f"\nmax_mvaID vs lead_pt_mass: {correlation_matrix.loc['max_mvaID', 'lead_pt_mass']:.4f}")

# Plot heatmap using matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(len(correlation_matrix.columns)))
ax.set_yticks(np.arange(len(correlation_matrix.columns)))
ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(correlation_matrix.columns)

# Add correlation values as text
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Pearson Correlation', rotation=270, labelpad=20)

ax.set_title('Correlation Matrix: Photon ID vs Kinematics')
plt.tight_layout()
plt.savefig(f"correlation_matrix_{suffix}.png")

"""present distribution of minID and apply gen-matching to only pick up fake photon"""
GJets_min_genPartFlav = ak.to_numpy(np.concatenate([
    np.where(
        ak.to_numpy(nt.lead_mvaID) >= ak.to_numpy(nt.sublead_mvaID),
        ak.to_numpy(nt.sublead_genPartFlav),
        ak.to_numpy(nt.lead_genPartFlav)
    )
    for name, nt in GJets_test_MC_ntuples.items()
]))

fake_mask = GJets_min_genPartFlav == 0

bins = np.linspace(-0.9, 1, 30)
bin_center = (bins[1:] + bins[:-1]) / 2

GJets_fake_sumWeight = np.sum(GJets_test_weight[fake_mask]) # Use sum of weights for events minID photon is fake
GJets_fake_hist, _ = np.histogram(
    GJets_test_min_gamma_ID[fake_mask],
    bins = bins,
    weights = GJets_test_weight[fake_mask] / GJets_fake_sumWeight
)

# Calculate histogram WITHOUT normalization first
GJets_fake_hist_raw, _ = np.histogram(
    GJets_test_min_gamma_ID[fake_mask],
    bins=bins,
    weights=GJets_test_weight[fake_mask]
)

# Calculate sum of weights squared in each bin (for error calculation)
GJets_fake_hist_weights_sq, _ = np.histogram(
    GJets_test_min_gamma_ID[fake_mask],
    bins=bins,
    weights=GJets_test_weight[fake_mask]**2
)

# Normalization factor
GJets_fake_sumWeight = np.sum(GJets_test_weight[fake_mask])

# Normalize
GJets_fake_hist_normalized = GJets_fake_hist_raw / GJets_fake_sumWeight

# Correct error after normalization: sqrt(sum(w_i^2)) / sum(w_i)
GJets_fake_hist_error = np.sqrt(GJets_fake_hist_weights_sq) / GJets_fake_sumWeight

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 1], "hspace" : 0.1}, sharex=True, figsize=(10, 12))

axs[0].errorbar(
    x = bin_center, y = GJets_fake_hist_normalized,
    yerr = GJets_fake_hist_error, fmt = "ko", color = "black", label = "GJets_MC"
)

axs[0].set_xlim((-0.9, 1))
axs[0].set_ylim(bottom = 0)
axs[0].set_ylabel("Events", loc = "top")
hep.cms.label(loc = 0, data = True, label = "Preliminary", lumi = 62.456, lumi_format = "{0:.1f}", com = 13.6, ax = axs[0])
axs[0].legend(loc = "upper left", ncol = 2)

plt.savefig(f"minID_distribution_raw_{suffix}.png")

"""fit the minID distribution with 7-order Bernstein polynomial"""
from iminuit import Minuit
from scipy.special import comb as scipy_comb
from scipy.stats import beta as beta_dist
import numpy as np
import iminuit
print(f"iminuit version: {iminuit.__version__}")

X_MIN, X_MAX = -0.9, 1.0
DEGREE = 7  # increase to 5 or 6 if fit quality is poor

def bernstein_pdf(x, lc1, lc2, lc3, lc4, lc5, lc6=0., lc7=0.):
    """
    Degree-7 Bernstein polynomial PDF on [X_MIN, X_MAX].
    lc0 is FIXED to 0 (reference coefficient = 1).
    The other lc_k are relative log-ratios — this removes the exact flat
    direction that caused 'Covariance FORCED pos. def.' when all 5 were free.
    Analytically normalised: integral = sum(c_k)/(n+1) * (X_MAX - X_MIN)
    """
    # lc0 = 0 fixed → c0 = exp(0) = 1 (reference)
    coeffs = np.exp(np.array([0., lc1, lc2, lc3, lc4, lc5, lc6, lc7]))
    n = DEGREE
    t = np.clip((x - X_MIN) / (X_MAX - X_MIN), 0.0, 1.0)
    poly = sum(c * scipy_comb(n, k, exact=True) * t**k * (1 - t)**(n - k)
               for k, c in enumerate(coeffs))
    norm = coeffs.sum() / (n + 1) * (X_MAX - X_MIN)
    return poly / norm

GJets_fake_min_gamma_ID = GJets_test_min_gamma_ID[fake_mask]
GJets_fake_weight       = GJets_test_weight[fake_mask]

values  = np.array(GJets_fake_min_gamma_ID, dtype=np.float64)
weights = np.array(GJets_fake_weight,       dtype=np.float64)
mask    = (values >= X_MIN) & (values <= X_MAX)
values, weights = values[mask], weights[mask]
weights = weights / weights.sum()   # normalise: stable NLL, doesn't affect shape

def weighted_nll(lc1, lc2, lc3, lc4, lc5, lc6=0., lc7=0.):
    pdf_vals = bernstein_pdf(values, lc1, lc2, lc3, lc4, lc5, lc6, lc7)
    log_pdf  = np.log(np.maximum(pdf_vals, 1e-300))
    return -np.sum(weights * log_pdf)

weighted_nll.errordef = Minuit.LIKELIHOOD   # 0.5 → correct 1σ for log-likelihood

m = Minuit(weighted_nll, lc1=0., lc2=0., lc3=0., lc4=0., lc5=0., lc6=0., lc7=0.)
m.migrad()
m.hesse()
print(m)

"""visualize the fit result"""
# normalize the event counts to be comparable with the PDF shape (density), but keep the original counts for error calculation and chi2/pull
# on the other hand, also can scale up the pdf shape to event counts by multiplying by total weight and bin width, which doesn't affect the fit but makes the visual comparison more intuitive.

# ── Reconstruct full 5-coefficient array from fit result ──────────────
fitted_coeffs = np.exp(np.array([0., *m.values]))   # c0=1 fixed, rest from fit

# ── Use ORIGINAL (un-normalised) weights for event-count histogram ────
values_orig  = np.array(GJets_fake_min_gamma_ID, dtype=np.float64)
weights_orig = np.array(GJets_fake_weight,        dtype=np.float64)
mask_orig    = (values_orig >= X_MIN) & (values_orig <= X_MAX)
values_orig, weights_orig = values_orig[mask_orig], weights_orig[mask_orig]

bins_val = np.linspace(X_MIN, X_MAX, 30)
bin_w    = bins_val[1] - bins_val[0]
bc       = (bins_val[1:] + bins_val[:-1]) / 2

# Total weighted events — used to scale the normalised PDF → event counts
total_w = weights_orig.sum()

# Weighted event counts per bin
hist_counts, _ = np.histogram(values_orig, bins=bins_val, weights=weights_orig / total_w)  # normalised to density
# Error: sqrt(sum(w_i^2)) per bin
hist_err, _    = np.histogram(values_orig, bins=bins_val, weights=(weights_orig / total_w)**2)
hist_err       = np.sqrt(hist_err)

# ── Fit curve scaled to event counts: PDF * total_events * bin_width ──
x_plot   = np.linspace(X_MIN, X_MAX, 400)
fit_plot = bernstein_pdf(x_plot, *m.values) * bin_w   # events/bin

# Scale each basis contribution the same way
n      = DEGREE
t_plot = (x_plot - X_MIN) / (X_MAX - X_MIN)
norm   = fitted_coeffs.sum() / (n + 1) * (X_MAX - X_MIN)
basis_scaled = [
    fitted_coeffs[k] * scipy_comb(n, k, exact=True)
    * t_plot**k * (1 - t_plot)**(n - k) / norm
    * bin_w
    for k in range(n + 1)
]

# ── Fit curve at bin centres (for chi2 and pull) ──────────────────────
fit_at_bc = bernstein_pdf(bc, *m.values) * bin_w

# ── Plot ──────────────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
                        sharex=True, figsize=(10, 10))

# Upper panel
axs[0].errorbar(bc, hist_counts, yerr=hist_err,
                fmt="ko", markersize=5, label="Gen-matched fake photons")
axs[0].plot(x_plot, fit_plot, "r-", lw=2.5, label=f"Bernstein deg-{DEGREE} fit")
colors = plt.cm.tab10(np.linspace(0, 1, n + 1))
for k, basis in enumerate(basis_scaled):
    axs[0].fill_between(x_plot, basis, alpha=0.25, color=colors[k], label=f"$B_{{k={k}}}$")

chi2 = np.sum(((hist_counts - fit_at_bc) / np.where(hist_err > 0, hist_err, 1))**2)
ndof = int(np.sum(hist_err > 0)) - len(m.values)
axs[0].text(0.97, 0.95, f"$\\chi^2$/ndf = {chi2:.1f}/{ndof}",
            transform=axs[0].transAxes, ha="right", va="top", fontsize=12)
axs[0].set_xlim(X_MIN, X_MAX)
axs[0].set_ylim(bottom=0)
axs[0].set_ylabel("Events", loc="top")
axs[0].legend(loc="upper right", fontsize=10, ncol=2)
hep.cms.label(loc=0, data=False, label="Simulation Preliminary", com=13.6, ax=axs[0])

# Lower panel: pull
pull = np.where(hist_err > 0, (hist_counts - fit_at_bc) / hist_err, 0)
axs[1].bar(bc, pull, width=bin_w * 0.9, color="steelblue", alpha=0.7)
axs[1].axhline( 0, color="red",  lw=1.5, linestyle="--")
axs[1].axhline( 2, color="gray", lw=1,   linestyle=":")
axs[1].axhline(-2, color="gray", lw=1,   linestyle=":")
axs[1].set_ylim(-4, 4)
axs[1].set_ylabel("Pull", loc="center")
axs[1].set_xlabel("min photon MVA ID")

plt.tight_layout()
plt.savefig(f"minID_distribution_fit_{suffix}.png")

"""generate minID from fitting result"""
################################################################################################################################
########## Generate min_gamma_ID from fitted Bernstein PDF and assign to data-driven sample ##########
# Range is per-event: [B_BOUND, orig_max_mvaID_i]
#   lower = B_BOUND = -0.7 (lower edge of signal region, same for all events)
#   upper = max(lead_mvaID, sublead_mvaID) of that event (the real photon's ID)
#
# Method: per-event truncated Beta inverse CDF sampling
#   For component k with Beta(k+1, n-k+1):
#     t = Beta_ppf( CDF(t_low) + u * (CDF(t_up_i) - CDF(t_low)),  k+1, n-k+1 )
#   where u ~ Uniform(0,1), t_low/t_up_i are boundaries mapped to [0,1] t-space.
#   Mixture weights are also re-normalised per event to account for different upper bounds.
################################################################################################################################
from scipy.stats import beta as beta_dist

# ── Per-event original max mvaID (real photon, stays unchanged) ───────
orig_lead_np    = ak.to_numpy(data_driven_ntuples.lead_mvaID)
orig_sublead_np = ak.to_numpy(data_driven_ntuples.sublead_mvaID)
orig_max_mvaID  = np.maximum(orig_lead_np, orig_sublead_np)   # shape (N,)

N = len(data_driven_ntuples)

# ── Boundaries in t-space: t = (x - X_MIN) / (X_MAX - X_MIN) ────────
B_BOUND = -0.7
t_low   = (B_BOUND    - X_MIN) / (X_MAX - X_MIN)              # scalar, same for all
t_up    = (np.clip(orig_max_mvaID, B_BOUND, X_MAX) - X_MIN) / (X_MAX - X_MIN)  # shape (N,)

# ── Fitted mixture weights (global, from Bernstein fit) ───────────────
fitted_coeffs = np.exp(np.array([0., *m.values]))  # shape (DEGREE+1,)
global_probs  = fitted_coeffs / fitted_coeffs.sum()

# ── Per-event mixture renormalisation ─────────────────────────────────
# w_ki = global_probs[k] * (CDF(t_up_i, k+1, n-k+1) - CDF(t_low, k+1, n-k+1))
# shape: (DEGREE+1, N)
cdf_low  = np.array([beta_dist.cdf(t_low,   k+1, DEGREE-k+1) for k in range(DEGREE+1)])[:, None]  # (DEGREE+1, 1)
cdf_up   = np.array([beta_dist.cdf(t_up,    k+1, DEGREE-k+1) for k in range(DEGREE+1)])           # (DEGREE+1, N)
w        = global_probs[:, None] * (cdf_up - cdf_low)          # (DEGREE+1, N)
w_sum    = w.sum(axis=0)                                         # (N,)  normalisation per event
mix_probs_per_event = w / w_sum                                  # (DEGREE+1, N)

# ── Per-event component draw ──────────────────────────────────────────
# np.random.choice does not vectorize over probabilities → use cumsum + uniform
u_comp    = np.random.uniform(size=N)                            # (N,)
cum_probs = np.cumsum(mix_probs_per_event, axis=0)               # (DEGREE+1, N)
components = (u_comp[None, :] > cum_probs).sum(axis=0)           # (N,) component index per event
components = np.clip(components, 0, DEGREE)

# ── Per-event truncated Beta inverse CDF sampling ────────────────────
t_samples = np.empty(N)
for k in range(DEGREE + 1):
    idx = np.where(components == k)[0]
    if len(idx) == 0:
        continue
    cdf_lo_k = float(cdf_low[k, 0])               # scalar — index both dims of (DEGREE+1, 1)
    cdf_hi_k = cdf_up[k, idx]                    # (n_k,) per-event upper CDF
    u_k      = np.random.uniform(size=len(idx))  # (n_k,)
    # Inverse CDF of truncated Beta
    p_k      = cdf_lo_k + u_k * (cdf_hi_k - cdf_lo_k)
    p_k      = np.clip(p_k, 1e-10, 1 - 1e-10)   # numerical safety
    t_samples[idx] = beta_dist.ppf(p_k, k+1, DEGREE-k+1)

# ── Map t back to x ──────────────────────────────────────────────────
gen_min_gamma_ID = X_MIN + t_samples * (X_MAX - X_MIN)
# Hard clip: beta_dist.ppf has floating-point rounding error near interval boundaries.
# For events where orig_max_mvaID ≈ B_BOUND (very narrow CDF interval), the inverse
# CDF can return a value fractionally above t_up_i. This clip enforces the constraint exactly.
gen_min_gamma_ID = np.clip(gen_min_gamma_ID, B_BOUND, orig_max_mvaID)

# ── Attach to data-driven sample ─────────────────────────────────────
data_driven_ntuples = ak.with_field(
    data_driven_ntuples,
    gen_min_gamma_ID,
    where="min_gamma_ID_gen"
)

n_out = np.sum((gen_min_gamma_ID < B_BOUND) | (gen_min_gamma_ID > orig_max_mvaID))
print(f"Generated {N} min_gamma_ID values")
print(f"  range : [{gen_min_gamma_ID.min():.4f}, {gen_min_gamma_ID.max():.4f}]")
print(f"  mean  : {gen_min_gamma_ID.mean():.4f}")
print(f"  Events outside [B_BOUND, orig_max_mvaID]: {n_out} (should be 0)")

# ── Closure plot: compare generated vs fitted PDF (truncated to [B_BOUND, X_MAX]) ─
bins_check  = np.linspace(B_BOUND, X_MAX, 40)
bin_w_check = bins_check[1] - bins_check[0]
bc_check    = (bins_check[1:] + bins_check[:-1]) / 2

gen_hist, _ = np.histogram(gen_min_gamma_ID, bins=bins_check)
gen_err     = np.sqrt(gen_hist)
# Reference: untruncated fit curve scaled to N events (approximate, since upper bound varies)
fit_curve   = bernstein_pdf(bc_check, *m.values) * N * bin_w_check

fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
                        sharex=True, figsize=(10, 8))
axs[0].errorbar(bc_check, gen_hist, yerr=gen_err, fmt="ko", markersize=4,
                label=f"Generated (per-event truncated, N={N:,})")
axs[0].plot(bc_check, fit_curve, "r--", lw=2, alpha=0.7,
            label=f"Bernstein deg-{DEGREE} (untruncated, reference)")
axs[0].axvline(B_BOUND, color="gray", lw=1.5, linestyle=":", label=f"B_BOUND = {B_BOUND}")
axs[0].set_xlim(B_BOUND - 0.05, X_MAX)
axs[0].set_ylim(bottom=0)
axs[0].set_ylabel("Events", loc="top")
axs[0].legend()
hep.cms.label(loc=0, data=False, label="Simulation Preliminary", com=13.6, ax=axs[0])

pull_check = np.where(gen_err > 0, (gen_hist - fit_curve) / gen_err, 0)
axs[1].bar(bc_check, pull_check, width=bin_w_check * 0.9, color="steelblue", alpha=0.7)
axs[1].axhline( 0, color="red",  lw=1.5, linestyle="--")
axs[1].axhline( 2, color="gray", lw=1,   linestyle=":")
axs[1].axhline(-2, color="gray", lw=1,   linestyle=":")
axs[1].set_ylim(-4, 4)
axs[1].set_ylabel("Pull", loc="center")
axs[1].set_xlabel("min photon MVA ID (generated)")
plt.tight_layout()
plt.savefig(f"minID_distribution_gen_{suffix}.png")

"""map generated minID back to data-driven sample"""
################################################################################################################################
########## Map generated min_gamma_ID_gen back to lead_mvaID / sublead_mvaID ##########
# Logic:
#   - lead/sublead ordering is fixed by pT (not mvaID) → do not swap photons
#   - for each event, the original lead_mvaID <= sublead_mvaID  OR  lead_mvaID > sublead_mvaID
#     → whichever slot was originally the "min" gets the generated value
#     → the other slot (the "max") keeps its original value unchanged
################################################################################################################################

orig_lead    = ak.to_numpy(data_driven_ntuples.lead_mvaID)
orig_sublead = ak.to_numpy(data_driven_ntuples.sublead_mvaID)
gen_min      = ak.to_numpy(data_driven_ntuples.min_gamma_ID_gen)

# Boolean mask: True where lead is the min photon
lead_is_min = orig_lead <= orig_sublead

# Reconstruct: replace the min-slot with gen_min, keep max-slot unchanged
new_lead_mvaID    = np.where(lead_is_min, gen_min,      orig_lead)
new_sublead_mvaID = np.where(lead_is_min, orig_sublead, gen_min)

# Attach back as new fields (originals preserved for validation)
data_driven_ntuples = ak.with_field(data_driven_ntuples, new_lead_mvaID,    "lead_mvaID_gen")
data_driven_ntuples = ak.with_field(data_driven_ntuples, new_sublead_mvaID, "sublead_mvaID_gen")

# ── Sanity checks ─────────────────────────────────────────────────────
print("=== Sanity checks ===")
print(f"Events where lead is min   : {lead_is_min.sum():,} / {len(lead_is_min):,}")
print(f"Events where sublead is min: {(~lead_is_min).sum():,} / {len(lead_is_min):,}")

# The fake-photon slot must equal gen_min exactly
fake_lead_err    = np.abs(new_lead_mvaID[lead_is_min]     - gen_min[lead_is_min]).max()
fake_sublead_err = np.abs(new_sublead_mvaID[~lead_is_min] - gen_min[~lead_is_min]).max()
print(f"max |new_lead[lead_is_min] - gen_min|       = {fake_lead_err:.2e}  (should be 0)")
print(f"max |new_sublead[~lead_is_min] - gen_min|   = {fake_sublead_err:.2e}  (should be 0)")

# The real-photon slot must be unchanged from original
real_sublead_err = np.abs(new_sublead_mvaID[lead_is_min]  - orig_sublead[lead_is_min]).max()
real_lead_err    = np.abs(new_lead_mvaID[~lead_is_min]    - orig_lead[~lead_is_min]).max()
print(f"max |new_sublead[lead_is_min] - orig_sublead| = {real_sublead_err:.2e}  (should be 0)")
print(f"max |new_lead[~lead_is_min]   - orig_lead|    = {real_lead_err:.2e}  (should be 0)")

# Informational: how often does gen_min exceed the real photon's ID?
n_cross = np.sum(
    np.where(lead_is_min, gen_min > orig_sublead, gen_min > orig_lead)
)
print(f"\nEvents where gen_min > real photon mvaID: {n_cross:,} / {N:,}")
print("(This is normal — fake photon ID can be higher than real photon ID after resampling)")
print(f"\nNew fields: 'lead_mvaID_gen', 'sublead_mvaID_gen'")
print(f"Available fields: {data_driven_ntuples.fields}")

"""apply per event weight(or said, scale factor) to data-driven sample"""
# per event weight w = (integral of fake PDF from B to max ID) / (integral of fake PDF from A to B)
# where A and B are the lower limit of mva ID in sideband and signal region respectively. i.e. A = -0.9, B = -0.7 in this case.
# actually, t_C, the maxID will depends on each event, as the field of data_driven_ntuples, max_gamma_ID

################################################################################################################################
########## Per-event transfer weight: sideband → signal region ##########
# w_i = integral(f, B, max_mvaID_i) / integral(f, A, B)
# A          = -0.9  (sideband lower edge, same for all events)
# B          = -0.7  (sideband / signal region boundary, same for all events)
# max_mvaID_i = max(lead_mvaID, sublead_mvaID) of event i  ← per-event upper limit
# Bernstein PDF is a Beta mixture → integrals are exact via Beta CDF
################################################################################################################################
from scipy.stats import beta as beta_dist

A_SIDE  = -0.9   # sideband lower edge
B_BOUND = -0.7   # sideband / signal region boundary

# ── Per-event upper limit: real photon mvaID ──────────────────────────
orig_lead_np_w    = ak.to_numpy(data_driven_ntuples.lead_mvaID)
orig_sublead_np_w = ak.to_numpy(data_driven_ntuples.sublead_mvaID)
orig_max_mvaID_w  = np.maximum(orig_lead_np_w, orig_sublead_np_w)          # shape (N,)
orig_max_mvaID_w  = np.clip(orig_max_mvaID_w, B_BOUND, X_MAX)              # safety clip

# ── Mixture probabilities from fitted coefficients ────────────────────
fitted_coeffs = np.exp(np.array([0., *m.values]))   # shape: (DEGREE+1,)
mix_probs     = fitted_coeffs / fitted_coeffs.sum() # p_k = c_k / sum(c)

# ── Transform boundaries to t-space: t = (x - X_MIN) / (X_MAX - X_MIN) ─
t_A   = (A_SIDE          - X_MIN) / (X_MAX - X_MIN)   # scalar
t_B   = (B_BOUND         - X_MIN) / (X_MAX - X_MIN)   # scalar
t_C_i = (orig_max_mvaID_w - X_MIN) / (X_MAX - X_MIN)  # shape (N,) — per-event upper bound

# ── Vectorised analytical integral via Beta CDF ───────────────────────
# integral(f, t1, t2) = sum_k p_k * [CDF(t2; k+1, n-k+1) - CDF(t1; k+1, n-k+1)]
# t1, t2 can be scalars or arrays of shape (N,); result is same shape as t2.
def bernstein_integral_vec(t1, t2):
    """Vectorised integral of the normalised Bernstein PDF.
    t1, t2: scalar or ndarray of shape (N,). Returns same shape as t2."""
    total = np.zeros_like(np.asarray(t2, dtype=float))
    for k in range(DEGREE + 1):
        total += mix_probs[k] * (
            beta_dist.cdf(t2, k + 1, DEGREE - k + 1) -
            beta_dist.cdf(t1, k + 1, DEGREE - k + 1)
        )
    return total

integral_sideband = bernstein_integral_vec(t_A, t_B)    # scalar  P(A < x < B)
integral_signal_i = bernstein_integral_vec(t_B, t_C_i)  # shape (N,)  P(B < x < max_i)

# ── Per-event transfer weight ─────────────────────────────────────────
transfer_weight_i = integral_signal_i / integral_sideband   # shape (N,)

print(f"Sideband    [{A_SIDE:.2f}, {B_BOUND:.2f}]        : integral = {float(integral_sideband):.6f}")
print(f"Signal mean [{B_BOUND:.2f}, max_mvaID_i] : integral mean = {integral_signal_i.mean():.6f}")
print(f"Transfer weight — mean: {transfer_weight_i.mean():.4f}  "
      f"min: {transfer_weight_i.min():.4f}  max: {transfer_weight_i.max():.4f}")

# ── Attach as per-event weight field ─────────────────────────────────
data_driven_ntuples = ak.with_field(
    data_driven_ntuples,
    transfer_weight_i,
    where="transfer_weight"
)

print(f"\nField 'transfer_weight' added to all {len(data_driven_ntuples):,} data-driven events.")
print(f"Available fields: {data_driven_ntuples.fields}")

"""sanity check: maxID should > minID"""
# sanity check for origin maxID with after minID generated
# the maxID shouldn't change
orig_max_mvaID_check = np.maximum(
    ak.to_numpy(data_driven_ntuples.lead_mvaID),
    ak.to_numpy(data_driven_ntuples.sublead_mvaID)
)
maxID_check = np.max(np.abs(orig_max_mvaID_check - orig_max_mvaID_w))
print(f"\nSanity check: max |orig_max_mvaID_check - orig_max_mvaID_w| = {maxID_check:.2e} (should be 0)")

"""apply simultaneous binned likelihood fit to extract SF"""

##################################################################################################
########## Simultaneous binned likelihood fit: max_gamma_ID + min_gamma_ID               ########
########## Free: sf_dd (data-driven γ+jets), sf_diphoton (non-res. diphoton)            ########
########## Fixed: Diboson, DYJets, Top at MC prediction                                 ########
##################################################################################################
from iminuit import Minuit

def _to_np(arr):
    """Force to plain float64 numpy array (handles awkward + masked arrays)."""
    if hasattr(arr, 'to_numpy'):
        return ak.to_numpy(arr).astype(float)
    return np.asarray(arr, dtype=float)

# ── Channel 1: max_gamma_ID  (range -0.4 to 1, 30 bins) ─────────────────────────────────────
bins_max     = np.linspace(-0.7, 1.0, 31)
N_obs_max    = np.histogram(Data_max_gamma_ID, bins=bins_max)[0].astype(float)
N_dd_max, _  = np.histogram(
    orig_max_mvaID, bins=bins_max,
    weights=ak.to_numpy(data_driven_ntuples["transfer_weight"])
)
N_diphoton_max = _to_np(np.histogram(Diphotons_max_gamma_ID, bins=bins_max, weights=Diphotons_weight)[0])
N_fixed_max = (
    _to_np(np.histogram(Diboson_max_gamma_ID, bins=bins_max, weights=Diboson_weight)[0]) +
    _to_np(np.histogram(DYJets_max_gamma_ID,  bins=bins_max, weights=DYJets_weight)[0])  +
    _to_np(np.histogram(Top_max_gamma_ID,     bins=bins_max, weights=Top_weight)[0])
)

# ── Channel 2: min_gamma_ID  (range -0.9 to 1, 30 bins) ─────────────────────────────────────
bins_min     = np.linspace(-0.7, 1.0, 31)
N_obs_min    = np.histogram(Data_min_gamma_ID, bins=bins_min)[0].astype(float)
N_dd_min, _  = np.histogram(
    ak.to_numpy(data_driven_ntuples["min_gamma_ID_gen"]), bins=bins_min,
    weights=ak.to_numpy(data_driven_ntuples["transfer_weight"])
)
N_diphoton_min = _to_np(np.histogram(Diphotons_min_gamma_ID, bins=bins_min, weights=Diphotons_weight)[0])
N_fixed_min = (
    _to_np(np.histogram(Diboson_min_gamma_ID, bins=bins_min, weights=Diboson_weight)[0]) +
    _to_np(np.histogram(DYJets_min_gamma_ID,  bins=bins_min, weights=DYJets_weight)[0])  +
    _to_np(np.histogram(Top_min_gamma_ID,     bins=bins_min, weights=Top_weight)[0])
)

# ── Simultaneous Poisson NLL (sum over both channels) ────────────────────────────────────────
def nll_simul(sf_dd, sf_diphoton):
    N_exp_max = sf_dd * N_dd_max + sf_diphoton * N_diphoton_max + N_fixed_max
    safe_max  = np.maximum(N_exp_max, 1e-10)
    nll_max   = 2.0 * np.sum(safe_max - N_obs_max * np.log(safe_max))
    N_exp_min = sf_dd * N_dd_min + sf_diphoton * N_diphoton_min + N_fixed_min
    safe_min  = np.maximum(N_exp_min, 1e-10)
    nll_min   = 2.0 * np.sum(safe_min - N_obs_min * np.log(safe_min))
    return nll_max + nll_min

nll_simul.errordef = Minuit.LIKELIHOOD

m_simul = Minuit(nll_simul, sf_dd=1.0, sf_diphoton=1.0)
m_simul.limits["sf_dd"]       = (0.0, 10.0)
m_simul.limits["sf_diphoton"] = (0.0, 10.0)
m_simul.migrad()
m_simul.hesse()

print(m_simul)
sf_dd_val       = m_simul.values["sf_dd"]
sf_diphoton_val = m_simul.values["sf_diphoton"]
sf_dd_unc       = m_simul.errors["sf_dd"]
sf_diphoton_unc = m_simul.errors["sf_diphoton"]
print(f"\nFit valid        : {m_simul.valid}")
print(f"SF(data-driven)  = {sf_dd_val:.4f} +/- {sf_dd_unc:.4f}")
print(f"SF(diphoton MC)  = {sf_diphoton_val:.4f} +/- {sf_diphoton_unc:.4f}")

# ── Post-fit comparison plots ────────────────────────────────────────────────────────────────
def _draw_postfit_channel(ax_top, ax_bot, bins, N_obs, N_dd, N_diphoton, N_fixed,
                          sf_dd, sf_diphoton, xlabel):
    bc    = (bins[1:] + bins[:-1]) / 2
    bw    = bins[1] - bins[0]
    N_exp = sf_dd * N_dd + sf_diphoton * N_diphoton + N_fixed
    # Stacked bars
    ax_top.bar(bc, N_fixed,                  width=bw,                          color="#ffd700", alpha=0.8,
               label="Fixed MC (Diboson+DY+Top)")
    ax_top.bar(bc, sf_diphoton * N_diphoton, width=bw, bottom=N_fixed,          color="#0000ff", alpha=0.8,
               label=f"Diphoton (SF={sf_diphoton:.3f})")
    ax_top.bar(bc, sf_dd * N_dd,             width=bw,
               bottom=N_fixed + sf_diphoton * N_diphoton,
               color="#7cfc00", alpha=0.8, label=rf"$\gamma$+jets DD (SF={sf_dd:.3f})")
    ax_top.errorbar(bc, N_obs, yerr=np.sqrt(np.maximum(N_obs, 0)), fmt="ko", label="Data")
    ax_top.set_ylabel("Events", loc="top")
    ax_top.set_ylim(bottom=0)
    ax_top.legend(fontsize=9, ncol=2)
    # Data/MC ratio (pure numpy, no awkward)
    safe_exp  = np.maximum(N_exp, 1e-10)
    ratio     = np.where(safe_exp > 0, N_obs / safe_exp, 0.0)
    ratio_err = np.where(N_obs > 0, np.sqrt(N_obs) / safe_exp, 0.0)
    ax_bot.errorbar(bc, ratio, yerr=ratio_err, fmt="ko")
    ax_bot.axhline(1, color="red", linestyle="--", linewidth=1)
    ax_bot.set_ylim(0, 2.0)
    ax_bot.set_ylabel("Data/MC")
    ax_bot.set_xlabel(xlabel)

fig, axes = plt.subplots(2, 2, figsize=(18, 12),
                         gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
                         sharey="row")
_draw_postfit_channel(axes[0, 0], axes[1, 0], bins_max,
                      N_obs_max, N_dd_max, N_diphoton_max, N_fixed_max,
                      sf_dd_val, sf_diphoton_val, r"Max $\gamma$ ID")
_draw_postfit_channel(axes[0, 1], axes[1, 1], bins_min,
                      N_obs_min, N_dd_min, N_diphoton_min, N_fixed_min,
                      sf_dd_val, sf_diphoton_val, r"Min $\gamma$ ID")
fig.suptitle(
    f"Simultaneous fit (max ID + min ID)\n"
    f"SF(data-driven) = {sf_dd_val:.3f} $\\pm$ {sf_dd_unc:.3f},   "
    f"SF(diphoton) = {sf_diphoton_val:.3f} $\\pm$ {sf_diphoton_unc:.3f}",
    fontsize=13
)
hep.cms.label(loc=0, data=True, label="Preliminary", lumi=62.456,
              lumi_format="{0:.1f}", com=13.6, ax=axes[0, 0])
plt.tight_layout()
plt.savefig(f"postfit_comparison_{suffix}.png")

"""export data-driven sample to merge-compatible parquet"""
################################################################################################################################
########## Export data-driven sample to parquet (merge_parquet.py compatible) ##########
# Output structure: DD_OUTPUT_DIR/data_driven_bkg/selected_Events_0-N.parquet
# merge_parquet.py globs *.parquet directly from the source_path, so files sit flat
# in the dataset folder (no variation sub-directory needed).
#
# Call merge with --is-data (or --skip-normalisation) because the weight field already
# encodes the full normalisation: transfer_weight × sf_dd_val.
# The metadata key sum_genw_presel = b"Data" signals that no gen-weight division is needed.
################################################################################################################################
import pathlib

DD_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_driven_output")
DD_DATASET    = "data_driven_bkg"

# ── Build export array ────────────────────────────────────────────────
# Replace photon ID fields with generated signal-region values so that
# downstream tools (BDT, FinalFits) see mvaID in the signal region.
dd_export = data_driven_ntuples
dd_export = ak.with_field(dd_export, ak.to_numpy(dd_export.lead_mvaID_gen),    "lead_mvaID")
dd_export = ak.with_field(dd_export, ak.to_numpy(dd_export.sublead_mvaID_gen), "sublead_mvaID")

# Final per-event weight = sideband→signal transfer weight × fitted scale factor.
# This is the fully-normalised physical weight, analogous to lumi × xsec × genWeight / sum_genw
# in a regular MC sample.
final_weight = ak.to_numpy(dd_export.transfer_weight) * sf_dd_val
dd_export = ak.with_field(dd_export, final_weight, "weight")

# Also expose as genWeight so that weight-computing functions that read ntuple.genWeight
# (e.g. get_weight() in BDT training scripts) pick up the correct value.
# sum_genw_presel is set to "1.0" so merge_parquet.py divides by 1 → weight unchanged.
dd_export = ak.with_field(dd_export, final_weight, "genWeight")

# ── Write parquet ─────────────────────────────────────────────────────
dataset_dir = os.path.join(DD_OUTPUT_DIR, DD_DATASET)
pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)

fname       = f"selected_Events_0-{len(dd_export)}.parquet"
destination = os.path.join(dataset_dir, fname)

pa_table  = ak.to_arrow_table(dd_export, extensionarray=False)
# Deterministic column ordering (mirrors HiggsDNA processor convention)
col_names = sorted(pa_table.schema.names)
pa_table  = pa.table([pa_table.column(n) for n in col_names], names=col_names)

# Byte-key metadata: merge_parquet.py reads b'sum_genw_presel' directly.
# sum_genw_presel = "1.0": merge divides weight by 1.0 → weight unchanged.
# This allows merge_parquet.py to process the sample as a regular MC (no --is-data needed).
export_metadata = {
    b"sum_genw_presel" : b"1.0",
    b"is_data_driven"  : b"True",
    b"sf_dd"           : str(sf_dd_val).encode(),
    b"sf_diphoton"     : str(sf_diphoton_val).encode(),
}
if pa_table.schema.metadata:
    export_metadata = {**pa_table.schema.metadata, **export_metadata}
pa_table = pa_table.replace_schema_metadata(export_metadata)

pq.write_table(pa_table, destination)

print(f"\n[EXPORT] Data-driven sample written to  : {destination}")
print(f"         Events                           : {len(dd_export):,}")
print(f"         Weight (transfer × SF) — "
      f"mean: {final_weight.mean():.4f}  "
      f"min: {final_weight.min():.4f}  "
      f"max: {final_weight.max():.4f}")
print(f"         SF(data-driven) used             : {sf_dd_val:.4f}")
print(f"         genWeight field                  : set equal to weight (for BDT get_weight() compat.)")
print(f"         sum_genw_presel                  : 1.0  (merge divides by 1 → weight unchanged)")
print(f"\n[MERGE] Run merge step with (no --is-data needed):")
print(f"  python merge_parquet.py \\")
print(f"    --source {dataset_dir}/ \\")
print(f"    --target <merged_output_dir>/ \\")
print(f"    --skip-normalisation")


