import awkward as ak
import json
import numpy as np
import os
import vector
import argparse
import glob
import pyarrow.parquet as pq

parser = argparse.ArgumentParser(description = "Variables for ZH MET BDT training.")

parser.add_argument(
    "-s",
    "--sample_name",
    type = str,
    help = "Specify the process name you want to load.",
    choices = ("Diphoton", "DYJets", "GJets", "QCD", "Top", "VV", "VG", "WG", "ZG", "VBF", "ggH", "ggH_powheg", "ttH", "WH_signal", "ZH_Zto2L_signal", "ZH_Zto2Nu_signal", "VH_bkg", "data_driven"),
)
parser.add_argument(
    "-o",
    "--output_filename",
    type = str,
    help = "Specify the name of the npy file.",
)

args = parser.parse_args()

def get_dielectron_4momentum(sample):
    lead_4momentum = vector.array(
        {
            "pt" : sample.electron0_pt,
            "eta" : sample.electron0_eta,
            "phi" : sample.electron0_phi,
            "mass" : sample.electron0_mass
        }
    )
    sublead_4momentum = vector.array(
        {
            "pt" : sample.electron1_pt,
            "eta" : sample.electron1_eta,
            "phi" : sample.electron1_phi,
            "mass" : sample.electron1_mass
        }
    )
    return lead_4momentum + sublead_4momentum

def get_dimuon_4momentum(sample):
    lead_4momentum = vector.array(
        {
            "pt" : sample.muon0_pt,
            "eta" : sample.muon0_eta,
            "phi" : sample.muon0_phi,
            "mass" : sample.muon0_mass
        }
    )
    sublead_4momentum = vector.array(
        {
            "pt" : sample.muon1_pt,
            "eta" : sample.muon1_eta,
            "phi" : sample.muon1_phi,
            "mass" : sample.muon1_mass
        }
    )
    return lead_4momentum + sublead_4momentum

def get_electron_met_transverse_momentum(sample):
    lead_electron_transverse_momentum = vector.array(
        {
            "rho" : sample.electron0_pt,
            "phi" : sample.electron0_phi
        }
    )
    met_transverse_momentum = vector.array(
        {
            "rho" : sample.MET_pt,
            "phi" : sample.MET_phi
        }
    )
    return lead_electron_transverse_momentum + met_transverse_momentum

def get_muon_met_transverse_momentum(sample):
    lead_muon_transverse_momentum = vector.array(
        {
            "rho" : sample.muon0_pt,
            "phi" : sample.muon0_phi
        }
    )
    met_transverse_momentum = vector.array(
        {
            "rho" : sample.MET_pt,
            "phi" : sample.MET_phi
        }
    )
    return lead_muon_transverse_momentum + met_transverse_momentum

preEE_luminosity = 7.9804 * 1000
postEE_luminosity = 26.6717 * 1000
preBPix_luminosity = 18.063 * 1000
postBPix_luminosity = 9.693 * 1000
with open('/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/cross_sections_for_Ntuples_v3.json', 'r') as f:
    cross_sections = json.load(f)
with open('/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/2022preEE_total_sum_genWeight_using_Ntuples_v3.json', 'r') as f:
    preEE_sum_genWeight = json.load(f)
with open('/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/2022postEE_total_sum_genWeight_using_Ntuples_v3.json', 'r') as f:
    postEE_sum_genWeight = json.load(f)
with open("/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/2023preBPix_total_sum_genWeight_using_Ntuples_v3.json", "r") as f:
    preBPix_sum_genWeight = json.load(f)
with open("/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/plotting_configs/2023postBPix_total_sum_genWeight_using_Ntuples_v3.json", "r") as f:
    postBPix_sum_genWeight = json.load(f)

""" pause for get_weight update
def get_weight(ntuple_name, ntuple):
    ntuple_name_without_suffix = ntuple_name

    if "_preEE" in ntuple_name:
        ntuple_name_without_suffix = ntuple_name.replace("_preEE_selected", "")
        ntuple_name_without_selected_suffix = ntuple_name.replace("_selected", "")
        luminosity = preEE_luminosity
        total_sum_genWeight = preEE_sum_genWeight[ntuple_name_without_selected_suffix]
        # total_sum_genWeight = preEE_sum_genWeight.get(ntuple_name, 1)
    elif "_postEE" in ntuple_name:
        ntuple_name_without_suffix = ntuple_name.replace("_postEE_selected", "")
        ntuple_name_without_selected_suffix = ntuple_name.replace("_selected", "")
        luminosity = postEE_luminosity
        total_sum_genWeight = postEE_sum_genWeight[ntuple_name_without_selected_suffix]
        # total_sum_genWeight = postEE_sum_genWeight.get(ntuple_name, 1)
    elif "_preBPix" in ntuple_name:
        ntuple_name_without_suffix = ntuple_name.replace("_preBPix_selected", "")
        ntuple_name_without_selected_suffix = ntuple_name.replace("_selected", "")
        luminosity = preBPix_luminosity
        total_sum_genWeight = preBPix_sum_genWeight[ntuple_name_without_selected_suffix]
        # total_sum_genWeight = preBPix_sum_genWeight.get(ntuple_name, 1)
    elif "_postBPix" in ntuple_name:
        ntuple_name_without_suffix = ntuple_name.replace("_postBPix_selected", "")
        ntuple_name_without_selected_suffix = ntuple_name.replace("_selected", "")
        luminosity = postBPix_luminosity
        total_sum_genWeight = postBPix_sum_genWeight[ntuple_name_without_selected_suffix]
        # total_sum_genWeight = postBPix_sum_genWeight.get(ntuple_name, 1)
    else:
        luminosity = 1
        total_sum_genWeight = 1

    if ntuple_name_without_suffix in cross_sections:
        if "M_125" in ntuple_name_without_suffix:
            xsec = cross_sections[ntuple_name_without_suffix] * 0.0025 # 0.0025 is the branching ratio of H -> gamma gamma.
        else:
            xsec = cross_sections[ntuple_name_without_suffix]
        return luminosity * xsec * (ntuple.genWeight / total_sum_genWeight)
    else:
        return None
"""

# Initialize sum_genWeight_dict - will be populated during file loading
sum_genWeight_dict = {}
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
        xsec = 1
        return luminosity * xsec * (ntuple.genWeight / total_sum_genWeight)

################################################################################################################################
########## Load the Ntuples of 2022 preEE and postEE data and MC ##########
################################################################################################################################
sample_name = args.sample_name
sample_name_dict = {
    "Diphoton": ["GG-Box"],
    "DYJets": ["DYto2L"],
    "GJets": ["GJet_PT"],
    "QCD": ["QCD"],
    "Top": ["TGJets", "TTG", "TTGG", "TTto"],
    "VV": ["WW", "WZ", "ZZ"],
    "VG": ["WG", "DYGto2LG", "ZGto2NuG"],
    "WG": ["WG"],
    "ZG": ["ZG", "DYGto2LG"],
    "ZH_Zto2Nu_signal": ["ZH_Hto2G_Zto2Nu"],
    "WH_signal": ["WminusH_Hto2G_WtoLNu", "WplusH_Hto2G_WtoLNu"],
    "data_driven": ["data_driven"]
}
ntuples_dict = {} # Since the available options of sample_name do not include Data, we do not need to add an additional if-else statement to check if the sample_name is Data or MC.

basepath = {
    "all" : "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/NTuples_test01_selected01"
    # "preEE"  : "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/SelectedNTuples/2022preEE",
    # "postEE" : "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/SelectedNTuples/2022postEE",
    # "preBPix" : "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/SelectedNTuples/2023preBPix",
    # "postBPix" : "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/SelectedNTuples/2023postBPix"
}
print("===========================================================================")
for tag, path in basepath.items(): # tag: all; path: .../Ntuples_test01_selected01
    for category in os.listdir(basepath[tag]): # category: DYG, QCD
        if any(sample in category for sample in sample_name_dict[sample_name]):
            category_path = os.path.join(basepath[tag], category) # category_path: .../NTuples_test01_selected01/QCD
            if os.path.exists(category_path):
                for process in os.listdir(category_path): # process: nominal, scale_up, scale_down
                    ntuple_path = os.path.join(category_path, process) # ntuple_path: .../NTuples_test01_selected01/QCD/nominal
                    ntuple_name = category.replace("-", "_")
                    ntuples_dict[ntuple_name] = ak.from_parquet(ntuple_path)
                    print(f"2022-{tag} {ntuple_name}'s Ntuples are loaded.")
                    # Extract sum_genWeight from metadata and store in sum_genWeight_dict
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
                print(f"Path {category_path} does not exist.")
        else:
            print(f"skip {category} since {sample_name} is not in {category}")

print("===========================================================================")
print("\n")
print("===========================================================================")
print(ntuples_dict.keys()) # Output should include what you enter in arg.sample_name.
print(ntuples_dict.values())
print("===========================================================================")
print("\n")
################################################################################################################################
########## Apply ZH-leptonic selections on the 2022 data and MC ntuples ##########
################################################################################################################################
ZH_ntuples_dict = {}

for name, ntuple in ntuples_dict.items():
    """
    # Use vector library for proper 4-momentum addition
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
    ntuple["leptons_phi"] = lepton_4momentum.phi
    ntuple["leptons_CosDeltaPhi"] = np.cos(leptons0.phi - leptons1.phi)
    ntuple["diphoton_CosDeltaPhi"] = np.cos(ntuple["lead_phi"] - ntuple["sublead_phi"])
    # Calculate deltaR using vector library (need rho, phi, eta for 3D cylindrical coordinates)
    lead_photon = vector.zip({"rho": ntuple["lead_pt"], "phi": ntuple["lead_phi"], "eta": ntuple["lead_eta"]})
    sublead_photon = vector.zip({"rho": ntuple["sublead_pt"], "phi": ntuple["sublead_phi"], "eta": ntuple["sublead_eta"]})
    ntuple["deltaR_leadPhoton_leadLepton"] = lead_photon.deltaR(leptons0)
    ntuple["deltaR_leadPhoton_subleadLepton"] = lead_photon.deltaR(leptons1)
    ntuple["deltaR_subleadPhoton_leadLepton"] = sublead_photon.deltaR(leptons0)
    ntuple["deltaR_subleadPhoton_subleadLepton"] = sublead_photon.deltaR(leptons1)
    """
    ntuple["diphoton_CosDeltaPhi"] = np.cos(ntuple["lead_phi"] - ntuple["sublead_phi"])
    ntuple["delta_phi"] = abs(ntuple.phi - ntuple.MET_phi)
    for i in range(4): # Assuming there are at most 4 jets to consider for delta_phi calculation
        ntuple[f"delta_phi_{i}"] = abs(ntuple["MET_phi"] - ntuple[f"Jet{i}_phi"])
    ntuple["min_delta_phi"] = np.minimum(np.minimum(ntuple["delta_phi_0"], ntuple["delta_phi_1"]), np.minimum(ntuple["delta_phi_2"], ntuple["delta_phi_3"]))
    # Selection cuts on lepton id/iso, lepton pt, and deltaR between lepton and photon are included when making the ntuples.
    #ntuple = ntuple[(ntuple.mass < 115) | (ntuple.mass > 135)] # Blind the Higgs signal region except ZH_signal.
    #ntuple = ntuple[(ntuple.lead_mvaID > -0.9) & (ntuple.sublead_mvaID > -0.9)] # Photon id cut.
    #ntuple = ntuple[(ntuple["event_category"] == 2)] # two-lepton cut.
    #ntuple = ntuple[((ntuple["electron_mass"] >60) & (ntuple["electron_mass"] <120)) | ((ntuple["muon_mass"] > 60) & (ntuple["muon_mass"] < 120))] # Z mass window veto
    ZH_ntuples_dict[name] = ntuple

print("===========================================================================")
print(f"ZH-leptonic selections are applied on {ZH_ntuples_dict.keys()}.")
print(ZH_ntuples_dict.keys())
print("===========================================================================")
print("\n")
################################################################################################################################
########## Input variables for ZH BDT ##########
################################################################################################################################
"""========== weight =========="""
weight = np.array(np.concatenate([get_weight(name, nt) for name, nt in ZH_ntuples_dict.items()], axis = 0))

"""========== Photons =========="""
lead_photon_pt_mgg_ratio = np.concatenate([nt.lead_pt / nt.mass for nt in ZH_ntuples_dict.values()], axis = 0)
sublead_photon_pt_mgg_ratio = np.concatenate([nt.sublead_pt / nt.mass for nt in ZH_ntuples_dict.values()], axis = 0)
lead_photon_eta = np.concatenate([nt.lead_eta for nt in ZH_ntuples_dict.values()], axis = 0)
sublead_photon_eta = np.concatenate([nt.sublead_eta for nt in ZH_ntuples_dict.values()], axis = 0)
cos_delta_phi_photons = np.concatenate([nt["diphoton_CosDeltaPhi"] for nt in ZH_ntuples_dict.values()], axis = 0)
max_photon_id = np.concatenate([np.maximum(nt.lead_mvaID, nt.sublead_mvaID) for nt in ZH_ntuples_dict.values()], axis = 0)
min_photon_id = np.concatenate([np.minimum(nt.lead_mvaID, nt.sublead_mvaID) for nt in ZH_ntuples_dict.values()], axis = 0)
# (wh) lead_photon_pixelSeed = np.concatenate([nt.lead_pixelSeed for nt in ZH_ntuples_dict.values()], axis = 0)
# (wh) sublead_photon_pixelSeed = np.concatenate([nt.sublead_pixelSeed for nt in ZH_ntuples_dict.values()], axis = 0)

"""========== Leptons =========="""
# (folded in leotons) lead_electron_pt = np.concatenate([nt.electron0_pt for nt in ZH_ntuples_dict.values()], axis = 0)
# (folded in leptons) lead_muon_pt = np.concatenate([nt.muon0_pt for nt in ZH_ntuples_dict.values()], axis = 0)
# (folded in leptons) lead_electron_eta = np.concatenate([nt.electron0_eta for nt in ZH_ntuples_dict.values()], axis = 0)
# (folded in leptons) lead_muon_eta = np.concatenate([nt.muon0_eta for nt in ZH_ntuples_dict.values()], axis = 0)
"""
lead_lepton_pt = np.concatenate([nt["leptons0_pt"] for nt in ZH_ntuples_dict.values()], axis = 0)
sublead_lepton_pt = np.concatenate([nt["leptons1_pt"] for nt in ZH_ntuples_dict.values()], axis = 0)
lead_lepton_eta = np.concatenate([nt["leptons0_eta"] for nt in ZH_ntuples_dict.values()], axis = 0)
sublead_lepton_eta = np.concatenate([nt["leptons1_eta"] for nt in ZH_ntuples_dict.values()], axis = 0)
cos_delta_phi_leptons = np.concatenate([nt["leptons_CosDeltaPhi"] for nt in ZH_ntuples_dict.values()], axis = 0)
deltaR_lpll = np.concatenate([nt["deltaR_leadPhoton_leadLepton"] for nt in ZH_ntuples_dict.values()], axis = 0)
deltaR_lpsl = np.concatenate([nt["deltaR_leadPhoton_subleadLepton"] for nt in ZH_ntuples_dict.values()], axis = 0)
deltaR_spll = np.concatenate([nt["deltaR_subleadPhoton_leadLepton"] for nt in ZH_ntuples_dict.values()], axis = 0)
deltaR_spsl = np.concatenate([nt["deltaR_subleadPhoton_subleadLepton"] for nt in ZH_ntuples_dict.values()], axis = 0)
dileptons_mass = np.concatenate([nt["leptons_mass"] for nt in ZH_ntuples_dict.values()], axis = 0)
"""
# (folded in leptons) delta_R_ld_photon_ld_electron = np.concatenate([np.sqrt((nt.lead_eta - nt.electron0_eta)**2 + (nt.lead_phi - nt.electron0_phi)**2) for nt in ZH_ntuples_dict.values()], axis = 0)
# (folded in leptons) delta_R_sld_photon_ld_electron = np.concatenate([np.sqrt((nt.sublead_eta - nt.electron0_eta)**2 + (nt.sublead_phi - nt.electron0_phi)**2) for nt in ZH_ntuples_dict.values()], axis = 0)
# (folded in leptons) delta_R_ld_photon_ld_muon = np.concatenate([np.sqrt((nt.lead_eta - nt.muon0_eta)**2 + (nt.lead_phi - nt.muon0_phi)**2) for nt in ZH_ntuples_dict.values()], axis = 0)
# (folded in leptons) delta_R_sld_photon_ld_muon = np.concatenate([np.sqrt((nt.sublead_eta - nt.muon0_eta)**2 + (nt.sublead_phi - nt.muon0_phi)**2) for nt in ZH_ntuples_dict.values()], axis = 0)

# Lepton (here, electron + muon) information is obtained by concatenating the lead-electron and lead-muon arrays.
# (folded in leotons) is_electron_event = np.concatenate([nt.n_electrons == 1 for nt in ZH_ntuples_dict.values()])
# (folded in leptons) is_muon_event = ~is_electron_event

# (folded in leptons) lead_lepton_pt = np.where(is_electron_event, lead_electron_pt, lead_muon_pt)
# (folded in leptons) lead_lepton_eta = np.where(is_electron_event, lead_electron_eta, lead_muon_eta)
# (folded in leptons) delta_R_ld_photon_ld_lepton = np.where(is_electron_event, delta_R_ld_photon_ld_electron, delta_R_ld_photon_ld_muon)
# (folded in leptons) delta_R_sld_photon_ld_lepton = np.where(is_electron_event, delta_R_sld_photon_ld_electron, delta_R_sld_photon_ld_muon)
"""========== Missing Transverse Energy =========="""
met_pt = np.concatenate([nt.MET_pt for nt in ZH_ntuples_dict.values()], axis = 0)
met_sumEt = np.concatenate([nt.MET_sumEt for nt in ZH_ntuples_dict.values()], axis = 0)
# (wh) met_electron_Mt = np.concatenate([np.sqrt(2 * nt.electron0_pt * nt.MET_pt * (1 - np.cos(nt.electron0_phi - nt.MET_phi))) for nt in WH_ntuples_dict.values()], axis = 0)
# (wh) met_muon_Mt = np.concatenate([np.sqrt(2 * nt.muon0_pt * nt.MET_pt * (1 - np.cos(nt.muon0_phi - nt.MET_phi))) for nt in WH_ntuples_dict.values()], axis = 0)

# (wh) met_lepton_Mt = np.where(is_electron_event, met_electron_Mt, met_muon_Mt)
"""========== Jets =========="""
jet_multiplicity = np.concatenate([nt.n_jets for nt in ZH_ntuples_dict.values()], axis = 0)
jet0_pt = np.concatenate([nt.Jet0_pt for nt in ZH_ntuples_dict.values()], axis = 0)
# jet1_pt = np.concatenate([nt.Jet1_pt for nt in ZH_ntuples_dict.values()], axis = 0)
# jet2_pt = np.concatenate([nt.Jet2_pt for nt in ZH_ntuples_dict.values()], axis = 0)
# jet3_pt = np.concatenate([nt.Jet3_pt for nt in ZH_ntuples_dict.values()], axis = 0)
# (wh) jet_max_btagPNetB = np.concatenate([np.maximum.reduce([nt.Jet0_btagPNetB, nt.Jet1_btagPNetB, nt.Jet2_btagPNetB, nt.Jet3_btagPNetB]) for nt in ZH_ntuples_dict.values()], axis = 0)
# jet_max_btagDeepFlavB = np.concatenate([np.maximum.reduce([nt.Jet0_btagDeepFlavB, nt.Jet1_btagDeepFlavB, nt.Jet2_btagDeepFlavB, nt.Jet3_btagDeepFlavB]) for nt in ZH_ntuples_dict.values()], axis = 0)
# jet_max_btagRobustParTAK4B = np.concatenate([np.maximum.reduce([nt.Jet0_btagRobustParTAK4B, nt.Jet1_btagRobustParTAK4B, nt.Jet2_btagRobustParTAK4B, nt.Jet3_btagRobustParTAK4B]) for nt in ZH_ntuples_dict.values()], axis = 0)

"""========== ZH Topology =========="""
deltaPhi_ggMET = np.concatenate([nt.delta_phi for nt in ZH_ntuples_dict.values()], axis = 0)
min_delta_phi = np.concatenate([nt.min_delta_phi for nt in ZH_ntuples_dict.values()], axis = 0)
# (zhll) deltaPhi_ggll = np.concatenate([nt["phi"] - nt["leptons_phi"] for nt in ZH_ntuples_dict.values()], axis = 0)
# (wh) delta_phi_diphoton_W_electron = np.concatenate([nt.phi - get_electron_met_transverse_momentum(nt).phi for nt in WH_ntuples_dict.values()], axis = 0)
# (wh) delta_phi_diphoton_W_muon = np.concatenate([nt.phi - get_muon_met_transverse_momentum(nt).phi for nt in WH_ntuples_dict.values()], axis = 0)
# (wh) WH_electron_pt_balance = np.concatenate([(nt.pt - get_electron_met_transverse_momentum(nt).rho) / nt.pt for nt in WH_ntuples_dict.values()], axis = 0)
# (wh) WH_muon_pt_balance = np.concatenate([(nt.pt - get_muon_met_transverse_momentum(nt).rho) / nt.pt for nt in WH_ntuples_dict.values()], axis = 0)
# (wh) Min_DPhi_W_Jets = np.concatenate([nt.Min_DPhi_W_Jets for nt in WH_ntuples_dict.values()], axis = 0)

# (wh) delta_phi_diphoton_W_lepton = np.where(is_electron_event, delta_phi_diphoton_W_electron, delta_phi_diphoton_W_muon)
# (wh) WH_lepton_pt_balance = np.where(is_electron_event, WH_electron_pt_balance, WH_muon_pt_balance)

# ZH_combined array with electron and muon concatenated into lepton.
ZH_combined = np.array(
    np.nan_to_num(
        [
            lead_photon_pt_mgg_ratio, sublead_photon_pt_mgg_ratio, lead_photon_eta, sublead_photon_eta, cos_delta_phi_photons, max_photon_id, min_photon_id, # photons
            # lead_lepton_pt, sublead_lepton_pt, lead_lepton_eta, sublead_lepton_eta, cos_delta_phi_leptons, deltaR_lpll, deltaR_lpsl, deltaR_spll, deltaR_spsl, dileptons_mass, # leptons
            jet_multiplicity, jet0_pt, 
            # jet1_pt, jet2_pt, jet3_pt, 
            # jet_max_btagDeepFlavB, jet_max_btagRobustParTAK4B,
            met_pt, met_sumEt, # MET
            deltaPhi_ggMET, min_delta_phi # topology
        ], nan = 0.0
    )
)
print("===========================================================================")
print(ZH_combined)
print(type(ZH_combined))
print(ZH_combined.shape)
print("===========================================================================")
print("\n")

ZH_combined_transposed = ZH_combined.T
print("output_array:")
print("===========================================================================")
print(ZH_combined_transposed)
print("===========================================================================")
print(ZH_combined_transposed.shape)
print("===========================================================================")
print(weight)
print("===========================================================================")
print(weight.shape)
################################################################################################################################
########## Save the feature arrays and weight array to npy files separately ##########
################################################################################################################################
output_dir = "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/ZH_met_BDT/ZH_met_BDT_features"
os.makedirs(output_dir, exist_ok = True)

np.save(file = os.path.join(output_dir, f"{args.output_filename}.npy"), arr = ZH_combined_transposed)
np.save(file = os.path.join(output_dir, f"{args.output_filename}_weights.npy"), arr = weight)
