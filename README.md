# Run3_HiggsDNA_project_GitHub
This GitHub repository stores the 2022 preEE and postEE data and signal & background MC samples, HiggsDNA processor (VHtoLeptonicGGProcessor) and some python scripts that are related to my Master analysis. The operational instruction of HiggsDNA written by me is also included.

Detector: The Compact Muon Solenoid (CMS) at CERN LHC

Target process: WH-leptonic process

HiggsDNA official GitLab: https://gitlab.cern.ch/cms-analysis/general/HiggsDNA

HiggsDNA Tutorial GitLab: https://gitlab.cern.ch/jspah/higgsdna_finalfits_tutorial_24

HiggsDNA operational instruction: https://docs.google.com/presentation/d/1cEex-89CI7qAtrTQWVIiCuZagjnJWtML_Perz7K1D8s/edit?usp=sharing

CMS Data Aggregation Service (DAS): https://cmsweb.cern.ch/das/ (This website provides the data and MC samples one needs and it requires valid CMS VO certificate to access.)

Cross Section Database (XSDB): https://xsecdb-xsdb-official.app.cern.ch/ (This website provides the xsection info of a MC sample.)

NanoAOD content self-documentation: https://cms-nanoaod-integration.web.cern.ch/autoDoc/ (This website provides you the details about what different version of NanoAOD contains.)

Egamma Physics Objects Group (POG): https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaPOG (You can find the info of cut-based and MVA-based ID cut of electrons and photons here.)

Muon POG: https://muon-wiki.docs.cern.ch/guidelines/recommendations/ (You can find the info of cut-based and MVA-based ID cut of muons here.)

High-Level Trigger (HLT): https://twiki.cern.ch/twiki/bin/view/CMS/EgHLTPathDetails (You can find the details of each HLT here.)

# Some workflow (arguments should be modified by case)

## NTuplize nanoAOD files by processor with pre-selection and attach desired information
```
run_analysis.py --json-analysis runner_preEE.json --dump ../NTuples_test01 --executor futures --skipbadfiles --nano-version 13
```
run multiple times with different json file for various samples and era

## Inspecting the NTuples with data/MC comparison
by using Comparison_plot_v2.py (prefer .ipynb)

## Generate data-driven sample (for MET)
```
python gen_data_driven_sample.py
```

## Apply additional selection and information
```
python sel_processor_style_met_BDT.py -i ../NTuples_test01_selected01 -o ../NTuples_test01_selected01_bdt01 -c ./selection_config_met_bdt.json -v --merge-compatible --skip-bdt
```
run multiple times with modified json file for various samples and era

## Prepare ntuple features for BDT training
```
python ZH_met_BDT_training_variables.py -s Diphoton -o Diphoton_train_val
```
run multiple times with different argument -s, -o for each sample

## Start BDT training by BDT classifier, create BDT model json file
```
python ZH_met_XGBoost_classifier.py
```
by default the BDT model json file will be stored at 06_vh_processor/ZH_met_BDT/ZH_met_BDT_classifier_output

## Apply BDT model to the selected NTuples with attaching BDT score
```
python sel_processor_style_met_BDT.py -i ../NTuples_test01_selected01 -o ../NTuples_test01_selected01_bdt01 -c ./selection_config_met_bdt.json -v --merge-compatible --bdt-model ../ZH_met_BDT/ZH_met_BDT_classifier_output/ZH_met_classifier_1500_3_0.05_1_5.json
```

## Inspecting the NTuples with data/MC comparison for different parameters and BDT score
