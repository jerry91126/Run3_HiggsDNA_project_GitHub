#!/usr/bin/env python
"""
Selection script using processor-style metadata handling.
This script applies custom selections to parquet files using the same metadata
approach as the HiggsDNA processor (dump_ak_array style).

The key difference from sel_with_metadata.py:
- Uses string keys for metadata (like the processor does)
- Does NOT explicitly encode to bytes (relies on PyArrow's handling)

Usage:
    python sel_processor_style.py --input /path/to/NTuples --output /path/to/SelectedNTuples --config config.json

Author: Based on HiggsDNA processor dumping_utils.py style
"""

import argparse
import json
import ast
import os
import glob
import pathlib
import awkward as ak
import pyarrow.parquet as pq
import pyarrow as pa
import vector
import numpy as np
import xgboost as xgb
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict

vector.register_awkward()


# =============================================================================
# Utility Functions (Processor Style - from dumping_utils.py)
# =============================================================================

def dump_ak_array_processor_style(
    akarr: ak.Array,
    fname: str,
    location: str,
    metadata: Optional[Dict] = None,
    subdirs: Optional[List[str]] = None,
) -> str:
    """
    Dump an awkward array to disk - using processor style (from dumping_utils.py).
    
    This mirrors the dump_ak_array function used in HiggsDNA processors.
    Metadata keys are passed as strings (not explicitly converted to bytes).
    PyArrow automatically converts string keys to bytes when writing.
    
    Args:
        akarr: Awkward array to save
        fname: Output filename
        location: Output directory
        metadata: Metadata dict (string keys, like processor uses)
        subdirs: Optional subdirectories
        
    Returns:
        Destination path
    """
    subdirs = subdirs or []

    merged_subdirs = "/".join(subdirs)
    if merged_subdirs:
        destination = f"{location.rstrip('/')}/{merged_subdirs}/{fname}"
    else:
        destination = f"{location.rstrip('/')}/{fname}"

    pa_table = ak.to_arrow_table(akarr, extensionarray=False)
    
    # Ensure deterministic column ordering in output parquet files to fix reading issue
    col_names = sorted(pa_table.schema.names)
    pa_table = pa.table([pa_table.column(n) for n in col_names], names=col_names)
    
    # Processor style: merge metadata directly without explicit byte conversion
    # PyArrow will convert string keys to bytes automatically when writing
    if metadata:
        merged_metadata = {**metadata, **(pa_table.schema.metadata or {})}
        pa_table = pa_table.replace_schema_metadata(merged_metadata)

    dirname = os.path.dirname(destination)
    if not os.path.exists(dirname):
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa_table, destination)
    
    return destination


def dump_ak_array_merge_compatible(
    akarr: ak.Array,
    dataset_name: str,
    location: str,
    metadata: Optional[Dict] = None,
    variation: str = "nominal",
) -> str:
    """
    Dump an awkward array in a format compatible with HiggsDNA merge_parquet.py.
    
    This creates the directory structure expected by merge_parquet.py:
    location/dataset_name/variation/filename.parquet
    
    The metadata uses byte keys/values to ensure compatibility with merge_parquet.py
    which reads metadata using byte keys like: metadata[b'sum_genw_presel']
    
    Args:
        akarr: Awkward array to save
        dataset_name: Name of the dataset (becomes directory name)
        location: Output base directory
        metadata: Metadata dict to include
        variation: Systematic variation name (default: "nominal")
        
    Returns:
        Destination path
    """
    # Create directory structure: location/dataset_name/variation/
    output_dir = os.path.join(location, dataset_name, variation)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename similar to processor output
    fname = f"selected_Events_0-{len(akarr)}.parquet"
    destination = os.path.join(output_dir, fname)
    
    pa_table = ak.to_arrow_table(akarr, extensionarray=False)
    
    # Ensure deterministic column ordering
    col_names = sorted(pa_table.schema.names)
    pa_table = pa.table([pa_table.column(n) for n in col_names], names=col_names)
    
    # Convert metadata to bytes for merge_parquet.py compatibility
    # merge_parquet.py reads with: metadata[b'sum_genw_presel']
    if metadata:
        byte_metadata = {}
        for key, value in metadata.items():
            # Convert key to bytes
            if isinstance(key, str):
                key = key.encode('utf-8')
            # Convert value to bytes
            if isinstance(value, str):
                value = value.encode('utf-8')
            elif not isinstance(value, bytes):
                value = str(value).encode('utf-8')
            byte_metadata[key] = value
        
        # Merge with existing table metadata
        if pa_table.schema.metadata:
            byte_metadata = {**pa_table.schema.metadata, **byte_metadata}
        
        pa_table = pa_table.replace_schema_metadata(byte_metadata)
    
    pq.write_table(pa_table, destination)
    
    return destination


def read_parquet_processor_style(path: str):
    """
    Read parquet file(s) and extract metadata (processor style).
    
    This handles both byte and string keys since files may have been
    written with either approach.
    
    Args:
        path: Path to parquet file or directory containing parquet files
        
    Returns:
        tuple: (awkward_array, metadata_dict, custom_accumulator)
    """
    if os.path.isdir(path):
        # Read all parquet files in directory (including in subdirectories like "nominal")
        files = glob.glob(os.path.join(path, "*.parquet"))
        
        # If no files at top level, search in subdirectories (variation folders like nominal/)
        if not files:
            files = glob.glob(os.path.join(path, "*", "*.parquet"))
        
        if not files:
            raise ValueError(f"No parquet files found in {path}")
        
        # Aggregate metadata from all files
        aggregated_metadata = {}
        sum_genw_presel = 0.0
        is_data = False
        custom_accumulator = defaultdict(float)
        
        for f in files:
            table = pq.read_table(f)
            file_metadata = table.schema.metadata or {}
            
            # Handle both byte and string keys for sum_genw_presel
            sum_genw_key = None
            if b'sum_genw_presel' in file_metadata:
                sum_genw_key = b'sum_genw_presel'
            elif 'sum_genw_presel' in file_metadata:
                sum_genw_key = 'sum_genw_presel'
            
            if sum_genw_key:
                val = file_metadata[sum_genw_key]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                if val == "Data":
                    is_data = True
                else:
                    sum_genw_presel += float(val)
            
            # Handle custom accumulator (both byte and string keys)
            acc_key = None
            if b'custom_accumulator' in file_metadata:
                acc_key = b'custom_accumulator'
            elif 'custom_accumulator' in file_metadata:
                acc_key = 'custom_accumulator'
                
            if acc_key:
                acc_val = file_metadata[acc_key]
                if isinstance(acc_val, bytes):
                    acc_val = acc_val.decode('utf-8')
                file_acc = ast.literal_eval(acc_val)
                for key, value in file_acc.items():
                    custom_accumulator[key] += value
        
        # Store aggregated sum_genw_presel (processor style: string keys)
        if is_data:
            aggregated_metadata["sum_genw_presel"] = "Data"
        else:
            aggregated_metadata["sum_genw_presel"] = str(sum_genw_presel) # can also times the scale factor if needed
        
        # Read all data
        data = ak.from_parquet(path)
        
        return data, aggregated_metadata, dict(custom_accumulator) if custom_accumulator else None
    else:
        # Single file
        table = pq.read_table(path)
        file_metadata = table.schema.metadata or {}
        
        # Convert metadata to processor style (string keys)
        metadata = {}
        for key, value in file_metadata.items():
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            val_str = value.decode('utf-8') if isinstance(value, bytes) else value
            metadata[key_str] = val_str
        
        custom_accumulator = None
        if 'custom_accumulator' in metadata:
            custom_accumulator = ast.literal_eval(metadata['custom_accumulator'])
        
        data = ak.from_parquet(path)
        return data, metadata, custom_accumulator


def inspect_parquet(path: str, show_fields: bool = True, show_metadata: bool = True, 
                   show_stats: bool = True, n_events: int = 5):
    """
    Inspect a parquet file or directory.
    
    Args:
        path: Path to parquet file or directory
        show_fields: Show available fields/columns
        show_metadata: Show metadata
        show_stats: Show basic statistics
        n_events: Number of events to preview
    """
    print("=" * 80)
    print(f"Inspecting: {path}")
    print("=" * 80)
    
    data, metadata, custom_acc = read_parquet_processor_style(path)
    
    if show_metadata:
        print("\n[METADATA]")
        print("-" * 40)
        if metadata:
            for key, value in metadata.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"  {key}: {value_str}")
        else:
            print("  No metadata found")
        
        if custom_acc:
            print("\n[CUSTOM ACCUMULATOR]")
            print("-" * 40)
            for key, value in custom_acc.items():
                print(f"  {key}: {value}")
    
    if show_fields:
        print("\n[FIELDS/COLUMNS]")
        print("-" * 40)
        fields = data.fields
        print(f"  Total fields: {len(fields)}")
        for i, field in enumerate(sorted(fields)):
            print(f"  [{i:3d}] {field}")
    
    if show_stats:
        print("\n[STATISTICS]")
        print("-" * 40)
        print(f"  Number of events: {len(data)}")
        
        # Show preview of first few events for key fields
        key_fields = ['mass', 'weight', 'pt', 'lead_pt', 'sublead_pt']
        print(f"\n  Preview (first {n_events} events):")
        for field in key_fields:
            if field in data.fields:
                preview = data[field][:n_events]
                print(f"    {field}: {preview}")
    
    print("=" * 80)
    return data, metadata, custom_acc


# =============================================================================
# Selection Functions
# =============================================================================

def add_kinematics(ntuple: ak.Array) -> ak.Array:
    """
    Add combined lepton 4-momentum variables to the ntuple.
    """
    """
    # Electrons
    electrons0 = vector.zip({
        "pt": ntuple["electron0_pt"], 
        "eta": ntuple["electron0_eta"], 
        "phi": ntuple["electron0_phi"], 
        "mass": ntuple["electron0_mass"]
    })
    electrons0_mask = electrons0.pt > 0
    
    electrons1 = vector.zip({
        "pt": ntuple["electron1_pt"], 
        "eta": ntuple["electron1_eta"], 
        "phi": ntuple["electron1_phi"], 
        "mass": ntuple["electron1_mass"]
    })
    electrons1_mask = electrons1.pt > 0
    
    electron_4momentum = electrons0 + electrons1
    ntuple["electron_pt"] = electron_4momentum.pt
    ntuple["electron_eta"] = electron_4momentum.eta
    ntuple["electron_phi"] = electron_4momentum.phi
    ntuple["electron_mass"] = electron_4momentum.mass
    
    # Muons
    muons0 = vector.zip({
        "pt": ntuple["muon0_pt"], 
        "eta": ntuple["muon0_eta"], 
        "phi": ntuple["muon0_phi"], 
        "mass": ntuple["muon0_mass"]
    })
    muons0_mask = muons0.pt > 0
    
    muons1 = vector.zip({
        "pt": ntuple["muon1_pt"], 
        "eta": ntuple["muon1_eta"], 
        "phi": ntuple["muon1_phi"], 
        "mass": ntuple["muon1_mass"]
    })
    muons1_mask = muons1.pt > 0
    
    muon_4momentum = muons0 + muons1
    ntuple["muon_pt"] = muon_4momentum.pt
    ntuple["muon_eta"] = muon_4momentum.eta
    ntuple["muon_phi"] = muon_4momentum.phi
    ntuple["muon_mass"] = muon_4momentum.mass
    
    # Combined leptons (prioritize electrons, then muons)
    leptons0 = vector.zip({
        "pt": ak.where(electrons0_mask, electrons0.pt, ak.where(muons0_mask, muons0.pt, 0)),
        "eta": ak.where(electrons0_mask, electrons0.eta, ak.where(muons0_mask, muons0.eta, 0)),
        "phi": ak.where(electrons0_mask, electrons0.phi, ak.where(muons0_mask, muons0.phi, 0)),
        "mass": ak.where(electrons0_mask, electrons0.mass, ak.where(muons0_mask, muons0.mass, 0))
    })
    
    leptons1 = vector.zip({
        "pt": ak.where(electrons1_mask, electrons1.pt, ak.where(muons1_mask, muons1.pt, 0)),
        "eta": ak.where(electrons1_mask, electrons1.eta, ak.where(muons1_mask, muons1.eta, 0)),
        "phi": ak.where(electrons1_mask, electrons1.phi, ak.where(muons1_mask, muons1.phi, 0)),
        "mass": ak.where(electrons1_mask, electrons1.mass, ak.where(muons1_mask, muons1.mass, 0))
    })
    
    lepton_4momentum = leptons0 + leptons1
    ntuple["leptons0_pt"] = leptons0.pt
    ntuple["leptons1_pt"] = leptons1.pt
    ntuple["leptons0_eta"] = leptons0.eta
    ntuple["leptons1_eta"] = leptons1.eta
    ntuple["leptons0_phi"] = leptons0.phi
    ntuple["leptons1_phi"] = leptons1.phi
    ntuple["leptons_mass"] = lepton_4momentum.mass
    """

    ntuple["diphoton_CosDeltaPhi"] = np.cos(ntuple["lead_phi"] - ntuple["sublead_phi"])
    ntuple["delta_phi"] = abs(ntuple.phi - ntuple.MET_phi)
    ntuple["lead_photon_pt_mgg_ratio"] = ntuple.lead_pt / ntuple.mass
    ntuple["sublead_photon_pt_mgg_ratio"] = ntuple.sublead_pt / ntuple.mass
    for i in range(4): # Assuming there are at most 4 jets to consider for delta_phi calculation
        ntuple[f"delta_phi_{i}"] = abs(ntuple["MET_phi"] - ntuple[f"Jet{i}_phi"])
    ntuple["min_delta_phi"] = np.minimum(np.minimum(ntuple["delta_phi_0"], ntuple["delta_phi_1"]), np.minimum(ntuple["delta_phi_2"], ntuple["delta_phi_3"]))
    
    return ntuple


def apply_selections(ntuple: ak.Array, selections: Dict[str, Any], verbose: bool = True) -> ak.Array:
    """
    Apply a dictionary of selections to the ntuple.
    
    Args:
        ntuple: Awkward array of events
        selections: Dictionary of selection configurations
        verbose: Print selection statistics
        
    Returns:
        Selected awkward array
    """
    initial_count = len(ntuple)
    
    if verbose:
        print(f"\n[SELECTIONS] Starting with {initial_count} events")
    
    # Apply each selection
    for sel_name, sel_config in selections.items():
        if not sel_config.get("enabled", True):
            continue
        
        # Handle compound selections (OR/AND logic across multiple fields)
        if "compound" in sel_config and sel_config["compound"]:
            logic = sel_config.get("logic", "OR")
            conditions = sel_config.get("conditions", [])
            
            masks = []
            for cond in conditions:
                field = cond.get("field")
                op = cond.get("operator")
                value = cond.get("value")
                
                if op == ">":
                    m = ntuple[field] > value
                elif op == ">=":
                    m = ntuple[field] >= value
                elif op == "<":
                    m = ntuple[field] < value
                elif op == "<=":
                    m = ntuple[field] <= value
                elif op == "==":
                    m = ntuple[field] == value
                elif op == "!=":
                    m = ntuple[field] != value
                elif op == "between":
                    m = (ntuple[field] > value[0]) & (ntuple[field] < value[1])
                elif op == "outside":
                    m = (ntuple[field] < value[0]) | (ntuple[field] > value[1])
                else:
                    raise ValueError(f"Unknown operator: {op}")
                masks.append(m)
            
            # Combine masks with specified logic
            if logic == "OR":
                mask = masks[0]
                for m in masks[1:]:
                    mask = mask | m
            elif logic == "AND":
                mask = masks[0]
                for m in masks[1:]:
                    mask = mask & m
            else:
                raise ValueError(f"Unknown logic: {logic}")
        
        else:
            # Simple single-field selection
            field = sel_config.get("field")
            op = sel_config.get("operator")
            value = sel_config.get("value")
            
            # Build mask based on operator
            if op == ">":
                mask = ntuple[field] > value
            elif op == ">=":
                mask = ntuple[field] >= value
            elif op == "<":
                mask = ntuple[field] < value
            elif op == "<=":
                mask = ntuple[field] <= value
            elif op == "==":
                mask = ntuple[field] == value
            elif op == "!=":
                mask = ntuple[field] != value
            elif op == "between":
                mask = (ntuple[field] > value[0]) & (ntuple[field] < value[1])
            elif op == "outside":
                mask = (ntuple[field] < value[0]) | (ntuple[field] > value[1])
            elif op == "in":
                mask = ak.any([ntuple[field] == v for v in value], axis=0)
            else:
                raise ValueError(f"Unknown operator: {op}")
        
        ntuple = ntuple[mask]
        
        if verbose:
            remaining = len(ntuple)
            eff = remaining / initial_count * 100 if initial_count > 0 else 0
            print(f"  After '{sel_name}': {remaining} events ({eff:.2f}% of initial)")
    
    return ntuple


def get_default_vh_selections() -> Dict[str, Any]:
    """
    Return default VH-leptonic selections as a dictionary.
    
    NOTE: These match the selections in sel.py:
    - event_category == 2 (Two-lepton cut)
    - lead_mvaID > -0.9 AND sublead_mvaID > -0.9 (Photon id cut)
    """
    return {
        "diphoton_mass_range": {
            "enabled": False,  # Disabled to match sel.py (commented out)
            "field": "mass",
            "operator": "between",
            "value": [100, 180],
            "description": "Diphoton mass range"
        },
        "two_lepton_category": {
            "enabled": True,
            "field": "event_category",
            "operator": "==",
            "value": 2,
            "description": "Two-lepton (ZH) category"
        },
        "lead_photon_mva": {
            "enabled": True,
            "field": "lead_mvaID",
            "operator": ">",
            "value": -0.9,
            "description": "Lead photon MVA ID cut"
        },
        "sublead_photon_mva": {
            "enabled": True,
            "field": "sublead_mvaID",
            "operator": ">",
            "value": -0.9,
            "description": "Sublead photon MVA ID cut"
        }
    }


# =============================================================================
# Main Processing Class
# =============================================================================

class ParquetSelectorProcessorStyle:
    """
    Class to handle parquet file selection using processor-style metadata handling.
    
    Key difference from ParquetSelector in sel_with_metadata.py:
    - Uses string keys for metadata (not byte keys)
    - Mirrors the approach used in HiggsDNA processors via dump_ak_array
    """
    
    def __init__(self, input_path: str, output_path: str, verbose: bool = True):
        self.input_path = input_path
        self.output_path = output_path
        self.verbose = verbose
        self.ntuples_dict = {}
        self.metadata_dict = {}  # Stores metadata with string keys (processor style)
        self.custom_acc_dict = {}
        
    def load_ntuples(self, 
                     targeted: List[str] = None, 
                     excluded: List[str] = None,
                     excluded_dirs: List[str] = None):
        """
        Load ntuples from input path.
        
        Args:
            targeted: List of substrings - only load if name contains any of these
            excluded: List of substrings - skip if name contains any of these
            excluded_dirs: List of directory names to skip
        """
        targeted = targeted or []
        excluded = excluded or []
        excluded_dirs = excluded_dirs or ["merged", "root"]
        
        for category in os.listdir(self.input_path):
            category_path = os.path.join(self.input_path, category)
            
            if not os.path.isdir(category_path):
                continue
            
            # Skip excluded directories
            if category in excluded_dirs:
                if self.verbose:
                    print(f"[SKIP] Excluding directory: {category}")
                continue
            
            # Check targeting/exclusion
            if targeted:
                if not any(t in category for t in targeted):
                    if self.verbose:
                        print(f"[SKIP] Not targeted: {category}")
                    continue
            
            if excluded:
                if any(e in category for e in excluded):
                    if self.verbose:
                        print(f"[SKIP] Excluded: {category}")
                    continue
            
            # Load the category
            if self.verbose:
                print(f"[LOAD] Loading: {category}")
            
            try:
                data, metadata, custom_acc = read_parquet_processor_style(category_path)
                self.ntuples_dict[category] = data
                self.metadata_dict[category] = metadata  # String keys (processor style)
                self.custom_acc_dict[category] = custom_acc
            except Exception as e:
                print(f"[ERROR] Failed to load {category}: {e}")
                continue
        
        print(f"\n[INFO] Loaded {len(self.ntuples_dict)} ntuples: {list(self.ntuples_dict.keys())}")
    
    def add_variables(self, add_kin: bool = True, custom_func: Callable = None):
        """
        Add derived variables to all ntuples.
        
        Args:
            add_kin: Add kinematics (combined electron/muon 4-vectors)
            custom_func: Custom function to apply to each ntuple
        """
        for name, ntuple in self.ntuples_dict.items():
            if add_kin:
                try:
                    self.ntuples_dict[name] = add_kinematics(ntuple)
                except Exception as e:
                    print(f"[WARNING] Could not add kinematics to {name}: {e}")
            
            if custom_func:
                self.ntuples_dict[name] = custom_func(self.ntuples_dict[name])
    
    def apply_selections_process(self, selections: Dict[str, Any]):
        """
        Apply selections to all loaded ntuples.
        
        Args:
            selections: Dictionary of selection configurations
        """
        selected_dict = {}
        
        for name, ntuple in self.ntuples_dict.items():
            if self.verbose:
                print(f"\n--- Processing: {name} ---")
            
            selected = apply_selections(ntuple, selections, verbose=self.verbose)
            selected_dict[name] = selected
        
        self.ntuples_dict = selected_dict
    
    def add_BDT_scores(self, model_path: str = None, feature_names: List[str] = None):
        """
        Add BDT scores to all loaded ntuples.
        
        Args:
            model_path: Path to the XGBoost model file (.json or .model)
            feature_names: List of feature names (optional, for verification)
        """
        if model_path is None:
            model_path = "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/ZH_met_BDT/ZH_met_BDT/ZH_leptonic_classifier_1500_3_0.05_1_5.json"
        
        if not os.path.exists(model_path):
            print(f"[WARNING] BDT model not found at {model_path}. Skipping BDT scoring.")
            return
        
        # Load XGBoost model
        print(f"[INFO] Loading BDT model from {model_path}")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        # Add BDT scores to each ntuple
        for name, ntuple in self.ntuples_dict.items():
            if self.verbose:
                print(f"[BDT] Adding BDT scores to {name}...")
            
            # Create features for this ntuple
            features = self._create_features_for_ntuple(ntuple)
            
            # Predict BDT scores
            bdt_scores = model.predict_proba(features)[:, 1]  # Probability of signal class
            
            # Add to ntuple
            ntuple["bdt_score"] = bdt_scores
            
            if self.verbose:
                print(f"  Added BDT scores: min={np.min(bdt_scores):.3f}, max={np.max(bdt_scores):.3f}, mean={np.mean(bdt_scores):.3f}")
        
        print(f"[INFO] BDT scores added to all ntuples")
    
    def _create_features_for_ntuple(self, ntuple: ak.Array) -> np.ndarray:
        """
        Create feature matrix for BDT prediction from a single ntuple.
        This matches the feature_creation() function in significance_and_cumulative_yield_calculation.py.
        
        Creates 20 features exactly as in the reference implementation (lines 406-417).
        
        Args:
            ntuple: Awkward array containing event data
            
        Returns:
            2D numpy array of shape (n_events, 20) ready for BDT prediction
        """
        # Check required fields exist
        required_fields = [
            'lead_pt', 'sublead_pt', 'mass', 'lead_eta', 'sublead_eta',
            'lead_mvaID', 'sublead_mvaID', 'diphoton_CosDeltaPhi', 
            # 'leptons0_pt', 'leptons1_pt', 'leptons0_eta', 'leptons1_eta', 
            # 'leptons_CosDeltaPhi', 'leptons_mass',
            # 'deltaR_leadPhoton_leadLepton', 'deltaR_leadPhoton_subleadLepton',
            # 'deltaR_subleadPhoton_leadLepton', 'deltaR_subleadPhoton_subleadLepton',
            # 'phi', 'leptons_phi'
        ]
        
        missing_fields = [f for f in required_fields if f not in ntuple.fields]
        if missing_fields:
            raise ValueError(f"Missing required fields for BDT: {missing_fields}")
        
        # Feature 1-2: Photon pT / mgg ratio
        lead_photon_pt_mgg_ratio = np.array(ntuple.lead_pt / ntuple.mass)
        sublead_photon_pt_mgg_ratio = np.array(ntuple.sublead_pt / ntuple.mass)
        
        # Feature 3-4: Photon eta
        lead_photon_eta = np.array(ntuple.lead_eta)
        sublead_photon_eta = np.array(ntuple.sublead_eta)
        
        # Feature 5: Cosine delta phi between photons
        cos_delta_phi_photons = np.array(ntuple["diphoton_CosDeltaPhi"])
        
        # Feature 6-7: Photon MVA ID (max and min)
        max_photon_id = np.array(np.maximum(ntuple.lead_mvaID, ntuple.sublead_mvaID))
        min_photon_id = np.array(np.minimum(ntuple.lead_mvaID, ntuple.sublead_mvaID))
        """
        # Feature 8-9: Lepton pT
        lead_lepton_pt = np.array(ntuple["leptons0_pt"])
        sublead_lepton_pt = np.array(ntuple["leptons1_pt"])
        
        # Feature 10-11: Lepton eta
        lead_lepton_eta = np.array(ntuple["leptons0_eta"])
        sublead_lepton_eta = np.array(ntuple["leptons1_eta"])
        
        # Feature 12: Cosine delta phi between leptons
        cos_delta_phi_leptons = np.array(ntuple["leptons_CosDeltaPhi"])
        
        # Feature 13-16: DeltaR between photons and leptons
        deltaR_lpll = np.array(ntuple["deltaR_leadPhoton_leadLepton"])
        deltaR_lpsl = np.array(ntuple["deltaR_leadPhoton_subleadLepton"])
        deltaR_spll = np.array(ntuple["deltaR_subleadPhoton_leadLepton"])
        deltaR_spsl = np.array(ntuple["deltaR_subleadPhoton_subleadLepton"])
        
        # Feature 17: Dilepton mass
        dileptons_mass = np.array(ntuple["leptons_mass"])
        """
        # Feature 8-9: MET variables
        met_pt = np.array(ntuple["MET_pt"])
        met_sumEt = np.array(ntuple["MET_sumEt"])
        # Feature 10: Number of jets
        if "n_jets" in ntuple.fields:
            jet_multiplicity = np.array(ntuple["n_jets"])
        else:
            if self.verbose:
                print(f"  [WARNING] n_jets field not found. Using zeros.")
            jet_multiplicity = np.zeros(len(ntuple))
        
        # Feature 11: Lead jet pT (note: reference uses "Jet0_pt", check both)
        if "Jet0_pt" in ntuple.fields:
            jet0_pt = np.array(ntuple["Jet0_pt"])
        elif "jet0_pt" in ntuple.fields:
            jet0_pt = np.array(ntuple["jet0_pt"])
        else:
            if self.verbose:
                print(f"  [WARNING] Jet0_pt/jet0_pt field not found. Using zeros.")
            jet0_pt = np.zeros(len(ntuple))
        
        # Feature 20: Delta phi between diphoton and dilepton system
        # deltaPhi_ggll = np.array(ntuple["phi"] - ntuple["leptons_phi"])

        # Feature 12-13: MET topology
        deltaPhi_ggMET = np.array(ntuple["delta_phi"])
        min_delta_phi = np.array(ntuple["min_delta_phi"])
        pt_balance_met = np.array((ntuple.pt - ntuple.MET_pt) / ntuple.pt)
        
        # Stack all 20 features into a 2D array (n_events, 20)
        # Order matches lines 406-417 in significance_and_cumulative_yield_calculation.py
        ## Warning: The order and exact features must match the reference in the BDT classifier
        features = np.column_stack([
            lead_photon_pt_mgg_ratio,      # Feature 1
            sublead_photon_pt_mgg_ratio,   # Feature 2
            lead_photon_eta,                # Feature 3
            sublead_photon_eta,             # Feature 4
            cos_delta_phi_photons,          # Feature 5
            max_photon_id,                  # Feature 6
            min_photon_id,                  # Feature 7
            # lead_lepton_pt,                 # Feature 8
            # sublead_lepton_pt,              # Feature 9
            # lead_lepton_eta,                # Feature 10
            # sublead_lepton_eta,             # Feature 11
            # cos_delta_phi_leptons,          # Feature 12
            # deltaR_lpll,                    # Feature 13
            # deltaR_lpsl,                    # Feature 14
            # deltaR_spll,                    # Feature 15
            # deltaR_spsl,                    # Feature 16
            # dileptons_mass,                 # Feature 17
            jet_multiplicity,               # Feature 8  (matches var list order in classifier)
            jet0_pt,                        # Feature 9
            met_pt,                         # Feature 10
            met_sumEt,                      # Feature 11
            # deltaPhi_ggll                   # Feature 20
            deltaPhi_ggMET,                 # Feature 12
            min_delta_phi,                   # Feature 13
            pt_balance_met                   # Feature 14
        ])
        
        # Handle NaN values (match reference: np.nan_to_num(..., nan=0.0))
        features = np.nan_to_num(features, nan=0.0)
        
        if self.verbose:
            print(f"  Created feature matrix with shape: {features.shape} (expected: (n_events, 20))")
        
        return features

    def save_selected(self, suffix: str = "_selected", merge_compatible: bool = False, variation: str = "nominal"):
        """
        Save all selected ntuples with preserved metadata.
        
        Args:
            suffix: Suffix to add to output filenames (used when merge_compatible=False)
            merge_compatible: If True, output in directory structure compatible with merge_parquet.py
                             If False, output as single parquet files (for plotting)
            variation: Systematic variation name (used when merge_compatible=True)
        """
        os.makedirs(self.output_path, exist_ok=True)
        
        for name, ntuple in self.ntuples_dict.items():
            # Get original metadata
            metadata = self.metadata_dict.get(name, {})
            
            # Add custom accumulator to metadata if present
            custom_acc = self.custom_acc_dict.get(name, None)
            if custom_acc:
                metadata["custom_accumulator"] = str(custom_acc)
            
            if merge_compatible:
                # Save in directory structure for merge_parquet.py compatibility
                # Structure: output_path/dataset_name/variation/filename.parquet
                full_path = dump_ak_array_merge_compatible(
                    akarr=ntuple,
                    dataset_name=name,
                    location=self.output_path,
                    metadata=metadata,
                    variation=variation
                )
            else:
                # Save as single file for plotting
                output_filename = f"{name}{suffix}.parquet"
                full_path = dump_ak_array_processor_style(
                    akarr=ntuple,
                    fname=output_filename,
                    location=self.output_path,
                    metadata=metadata
                )
            
            if self.verbose:
                print(f"[SAVED] {full_path} ({len(ntuple)} events)")
        
        if merge_compatible:
            print(f"\n[INFO] All selected ntuples saved to {self.output_path} (merge_parquet.py compatible)")
        else:
            print(f"\n[INFO] All selected ntuples saved to {self.output_path} (single files for plotting)")
    
    def get_summary(self):
        """
        Print summary of loaded ntuples.
        """
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        total_events = 0
        for name, ntuple in self.ntuples_dict.items():
            n_events = len(ntuple)
            total_events += n_events
            
            # Processor style: string keys
            metadata = self.metadata_dict.get(name, {})
            sum_genw = metadata.get("sum_genw_presel", "N/A")
            
            print(f"  {name}: {n_events} events, sum_genw_presel={sum_genw}")
        
        print(f"\nTotal: {total_events} events across {len(self.ntuples_dict)} samples")
        print("=" * 80)


# =============================================================================
# Main Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Apply selections to parquet files using processor-style metadata handling."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing NTuples"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for selected NTuples"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to JSON config file with selections"
    )
    parser.add_argument(
        "--targeted",
        type=str,
        nargs="+",
        default=None,
        help="Only process ntuples containing these substrings"
    )
    parser.add_argument(
        "--excluded",
        type=str,
        nargs="+",
        default=None,
        help="Skip ntuples containing these substrings"
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="Inspect a specific parquet file/directory and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output"
    )
    parser.add_argument(
        "--merge-compatible",
        action="store_true",
        default=False,
        dest="merge_compatible",
        help="Output in directory structure compatible with merge_parquet.py (dataset/variation/file.parquet)"
    )
    parser.add_argument(
        "--variation",
        type=str,
        default="nominal",
        help="Systematic variation name for merge-compatible output (default: nominal)"
    )
    parser.add_argument(
        "--bdt-model",
        type=str,
        default=None,
        dest="bdt_model",
        help="Path to XGBoost BDT model file (default: auto-detect from standard location)"
    )
    parser.add_argument(
        "--skip-bdt",
        action="store_true",
        default=False,
        dest="skip_bdt",
        help="Skip BDT scoring step"
    )
    
    args = parser.parse_args()
    
    # Inspect mode
    if args.inspect:
        inspect_parquet(args.inspect)
        return
    
    # Load config from JSON or use defaults
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        selections = config.get("selections", {})
        print(f"[INFO] Loaded selections from {args.config}")
    else:
        selections = get_default_vh_selections()
        print("[INFO] Using default VH-leptonic selections")
    
    # Get targeting options: command line args override JSON config
    targeting_config = config.get("targeting", {})
    
    # Priority: command line > JSON config > empty list
    targeted = args.targeted if args.targeted else targeting_config.get("targeted_ntuples", None)
    excluded = args.excluded if args.excluded else targeting_config.get("excluded_ntuples", None)
    excluded_dirs = targeting_config.get("excluded_dirs", ["merged", "root"])
    
    if targeted:
        print(f"[INFO] Targeting ntuples containing: {targeted}")
    if excluded:
        print(f"[INFO] Excluding ntuples containing: {excluded}")
    print(f"[INFO] Excluding directories: {excluded_dirs}")
    if args.merge_compatible:
        print(f"[INFO] Output format: merge_parquet.py compatible (variation: {args.variation})")
    else:
        print("[INFO] Output format: single parquet files (for plotting)")
    
    # Create selector (processor style)
    selector = ParquetSelectorProcessorStyle(
        input_path=args.input,
        output_path=args.output,
        verbose=args.verbose
    )
    
    # Load ntuples
    selector.load_ntuples(
        targeted=targeted,
        excluded=excluded,
        excluded_dirs=excluded_dirs
    )
    
    # Add derived variables
    selector.add_variables(add_kin=True)
    
    # Add BDT scores
    if not args.skip_bdt:
        selector.add_BDT_scores(model_path=args.bdt_model)
    else:
        print("[INFO] Skipping BDT scoring (--skip-bdt flag set)")

    # Apply selections
    selector.apply_selections_process(selections)
    
    # Save selected ntuples
    selector.save_selected(
        merge_compatible=args.merge_compatible,
        variation=args.variation
    )
    
    # Print summary
    selector.get_summary()


if __name__ == "__main__":
    main()
