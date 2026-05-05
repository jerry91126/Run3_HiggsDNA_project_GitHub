import os
import sys
import argparse
import numpy as np
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost
import matplotlib.pyplot as plt
import mplhep as hep

# Parse command line arguments
parser = argparse.ArgumentParser(description='XGBoost BDT Training')
parser.add_argument('--outdir', type=str, 
                    default='/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/ZH_met_BDT/ZH_met_BDT',
                    help='Output directory for plots and models')
args = parser.parse_args()

# variables abbreviations:
# ld = lead, sld = sublead, M = diphoton invariant mass, id = MVA ID, Pho = photon, Lep = lepton,

var = [
    "lead_photon_pt_mgg_ratio",       # f0
    "sublead_photon_pt_mgg_ratio",    # f1
    "lead_photon_eta",                # f2
    "sublead_photon_eta",             # f3
    "cos_delta_phi_photons",          # f4
    "max_photon_id",                  # f5
    "min_photon_id",                  # f6
    "jet_multiplicity",               # f7
    "jet0_pt",                        # f8
    "met_pt",                         # f9
    "met_sumEt",                      # f10
    "deltaPhi_ggMET",                 # f11
    "min_delta_phi"                   # f12   
]

outdir = args.outdir
os.makedirs(outdir, exist_ok=True)
print(f"Output directory: {outdir}")

####################################################################################
###### Background npy for training and validation ######
####################################################################################
# Each sample except ggH is split into 70% for training and 30% for validation.
basepath = "/eos/home-h/hshsu/pract1/higgsdna_finalfits_tutorial_24/06_vh_processor/ZH_met_BDT/ZH_met_BDT_features"
########## Diphoton ##########
Diphoton_train_val_path = f"{basepath}/Diphoton_train_val.npy"
Diphoton_weight_path = f"{basepath}/Diphoton_train_val_weights.npy"
########## Drell-Yan jets ##########
DYJets_train_val_path = f"{basepath}/DYJets_train_val.npy"
DYJets_weight_path = f"{basepath}/DYJets_train_val_weights.npy"
########## gamma + jets ##########
# (temp) GJets_train_val_path = f"{basepath}/GJets_train_val.npy"
# (temp) GJets_weight_path = f"{basepath}/GJets_train_val_weights.npy"
########## QCD ##########
# (temp) QCD_train_val_path = f"{basepath}/QCD_train_val.npy"
# (temp) QCD_weight_path = f"{basepath}/QCD_train_val_weights.npy"
########## Top quarks ##########
Top_train_val_path = f"{basepath}/Top_train_val.npy"
Top_weight_path = f"{basepath}/Top_train_val_weights.npy"
########## VV ##########
VV_train_val_path = f"{basepath}/VV_train_val.npy"
VV_weight_path = f"{basepath}/VV_train_val_weights.npy"
########## VG ###########
VG_train_val_path = f"{basepath}/VG_train_val.npy"
VG_weight_path = f"{basepath}/VG_train_val_weights.npy"
########## W + gamma ##########
# (temp) WG_train_val_path = f"{basepath}/WG_train_val.npy"
# (temp) WG_weight_path = f"{basepath}/WG_train_val_weights.npy"
########## Z + gamma ##########
# (temp) ZG_train_val_path = f"{basepath}/ZG_train_val.npy"
# (temp) ZG_weight_path = f"{basepath}/ZG_train_val_weights.npy"
########## data-driven ##########
data_driven_train_val_path = f"{basepath}/data_driven_train_val.npy"
data_driven_weight_path = f"{basepath}/data_driven_train_val_weights.npy"
####################################################################################
########## ggH ##########
# (temp) ggH_train_path = f"{basepath}/ggH_train.npy"
# (temp) ggH_train_weight_path = f"{basepath}/ggH_train_weights.npy"

""" To make the matters simpler, we do not use the separate set for ggH validation. Instead, we split the ggH_train sample into 70% for training and 30% for validation,
just like the other samples. """
# ggH_val_path = f"{basepath}/ggH_val.npy"
# ggH_val_weight_path = f"{basepath}/ggH_val_weights.npy"
########## VBF ##########
# (temp) VBF_train_val_path = f"{basepath}/VBF_train_val.npy"
# (temp) VBF_weight_path = f"{basepath}/VBF_train_val_weights.npy"
########## VH-hadronic (W -> 2Q or Z -> 2Q) ##########
# (temp) VH_bkg_train_val_path = f"{basepath}/VH_bkg_train_val.npy"
# (temp) VH_bkg_weight_path = f"{basepath}/VH_bkg_train_val_weights.npy"
########## ZH-leptonic (Z -> 2L or Z -> LNu) ##########
# (temp) ZH_Zto2L_signal_train_val_path = f"{basepath}/ZH_Zto2L_signal_train_val.npy"
# (temp) ZH_Zto2L_signal_weight_path = f"{basepath}/ZH_Zto2L_signal_train_val_weights.npy"
ZH_Zto2Nu_signal_train_val_path = f"{basepath}/ZH_Zto2Nu_signal_train_val.npy"
ZH_Zto2Nu_signal_weight_path = f"{basepath}/ZH_Zto2Nu_signal_train_val_weights.npy"
########## ttH ##########
# (temp) ttH_train_val_path = f"{basepath}/ttH_train_val.npy"
# (temp) ttH_weight_path = f"{basepath}/ttH_train_val_weights.npy"
####################################################################################
###### Signal npy for training and validation ######
####################################################################################
# (temp) WH_signal_train_val_path = f"{basepath}/WH_signal_train_val.npy"
# (temp) WH_signal_weight_path = f"{basepath}/WH_signal_train_val_weights.npy"
####################################################################################
###### Training and test sample preparation ######
####################################################################################
X_sig = np.load(ZH_Zto2Nu_signal_train_val_path)[ : , : ]
W_sig = np.load(ZH_Zto2Nu_signal_weight_path)

X_bkg = np.concatenate([
    np.load(Diphoton_train_val_path)[ : , : ],
    np.load(DYJets_train_val_path)[ : , : ],
    # (temp) np.load(GJets_train_val_path)[ : , : ],
    np.load(Top_train_val_path)[ : , : ],
    np.load(VV_train_val_path)[ : , : ],
    np.load(VG_train_val_path)[ : , : ],
    np.load(data_driven_train_val_path)[ : , : ],
    # (temp) np.load(WG_train_val_path)[ : , : ],
    # (temp) np.load(ZG_train_val_path)[ : , : ],
    # (temp) np.load(VH_bkg_train_val_path)[ : , : ],
    # (temp) np.load(ttH_train_val_path)[ : , : ],
    # (temp) np.load(ggH_train_path)[ : , : ]
])
W_bkg = np.concatenate([
    np.load(Diphoton_weight_path),
    np.load(DYJets_weight_path),
    # (temp) np.load(GJets_weight_path),
    np.load(Top_weight_path),
    np.load(VV_weight_path),
    np.load(VG_weight_path),
    np.load(data_driven_weight_path),
    # (temp) np.load(WG_weight_path),
    # (temp) np.load(ZG_weight_path),
    # (temp) np.load(VH_bkg_weight_path),
    # (temp) np.load(ttH_weight_path),
    # (temp) np.load(ggH_train_weight_path)
])

X_sig_train, X_sig_val, W_sig_train, W_sig_val = train_test_split(
    X_sig, W_sig, test_size = 0.3, random_state = 69
)
X_bkg_train, X_bkg_val, W_bkg_train, W_bkg_val = train_test_split(
    X_bkg, W_bkg, test_size = 0.3, random_state = 69
)

Y_sig_train = np.ones(len(X_sig_train))
Y_sig_val = np.ones(len(X_sig_val))
Y_bkg_train = np.zeros(len(X_bkg_train))
Y_bkg_val = np.zeros(len(X_bkg_val))

X_train = np.concatenate((X_sig_train, X_bkg_train), axis = 0)
Y_train = np.concatenate((Y_sig_train, Y_bkg_train), axis = 0)

X_val = np.concatenate((X_sig_val, X_bkg_val), axis = 0)
Y_val = np.concatenate((Y_sig_val, Y_bkg_val), axis = 0)

# Since XGBoost accepts only positive weights, we take the absolute value of the weights for both training and validation.
W_sig_train_pos = np.abs(W_sig_train)
W_sig_val_pos = np.abs(W_sig_val)
W_bkg_train_pos = np.abs(W_bkg_train)
W_bkg_val_pos = np.abs(W_bkg_val)

W_sig_train_sum = np.sum(W_sig_train_pos)
W_sig_val_sum = np.sum(W_sig_val_pos)
W_bkg_train_sum = np.sum(W_bkg_train_pos)
W_bkg_val_sum = np.sum(W_bkg_val_pos)

train_ratio = W_bkg_train_sum / W_sig_train_sum
val_ratio = W_bkg_val_sum / W_sig_val_sum

W_sig_train_scaled = W_sig_train_pos * train_ratio
W_sig_val_scaled = W_sig_val_pos * val_ratio

W_train = np.concatenate((W_sig_train_scaled, W_bkg_train_pos), axis = 0)
W_val = np.concatenate((W_sig_val_scaled, W_bkg_val_pos), axis = 0)

####################################################################################
###### XGBoost training engine ######
####################################################################################
# XGBoost sklearn configuration
n_estimators = 1500
max_depth = 3
learning_rate = 0.05
early_stopping_rounds = 1

XGBEngine = xgboost.XGBClassifier(
    n_estimators  = n_estimators,
    learning_rate = learning_rate,
    max_depth     = max_depth,
    tree_method   = "hist",
    device        = "cpu",  # "(temp) "cuda" encounter the virtual memory issue. | just say "cuda", not "cuda:0" for sklearn API
    eval_metric   = ["logloss"],  # can be a string if single metric
    early_stopping_rounds = early_stopping_rounds,
    scale_pos_weight = 1,
    # use_label_encoder=False  # recommended for sklearn >= 1.0
)

# Make sure labels are consistent with the variables used above
eval_set = [(X_train, Y_train), (X_val, Y_val)]

# Train the model
XGBEngine.fit(
    X_train, Y_train,
    sample_weight = W_train,
    eval_set = eval_set,
    sample_weight_eval_set = [W_train, W_val],
    verbose  = True
    # (using features name) feature_names = var  # Use the variable names from var list
)

# Retrieve training history
results = XGBEngine.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

version = 5
suffix = f'{n_estimators}_{max_depth}_{learning_rate}_{early_stopping_rounds}_{version}'

hep.style.use("CMS")
# Plot training/validation logloss
fig, ax = plt.subplots(figsize=(14, 14))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train', linewidth = 3, color = "blue")
ax.plot(x_axis, results['validation_1']['logloss'], label='Validation', linewidth = 3, color = "orange")
ax.legend()
ax.set_ylabel(ylabel = 'Log Loss', fontsize = "medium")
ax.set_xlabel(xlabel = 'Number of Trees', fontsize = "medium")
ax.set_yscale('log')
ax.set_title(label = 'XGBoost Log Loss', fontsize = "large")
plt.savefig(f'{outdir}/logloss_{suffix}.png')
# plt.savefig(f'{outdir}/logloss_{suffix}.pdf')

####################################################################################
###### Training Accuracy ######
####################################################################################
from sklearn import metrics
train_accuracy = metrics.accuracy_score(Y_train, XGBEngine.predict(X_train))
val_accuracy   = metrics.accuracy_score(Y_val,   XGBEngine.predict(X_val))
print(f"Training Accuracy = {train_accuracy:.4f}")
print(f"Validation Accuracy = {val_accuracy:.4f}")

####################################################################################
###### ROC curve ######
####################################################################################

# Predict probabilities on train/val
y_train_pred_proba = XGBEngine.predict_proba(X_train)[:, 1]
y_val_pred_proba   = XGBEngine.predict_proba(X_val)[:, 1]

# ROC for train
fpr_train, tpr_train, _ = metrics.roc_curve(Y_train, y_train_pred_proba, pos_label=1, sample_weight = W_train)
roc_auc_train = metrics.auc(fpr_train, tpr_train)
rfpr_train = [1 - i for i in fpr_train]

# ROC for val
fpr_val, tpr_val, _ = metrics.roc_curve(Y_val, y_val_pred_proba, pos_label=1, sample_weight = W_val)
roc_auc_val = metrics.auc(fpr_val, tpr_val)
rfpr_val = [1 - i for i in fpr_val]

# np.savez(f'{outdir}/xgb_valroc.npz', tpr=tpr_val, fpr=fpr_val, auc=np.array([roc_auc_val], dtype=np.float32))

# Plot ROC
fig, ax = plt.subplots()
ax.set_title(label = 'Receiver Operating Characteristic', fontsize = "large")
ax.plot(tpr_train, np.array(rfpr_train), 'b', label=f'AUC (Train) = {roc_auc_train:.2f}')
ax.plot(tpr_val,   np.array(rfpr_val),   'g', label=f'AUC (Val)   = {roc_auc_val:.2f}')
ax.legend(loc='lower left')
ax.plot([0, 1], [1, 0], '--', color='gray')
ax.set_ylabel(ylabel = 'Signal Efficiency', fontsize = "medium")
ax.set_xlabel(xlabel = 'Background Rejection', fontsize = "medium")
plt.savefig(f'{outdir}/roc_{suffix}.png')
# plt.savefig(f'{outdir}/roc_{suffix}.pdf')

####################################################################################
###### Histogram comparison (validation set only) ######
####################################################################################
# Plot predicted scores for val signal vs background (reconstructed from Y_val)
sig_mask = Y_val == 1
bkg_mask = Y_val == 0
fig, ax = plt.subplots()
ax.hist(y_val_pred_proba[sig_mask], bins=60, label='Signal (Val)', histtype='step', weights = W_val[sig_mask], linewidth = 3, color = "blue")
ax.hist(y_val_pred_proba[bkg_mask], bins=60, label='Background (Val)', histtype='step', weights = W_val[bkg_mask], linewidth = 3, color = "orange")
ax.set_yscale('log')
ax.set_xlabel(xlabel = "BDT response", fontsize = "medium")
ax.legend(loc='upper right')
plt.savefig(f'{outdir}/hist_{suffix}.png')
# plt.savefig(f'{outdir}/hist_{suffix}.pdf')

####################################################################################
###### Feature importance ######
####################################################################################
fig, ax = plt.subplots(figsize=(12, 12))
xgboost.plot_importance(XGBEngine, ax=ax, importance_type='weight')
# (using features name) fig, ax = plt.subplots(figsize=(14, 10))
# (using features name) xgboost.plot_importance(XGBEngine, ax=ax, importance_type='gain')
# (using features name) ax.set_xlabel('Gain', fontsize='medium')
# (using features name) ax.set_ylabel('Features', fontsize='medium')
plt.tight_layout()
plt.savefig(f'{outdir}/importance_{suffix}.png')
# plt.savefig(f'{outdir}/importance_{suffix}.pdf')

XGBEngine.save_model(f"{outdir}/ZH_leptonic_classifier_{suffix}.json")

# ####################################################################################
# ###### Loss Function ######
# ####################################################################################
# results = XGBEngine.evals_result()
# epochs = len(results['validation_0']['logloss'])
# x_axis = range(0, epochs)

# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')
# ax.legend()
# ax.set_ylabel('Log Loss')
# ax.set_yscale('log')
# ax.set_title('XGBoost Log Loss')
# plt.savefig(f'{outdir}/logloss.png')
# plt.savefig(f'{outdir}/logloss.pdf')
# ####################################################################################
# ###### Training Accuracy ######
# ####################################################################################
# train_accuracy = metrics.accuracy_score(Y_train, XGBEngine.predict(X_train))
# val_accuracy   = metrics.accuracy_score(Y_val,   XGBEngine.predict(X_val))
# print(f"Training Accuracy = {train_accuracy:.4f}")
# print(f"Validation Accuracy = {val_accuracy:.4f}")

# ####################################################################################
# ###### ROC curve ######
# ####################################################################################

# # Predict probabilities
# y_sig_train_pred = XGBEngine.predict_proba(X_sig_train)[:, 1]
# y_bkg_train_pred = XGBEngine.predict_proba(X_bkg_train)[:, 1]
# y_sig_val_pred   = XGBEngine.predict_proba(X_sig_val)[:, 1]
# y_bkg_val_pred   = XGBEngine.predict_proba(X_bkg_val)[:, 1]

# # ROC for train
# fpr_train, tpr_train, _ = metrics.roc_curve(
#     y_train, np.concatenate((y_sig_train_pred, y_bkg_train_pred)), pos_label=1)
# roc_auc_train = metrics.auc(fpr_train, tpr_train)
# rfpr_train = [1 - i for i in fpr_train]

# # ROC for validation
# fpr_val, tpr_val, _ = metrics.roc_curve(
#     y_val, np.concatenate((y_sig_val_pred, y_bkg_val_pred)), pos_label=1)
# roc_auc_val = metrics.auc(fpr_val, tpr_val)
# rfpr_val = [1 - i for i in fpr_val]

# np.savez(f'{outdir}/xgb_valroc.npz', tpr=tpr_val, fpr=fpr_val, auc=np.array([roc_auc_val], dtype=np.float32))

# # Plot ROC
# fig, ax = plt.subplots()
# ax.set_title('Receiver Operating Characteristic')
# ax.plot(tpr_train, np.array(rfpr_train), 'b', label=f'AUC (Train) = {roc_auc_train:.2f}')
# ax.plot(tpr_val,   np.array(rfpr_val),   'g', label=f'AUC (Val)   = {roc_auc_val:.2f}')
# ax.legend(loc='lower right')
# ax.plot([0, 1], [1, 0], '--', color='gray')
# ax.set_ylabel('Signal Efficiency')
# ax.set_xlabel('Background Rejection')
# plt.savefig(f'{outdir}/roc.png')
# plt.savefig(f'{outdir}/roc.pdf')

# ####################################################################################
# ###### Histogram comparison (validation set only) ######
# ####################################################################################
# fig, ax = plt.subplots()
# ax.hist(y_sig_val_pred, bins=60, label='Signal (Val)', histtype='step')
# ax.hist(y_bkg_val_pred, bins=60, label='Background (Val)', histtype='step')
# ax.set_yscale('log')
# ax.legend(loc='upper right')
# plt.savefig(f'{outdir}/hist.png')
# plt.savefig(f'{outdir}/hist.pdf')

# ####################################################################################
# ###### Feature importance ######
# ####################################################################################
# fig, ax = plt.subplots(figsize=(10, 8))
# xgboost.plot_importance(XGBEngine, ax=ax)
# plt.tight_layout()
# plt.savefig(f'{outdir}/importance.png')
# plt.savefig(f'{outdir}/importance.pdf')
