import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial import cKDTree
import uproot

df = pd.read_csv("cms_table4_ljets.csv")

df["value"] = df["value"] * 1000

data = uproot.concatenate({f: 'tree' for f in [
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/00000/ttbarSel_merged.root',
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/00001/ttbarSel_merged.root',
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010000/ttbarSel_merged.root',
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010001/ttbarSel_merged.root',
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010002/ttbarSel_merged.root',
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010003/ttbarSel_merged.root',
    ]}, ['mcEventType', 'mcT', 'mcTbar'], library='np')


data_t = data['mcT']
data_t_bar = data['mcTbar']
data_all = np.concatenate([data['mcT'], data['mcTbar']])

n = data_all.shape[0]
n1 = int(n/10)

data_all1 = data_all[:n1].copy()

data_a = np.hstack([data["mcT"], data["mcTbar"]])

n_a = data_a.shape[0]
n1_a = int(n_a/10)

data_a1 = data_a[:n1_a].copy()

def nn(x1, x2):
  # найближчий сусід
  tree = cKDTree(x1)
  _, i = tree.query(x2)
  # ваги
  weights = np.bincount(i, minlength=len(x1))
  sum_x2 = np.zeros_like(x1)
  np.add.at(sum_x2, i, x2)
  
  mask = weights > 0
  x1_corr = x1.copy()
  x1_corr[mask] = sum_x2[mask] / weights[mask, None]
  return x1, weights, x1_corr

_, weight, data_all1 = nn(data_all1, data_all)
_, weight, data_all1 = nn(data_all1, data_all)
_, weight, data_all1 = nn(data_all1, data_all)

_, weight_t, data_a1 = nn(data_a1, data_a)
_, weight_t, data_a1 = nn(data_a1, data_a)
_, weight_t, data_a1 = nn(data_a1, data_a)

data_t_comp = data_a1[:, :4]
data_t_bar_comp = data_a1[:, 4:]


def pT(p):
    return np.sqrt(p[:, 0]**2 + p[:, 1]**2)

def rapidity(p):
    E = np.sqrt(p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2 )
    return 0.5*np.log((E + p[:, 2]) / ((E - p[:, 2])))

def pT_tt(pT_t, pT_tbar):
    return np.sqrt((pT_t[:, 0] + pT_tbar[:, 0])**2 + (pT_t[:, 1] + pT_tbar[:, 1])**2)

def rapidity_tt(p, p_bar):
    E_t = np.sqrt(p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2)
    E_t_bar = np.sqrt(p_bar[:, 0]**2 + p_bar[:, 1]**2 + p_bar[:, 2]**2 + p_bar[:, 3]**2)
    return 0.5*np.log((E_t + E_t_bar + p[:, 2] + p_bar[:, 2]) / (E_t + E_t_bar - p[:, 2]- p_bar[:, 2]))

def mass(p, p_bar):
    E_t = np.sqrt(p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2)
    E_t_bar = np.sqrt(p_bar[:, 0]**2 + p_bar[:, 1]**2 + p_bar[:, 2]**2 + p_bar[:, 3]**2)
    p_tt = ((p[:, 0] + p_bar[:, 0])**2 + (p[:, 1] + p_bar[:, 1])**2 + (p[:, 2] + p_bar[:, 2])**2)

    return np.sqrt((E_t + E_t_bar)**2 - p_tt)

observable = [
    {
        "name": "pT_t",
        "func": pT,
        "is_pair":  False,
        "xlabel": "pT_t, GeV"  
    },
    {
        "name": "y_t",
        "func": rapidity,
        "is_pair":  False,
        "xlabel": "y_t"
    },
    {
        "name": "pT_tt",
        "func": pT_tt,
        "is_pair":  True,
        "xlabel": "pT_tt, GeV"
    },
    {
        "name": "y_tt",
        "func": rapidity_tt,
        "is_pair":  True,
        "xlabel": "y_tt"
    },
    {
        "name": "m_tt",
        "func": mass,
        "is_pair":  True,
        "xlabel": "m_tt, GeV"
    }
]
