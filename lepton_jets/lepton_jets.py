from pathlib import Path
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

def rms(data_y, data_y_comp):
    mask = data_y > 0
    return np.sqrt(np.mean((data_y_comp[mask] / data_y[mask] - 1)**2))

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
        "xlabel": "m_tt, GeV",
        "log_scale": True,
        "xlim": (345, 1600)
    }
]

if __name__ == "__main__":
    output = Path("plots")
    output.mkdir(exist_ok=True)

    for obs in observable:
        pt = df[df["observable"] == obs["name"]]

        x = (pt["bin_high"] + pt["bin_low"]) / 2
        x_err = (pt["bin_high"] - pt["bin_low"]) / 2    

        y = pt["value"]
        y_total = pt["value"] * (pt["total"]/100)

        bin = pt['bin_low']
        bin = np.append(bin, pt['bin_high'].iloc[-1])
        bin_width = bin[1:] - bin[:-1]

        if obs["is_pair"]:
            vals = obs["func"](data_t, data_t_bar)
            vals_comp = obs["func"](data_t_comp, data_t_bar_comp)
            w_comp = weight_t
        else:
            vals = obs["func"](data_all)
            vals_comp = obs["func"](data_all1)
            w_comp = weight

        count, _ = np.histogram(vals, bins=bin)
        data_y = count / (len(vals) * bin_width)

        count_comp, _ = np.histogram(vals_comp, bins=bin, weights=w_comp)
        data_y_comp = count_comp / (np.sum(w_comp) * bin_width)

        print(rms(data_y, data_y_comp))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(obs["xlabel"])
        ax.set_ylabel("sigma * 10^(-3)")

        if obs.get("log_scale"):
            ax.set_yscale("log")
        if "xlim" in obs:
            ax.set_xlim(obs["xlim"])

        ax.errorbar(x, y, xerr=x_err, yerr=y_total, fmt='o', color='black', capsize=0, label='Data')
        ax.stairs(data_y * 1000, bin, color='red', label='MadGraph')
        ax.stairs(data_y_comp * 1000, bin, color='blue', label='MadGraph comp')

        ax.legend(loc='upper right', frameon=False, fontsize=12)

        plt.savefig(output / f"{obs['name']}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
