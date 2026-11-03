# Uso: python PlottingCSV.py --file "acquisizioni/Volt/acquisizione_python(1).csv"
# %%
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

NUM_OF_SENSORS = 8

# Usato quando si eseguono le celle in modalità interattiva: modificare qui il file da plottare
DEFAULT_FILE = Path("acquisizioni/Volt/acquisizione_python(1).csv")

parser = argparse.ArgumentParser(description="Plot di segnali e forza ricostruita", allow_abbrev=False)
parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="CSV da plottare (default: %(default)s)")
# parse_known_args ignora gli argomenti passati dal kernel Jupyter
args, _ = parser.parse_known_args()
file_path = args.file

# --- Read CSV ---
df = pd.read_csv(file_path, sep="\t", header=None, decimal=",")
df = df.apply(pd.to_numeric, errors="coerce")

# --- Remove first samples ---
df = df.iloc[100:].reset_index(drop=True)

# --- Extract timestamp and sensors ---
timestamp = df.iloc[:, 0]
signals = df.iloc[:, 1:NUM_OF_SENSORS+1]   # solo gli 8 sensori
force = df.iloc[:, (NUM_OF_SENSORS+1) : (2 * NUM_OF_SENSORS + 1)]  # solo le forze

# --- Relative time ---
time_axis = (timestamp - timestamp.iloc[0]) / 1e6


# --- Plot ---
fig, ax = plt.subplots(figsize=(12,6))
colori = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'] # Lista di colori standard per distinguere gli 8 sensori

for i in range(NUM_OF_SENSORS):
    ax.plot(time_axis, signals.iloc[:, i], color=colori[i], label=f"Sensor {i+1}")
    ax.plot(time_axis, force.iloc[:, i], color=colori[i], label=f"Force {i+1}", linestyle='--')

# Asse Y dinamico (si adatta al massimo valore registrato tra tutti i sensori)
ax.set_xlabel("Time [s]", fontsize=11)
ax.set_ylabel("Voltage / Normalized Unit", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.6)
max_volt_reale = signals.values.max()   # valore massimo assoluto registrato tra tutti i sensori nei dati grezzi
ax.set_ylim(-0.1, max_volt_reale * 1.1) #limite sup con 10% margine, limite inf leggermente sotto zero per vedere base onde quadre


#Legenda
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0), # 1.02 significa appena fuori dal bordo destro
    fontsize=8.5,
    ncol=1, # Sviluppata in verticale, una colonna è perfetta per il lato destro
    title="Legenda Segnali\n(Continuo = Volt | Tratteggiato = Forza)"
)

plt.title("Segnale Acquisizione vs Forza Ricostruita", fontsize=14, pad=15)
plt.tight_layout()
plt.show()
# %%
