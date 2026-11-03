# Uso: python SerialCommunication.py --port COM3 [--output-dir acquisizioni]
import argparse
import csv
import struct
import time
from pathlib import Path

import serial

# ==========================
# CONFIGURAZIONE
# ==========================

parser = argparse.ArgumentParser(description="Acquisizione seriale di segnali e forza ricostruita")
parser.add_argument("--port", required=True, help="porta seriale, es. COM3 o /dev/ttyACM0")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("acquisizioni"),
    help="cartella in cui creare Raw/ e Volt/ (default: %(default)s)",
)
args = parser.parse_args()

PORT = args.port
BAUDRATE = 3000000
TIMEOUT = 1

VREF = 4.096
ADC_MAX = 65535
FORCE_SCALE = 65535.0
N_FORCE_SENSORS = 8

N_SAMPLES = 4000

# Primo numero libero, per non sovrascrivere le acquisizioni precedenti
VOLT_DIR = args.output_dir / "Volt"
RAW_DIR = args.output_dir / "Raw"
VOLT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

n = 1
while (VOLT_DIR / f"acquisizione_python({n}).csv").exists() or (
    RAW_DIR / f"acquisizione_python_raw({n}).csv"
).exists():
    n += 1

VOLT_FILE = VOLT_DIR / f"acquisizione_python({n}).csv"
RAW_FILE = RAW_DIR / f"acquisizione_python_raw({n}).csv"

# ==========================
# SERIAL
# ==========================

ser = serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=TIMEOUT)

print("Connessione aperta:", PORT)

time.sleep(1)

ser.write(b"start,1-8\n")
time.sleep(0.05)
ser.reset_input_buffer()
print("Comando inviato")

HEADER = b"\x3C\x3E\x00"


# ==========================
# LETTURA PACCHETTO
# ==========================


def read_packet():
    while True:
        b = ser.read(1)

        if not b:
            return None

        if b == HEADER[0:1]:
            rest = ser.read(2)

            if rest == HEADER[1:]:
                break

    # Numero canali
    data = ser.read(1)
    if len(data) != 1:
        return None

    n_channels = struct.unpack("<B", data)[0]

    if n_channels != 8:
        return None

    # Timestamp
    data = ser.read(4)
    if len(data) != 4:
        return None

    timestamp = struct.unpack("<I", data)[0]

    # Canali ADC (Unsigned 16-bit, big-endian come da tuo codice)
    data = ser.read(n_channels * 2)
    if len(data) != n_channels * 2:
        return None

    channels = struct.unpack(f">{n_channels}H", data)

    # -------------------------------------------------------------
    # NUOVA SEZIONE: Lettura e unpack della Forza (8 sensori)
    # -------------------------------------------------------------
    data_force = ser.read(N_FORCE_SENSORS * 2)
    if len(data_force) != N_FORCE_SENSORS * 2:
        return None

    
    force_raw = struct.unpack(f">{N_FORCE_SENSORS}H", data_force)

    # Convertiamo la forza nel valore float originale tra 0.0 e 1.0
    forces = [f / FORCE_SCALE for f in force_raw]

    return timestamp, channels, forces


# ==========================
# ACQUISIZIONE
# ==========================

try:
    print(f"Acquisizione {N_SAMPLES} campioni...")

    with (
        open(RAW_FILE, "w", newline="") as raw_file,
        open(VOLT_FILE, "w", newline="") as volt_file,
    ):

        raw_writer = csv.writer(raw_file, delimiter="\t")
        volt_writer = csv.writer(volt_file, delimiter="\t")

        # Intestazioni colonne (Opzionale, rende il file più leggibile)
        # raw_writer.writerow(["Timestamp"] + [f"Ch_{i+1}" for i in range(8)] + [f"Force_{i+1}" for i in range(8)])
        # volt_writer.writerow(["Timestamp"] + [f"Volt_{i+1}" for i in range(8)] + [f"Force_{i+1}" for i in range(8)])

        samples = 0

        while samples < N_SAMPLES:

            packet = read_packet()

            if packet is None:
                continue

            timestamp, channels, forces = packet

            # Conversione volt
            voltages = [ch * VREF / ADC_MAX for ch in channels]

            # Salva RAW (timestamp + 8 canali ADC raw + 8 valori di forza ricostruita)
            raw_writer.writerow(
                [timestamp] + list(channels) + [round(f, 4) for f in forces]
            )

            # Salva VOLT (timestamp + 8 canali in Volt + 8 valori di forza ricostruita)
            volt_writer.writerow(
                [timestamp] + voltages + [round(f, 4) for f in forces]
            )

            samples += 1

            if samples % 500 == 0:
                print(f"{samples}/{N_SAMPLES}")

    print("Acquisizione terminata")
    print("RAW:", RAW_FILE)
    print("VOLT:", VOLT_FILE)

except KeyboardInterrupt:
    print("Interrotto")

finally:
    ser.write(b"stop\n")
    time.sleep(0.05)
    ser.close()