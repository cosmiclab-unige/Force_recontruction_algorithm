from mcap.reader import make_reader
import struct
import matplotlib.pyplot as plt
import numpy as np

header_size = 28  # bytes da saltare per arrivare ai valori float32

sensors_data = []

with open("rosbag2.mcap", "rb") as f:
    for schema, channel, message in make_reader(f).iter_messages():
        data = message.data  # bytearray con il messaggio serializzato

        # --- 1️⃣ salta header ---
        payload = data[header_size:]

        # --- 2️⃣ leggi lunghezza array float32[] ---
        if len(payload) < 4:
            continue  # messaggio troppo corto, skip
        num_values = struct.unpack('<I', payload[:4])[0]

        # --- 3️⃣ estrai float32 ---
        expected_len = 4 + num_values * 4
        if len(payload) < expected_len:
            continue  # messaggio incompleto
        float_data = payload[4:expected_len]
        values = struct.unpack('<' + 'f' * num_values, float_data)

        # Salva i valori nel buffer
        sensors_data.append(values)


sensors_data = np.array(sensors_data)

plt.figure(figsize=(10, 5))
plt.title("Tactile sensor readings")
plt.xlabel("Nb of samples")
plt.ylabel("Voltage [V]")
plt.grid(True)

# tracciamo ogni sensore come linea
for i in range(sensors_data.shape[1]):
    plt.plot(sensors_data[:, i], alpha=0.7)

plt.tight_layout()
plt.show()