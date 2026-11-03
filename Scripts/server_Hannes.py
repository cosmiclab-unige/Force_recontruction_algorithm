import socket
import struct
import numpy as np
from force_reconstructor_Hannes import ForceReconstructor  # la tua classe
import threading
import sys

HOST = "127.0.0.1"
PORT = 4000

n_sensors = 16  # <<< metti il tuo numero reale

reconstructor = ForceReconstructor(
    n_sensors=n_sensors,
    NW=2000,
    Thr_samples=2000,
    warmup_samples=500,
    press_sigma=10,
    alpha=0.1,
    nSamples_adaptive_offset=50,
    press_confirm=2,
    samples_after_release=500,
    min_press_sensors=1,
    release_ratio=0.5
)

def keyboard_listener():
    while True:
        cmd = input()
        if cmd.lower() == "t":
            print("Thresholds recomputed")
            reconstructor.recalibrate_thresholds()
        if cmd.lower() == "r":
            print("Resel all")
            reconstructor.reset_all()
            

def recv_exact(conn, size):
    data = b""
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print("Python server in ascolto...")
    conn, addr = s.accept()
    print("Connesso:", addr)

    with conn:
        threading.Thread(target=keyboard_listener, daemon=True).start()
        while True:
            # --- ricevi header (numero campioni) ---
            header = recv_exact(conn, 4)
            if header is None:
                break

            (N,) = struct.unpack(">I", header)
         
            # --- ricevi dati ---
            data_bytes = recv_exact(conn, N * n_sensors * 8)
            if data_bytes is None:
                break

            block = np.frombuffer(data_bytes, dtype=">f8")
            block = block.reshape((N, n_sensors))

            # --- elaborazione ---
            out, colors = reconstructor.process_block(block)

            # --- invio risposta ---
            conn.sendall(out.astype('>f8').tobytes())
            conn.sendall(colors.tobytes())