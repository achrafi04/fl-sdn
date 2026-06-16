"""
SERVEUR FL (Palier 2) - corrige : agrege SwitchPart ET ServerPart.
"""
import socket, pickle, struct, sys, copy
sys.path.append("../model")
import numpy as np
import torch
from split_model import SwitchPart, ServerPart

HOST, PORT = "127.0.0.1", 9000
N_CLIENTS = 2
ROUNDS = 5

def send_obj(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(struct.pack(">I", len(data)) + data)

def recvall(conn, n):
    data = b""
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet: return None
        data += packet
    return data

def recv_obj(conn):
    raw_len = recvall(conn, 4)
    if not raw_len: return None
    length = struct.unpack(">I", raw_len)[0]
    return pickle.loads(recvall(conn, length))

def fedavg(states):
    avg = copy.deepcopy(states[0])
    for k in avg.keys():
        for s in states[1:]:
            avg[k] += s[k]
        avg[k] /= len(states)
    return avg

d = np.load("../data/cicids2017_processed.npz")
X_test = torch.tensor(d["X_test"], dtype=torch.float32)
y_test = torch.tensor(d["y_test"], dtype=torch.long)

global_switch = SwitchPart()
global_server = ServerPart()

def evaluate():
    global_switch.eval(); global_server.eval()
    with torch.no_grad():
        return (global_server(global_switch(X_test)).argmax(1) == y_test).float().mean().item()

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT))
srv.listen(N_CLIENTS)
print(f"Serveur en ecoute sur {HOST}:{PORT}, j'attends {N_CLIENTS} clients...")

conns = []
for i in range(N_CLIENTS):
    conn, addr = srv.accept()
    conns.append(conn)
    print(f"  Client {i+1} connecte depuis {addr}")

print(f"\nDemarrage du federated ({ROUNDS} rounds)\n")
for r in range(ROUNDS):
    # 1. envoyer le modele global (les DEUX parties) a chaque client
    for conn in conns:
        send_obj(conn, {"switch": global_switch.state_dict(),
                        "server": global_server.state_dict()})
    # 2. recevoir les DEUX parties entrainees de chaque client
    replies = [recv_obj(conn) for conn in conns]
    switch_states = [rep["switch"] for rep in replies]
    server_states = [rep["server"] for rep in replies]
    # 3. FedAvg sur les DEUX parties
    global_switch.load_state_dict(fedavg(switch_states))
    global_server.load_state_dict(fedavg(server_states))
    print(f"  Round {r+1}/{ROUNDS} | acc globale {evaluate()*100:.2f}%")

for conn in conns:
    send_obj(conn, "DONE")
    conn.close()
print("\nFederated termine. Serveur arrete.")
srv.close()
