"""
CLIENT FL (Palier 2) : se connecte au serveur, recoit le modele global,
entraine SA partie sur SES donnees, renvoie ses poids. Tout par socket TCP.

Usage : python3 client.py <client_id> <subnet>
  ex : python3 client.py 1 X    (client 1, sous-reseau X)
       python3 client.py 2 Y    (client 2, sous-reseau Y)
"""
import socket, pickle, struct, sys
sys.path.append("../model")
import numpy as np
import torch
import torch.nn as nn
from split_model import SwitchPart, ServerPart

HOST, PORT = "127.0.0.1", 9000

# --- arguments ---
client_id = sys.argv[1] if len(sys.argv) > 1 else "1"
subnet    = sys.argv[2] if len(sys.argv) > 2 else "X"

# --- helpers socket (identiques au serveur) ---
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

# --- charger SES donnees (selon le sous-reseau) ---
d = np.load("../data/cicids2017_processed.npz")
X = torch.tensor(d["X_train"], dtype=torch.float32)
y = torch.tensor(d["y_train"], dtype=torch.long)

torch.manual_seed(int(client_id))
# sous-reseau X : surtout normal | sous-reseau Y : surtout attaques
idx_normal = (y == 0).nonzero().squeeze()
idx_attack = (y == 1).nonzero().squeeze()
if subnet == "X":
    idx = torch.cat([idx_normal[:300000], idx_attack[:30000]])   # surtout normal
else:
    idx = torch.cat([idx_normal[:30000], idx_attack[:150000]])   # surtout attaques
idx = idx[torch.randperm(len(idx))]
X, y = X[idx], y[idx]
pct_atk = 100 * (y == 1).sum().item() / len(y)
print(f"[Client {client_id}/{subnet}] {len(y)} flux | {pct_atk:.0f}% attaques")

# --- se connecter au serveur ---
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
print(f"[Client {client_id}] connecte au serveur")

# --- modele local ---
switch = SwitchPart()
server = ServerPart()   # copie locale pour entrainer le split
loss_fn = nn.CrossEntropyLoss()

# --- boucle federee ---
while True:
    msg = recv_obj(sock)               # recoit le modele global (ou "DONE")
    if msg == "DONE":
        print(f"[Client {client_id}] termine.")
        break
    switch.load_state_dict(msg["switch"])    # part du modele global recu
    server.load_state_dict(msg["server"])      # part du modele global recu
    opt = torch.optim.Adam(list(switch.parameters())+list(server.parameters()), lr=0.001)
    switch.train(); server.train()
    pm = torch.randperm(X.shape[0])
    for i in range(0, X.shape[0], 4096):
        b = pm[i:i+4096]
        opt.zero_grad()
        loss_fn(server(switch(X[b])), y[b]).backward()
        opt.step()
    send_obj(sock, {"switch": switch.state_dict(), "server": server.state_dict()})
    print(f"[Client {client_id}] round termine, poids envoyes")

sock.close()
