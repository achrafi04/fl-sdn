#!/usr/bin/env python3
"""
Topologie Mininet pour le FL-SDN (Palier 3).
- 1 serveur (le controleur/federateur) : h_server
- 2 clients dans le sous-reseau X : hX1, hX2
- 2 clients dans le sous-reseau Y : hY1, hY2
Tous connectes pour pouvoir atteindre le serveur (FL possible).
"""
from mininet.net import Mininet
from mininet.node import OVSController
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel

class FLTopo(Topo):
    def build(self):
        # le serveur FL (= controleur/federateur)
        server = self.addHost("hserver", ip="10.0.0.1/24")

        # sous-reseau X : 2 clients
        hX1 = self.addHost("hX1", ip="10.0.0.11/24")
        hX2 = self.addHost("hX2", ip="10.0.0.12/24")
        # sous-reseau Y : 2 clients
        hY1 = self.addHost("hY1", ip="10.0.0.21/24")
        hY2 = self.addHost("hY2", ip="10.0.0.22/24")

        # switches : un par sous-reseau + un central
        sX = self.addSwitch("s1")    # switch sous-reseau X
        sY = self.addSwitch("s2")    # switch sous-reseau Y
        sC = self.addSwitch("s3")    # switch central (vers le serveur)

        # liens clients X -> switch X
        self.addLink(hX1, sX, cls=TCLink, bw=10)
        self.addLink(hX2, sX, cls=TCLink, bw=10)
        # liens clients Y -> switch Y
        self.addLink(hY1, sY, cls=TCLink, bw=10)
        self.addLink(hY2, sY, cls=TCLink, bw=10)
        # serveur -> switch central
        self.addLink(server, sC, cls=TCLink, bw=20)
        # interconnexion des switches (pour que tout le monde atteigne le serveur)
        self.addLink(sX, sC, cls=TCLink, bw=20)
        self.addLink(sY, sC, cls=TCLink, bw=20)

def run():
    net = Mininet(topo=FLTopo(), controller=OVSController, link=TCLink, autoSetMacs=True)
    net.start()
    print("\n=== Topologie FL-SDN lancee ===")
    print("Serveur : hserver (10.0.0.1)")
    print("Sous-reseau X : hX1 (10.0.0.11), hX2 (10.0.0.12)")
    print("Sous-reseau Y : hY1 (10.0.0.21), hY2 (10.0.0.22)")
    print("\nTest de connectivite (tous doivent s'atteindre) :")
    net.pingAll()
    print("\nPour lancer le FL, utilise les commandes dans le CLI mininet (voir guide).")
    CLI(net)
    net.stop()

if __name__ == "__main__":
    setLogLevel("info")
    run()
