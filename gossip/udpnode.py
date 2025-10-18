import socket
import threading
import pickle
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(threadName)s] %(message)s')

class Node:
    """A node in the gossip protocol network using UDP."""

    def __init__(self, node_id, host, port, time_to_failure=10, verbose=True):
        """
        Args:
            node_id (str): A unique identifier for the node.
            host (str): The host address the node will bind to.
            port (int): The port the node will listen on.
            time_to_failure (int): Seconds of inactivity before a peer is considered failed.
            verbose (bool): If True, enables detailed logging.
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.time_to_failure = time_to_failure

        self.lamport_clock = 0
        self.peers = {}  # {node_id: {"host": host, "port": port, "last_seen": timestamp, "lamport_clock": clock}}
        self.data = {}   # {key: (value, lamport_clock)}
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        
        self.is_running = True
        self._listener_thread = None
        self._gossip_threaad = None
        self.verbose = verbose
        if self.verbose:
            logging.info(f"{self.node_id} initialized at {self.host}:{self.port} with verbose logging.")

    def _increment_clock(self):
        """Increments Lamport clock for an event."""
        self.lamport_clock += 1

    def get_id(self):
        return self.node_id

    def get_data(self):
        return self.data

    def get_peers(self):
        return self.peers

    def set_data(self, key, value):
        """Sets KV pair, creating a local event."""
        self._increment_clock()
        self.data[key] = (value, self.lamport_clock)
        if self.verbose:
            logging.info(f"{self.node_id} set data: {key}={value} at time {self.lamport_clock}")

    def add_peer(self, peer_id, host, port):
        """Adds a new peer to the node's peer list."""
        if peer_id != self.node_id and peer_id not in self.peers:
            self._increment_clock()
            self.peers[peer_id] = {"host": host, "port": port, "last_seen": time.time(), "lamport_clock": 0}
            if self.verbose:
                logging.info(f"{self.node_id} added peer {peer_id} at {host}:{port}")

    def _listener(self):
        """Listens for incoming UDP gossip messages."""
        if self.verbose:
            logging.info(f"{self.node_id} listener started.")
        while self.is_running:
            try:
                # Use a larger buffer size for pickled objects
                raw_data, _ = self.sock.recvfrom(4096)
                message = pickle.loads(raw_data)
                self._handle_gossip(message)
            except (socket.error, pickle.UnpicklingError, EOFError):
                if self.is_running:
                    logging.error(f"{self.node_id} encountered a listener error.", exc_info=True)
                break
        if self.verbose:
            logging.info(f"{self.node_id} listener stopped.")
        
    def _handle_gossip(self, message):
        """Processes a received gossip message."""
        sender_id = message['id']
        if not self.is_running or sender_id == self.node_id:
            return

        self._increment_clock()
        self.lamport_clock = max(self.lamport_clock, message['lamport_clock']) + 1

        if sender_id in self.peers:
            self.peers[sender_id]["last_seen"] = time.time()
            self.peers[sender_id]["lamport_clock"] = message['lamport_clock']

        # Merge data based on Lamport timestamps
        for key, (value, ts) in message['data'].items():
            if key not in self.data or ts > self.data[key][1]:
                self.data[key] = (value, ts)
                if self.verbose:
                    logging.info(f"{self.node_id} updated data from {sender_id}: {key}={value}")

        # Discover new peers
        for peer_id, peer_info in message['peers'].items():
            self.add_peer(peer_id, peer_info['host'], peer_info['port'])
        
        # Add the sender if they are unknown
        sender_info = message['sender_address']
        self.add_peer(sender_id, sender_info['host'], sender_info['port'])


    def _gossiper(self, interval):
        """Periodically sends gossip to a random peer."""
        if self.verbose:
            logging.info(f"{self.node_id} gossiper started.")
        while self.is_running:
            time.sleep(interval)
            self._check_failed_nodes()

            if not self.peers:
                continue

            random_peer_id = random.choice(list(self.peers.keys()))
            peer_info = self.peers[random_peer_id]

            self._increment_clock()
            
            payload = {
                'id': self.node_id,
                'sender_address': {'host': self.host, 'port': self.port},
                'data': self.data,
                'peers': self.peers,
                'lamport_clock': self.lamport_clock
            }
            
            try:
                serialized_payload = pickle.dumps(payload)
                self.sock.sendto(serialized_payload, (peer_info['host'], peer_info['port']))
                if self.verbose:
                    logging.info(f"{self.node_id} gossiped to {random_peer_id}")
            except socket.error:
                 logging.warning(f"{self.node_id} failed to send gossip to {random_peer_id}", exc_info=True)
        if self.verbose:
            logging.info(f"{self.node_id} stopped gossiping.")

    def _check_failed_nodes(self):
        """Removes peers that have not been seen recently."""
        now = time.time()
        failed_nodes = [
            peer_id for peer_id, peer_info in self.peers.items()
            if now - peer_info["last_seen"] > self.time_to_failure
        ]
        for peer_id in failed_nodes:
            del self.peers[peer_id]
            logging.warning(f"{self.node_id} detected failure of node {peer_id}")

    def start(self, gossip_interval=1):
        """Starts the listener and gossiper threads."""
        self._listener_thread = threading.Thread(target=self._listener, name=f"{self.node_id}-Listener")
        self._gossip_threaad = threading.Thread(target=self._gossiper, args=(gossip_interval,), name=f"{self.node_id}-Gossiper")
        
        self._listener_thread.start()
        self._gossip_threaad.start()

    def stop(self):
        """Stops the node and its threads gracefully."""
        if not self.is_running:
            return
            
        self.is_running = False
        self.sock.close() # This will cause recvfrom in the listener to raise an exception.
        
        if self._gossip_threaad:
            self._gossip_threaad.join()
        if self._listener_thread:
            self._listener_thread.join()
        if self.verbose:
            logging.info(f"{self.node_id} has been stopped.")
