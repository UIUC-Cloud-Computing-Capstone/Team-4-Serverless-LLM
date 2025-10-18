import time
from udpnode import Node

def main():
    """Main function to run the UDP-based gossip protocol simulation."""
    
    # Simulation parameters
    num_nodes = 5
    gossip_duration = 20  # seconds
    gossip_interval = 1   # seconds
    time_to_failure = 5   # seconds
    base_port = 8000
    host = "127.0.0.1"

    nodes = []
    print("--- Creating Nodes ---")
    for i in range(num_nodes):
        node = Node(f"Node-{i}", host, base_port + i, time_to_failure)
        nodes.append(node)

    print("\n--- Initializing Peer Connections ---")
    # Initial peer connections (fully connected for robustness)
    for node1 in nodes:
        for node2 in nodes:
            if node1.get_id() != node2.get_id():
                node1.add_peer(node2.get_id(), node2.host, node2.port)

    # Set some initial data
    nodes[0].set_data("message", "Hello UDP Gossip!")
    nodes[2].set_data("status", "active")
    
    print("\n--- Starting Gossip Simulation ---")
    for node in nodes:
        node.start(gossip_interval)
    
    # Let the simulation run and then simulate a failure
    time.sleep(gossip_duration / 2)
    failed_node_index = 1
    print(f"\n--- Stopping {nodes[failed_node_index].get_id()} to simulate failure ---\n")
    nodes[failed_node_index].stop()

    # Allow the remaining nodes to detect the failure
    time.sleep(gossip_duration / 2)

    print("--- Stopping all remaining nodes ---")
    for i, node in enumerate(nodes):
        if i != failed_node_index:
            node.stop()

    print("\n--- Gossip Simulation Finished ---")
    # Print final state of each active node
    for i, node in enumerate(nodes):
         if i != failed_node_index:
            print(f"\n--- State of {node.get_id()} ---")
            print(f"Data: {node.get_data()}")
            print(f"Peers: {list(node.get_peers().keys())}")

if __name__ == "__main__":
    main()
