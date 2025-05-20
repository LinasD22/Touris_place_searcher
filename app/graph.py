
from pyvis.network import Network
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import pickle

#plt.ion()
import numpy as np
import json


def build_and_plot_graph(objects_to_props, similarity_coof):
    """
    Creates an undirected graph where each node is an object (key in objects_to_props).
    An edge is added between two objects if at least 60% of their combined properties match.
    """

    # Initialize an empty graph
    G = nx.Graph()

    # 1) Add all objects as nodes
    for obj in objects_to_props.keys():
        G.add_node(obj)

    # 2) For each pair of objects, measure the intersection ratio
    object_list = list(objects_to_props.keys())
    n = len(object_list)

    for i in range(n):
        for j in range(i + 1, n):
            objA = object_list[i]
            objB = object_list[j]

            propsA = objects_to_props[objA]
            propsB = objects_to_props[objB]

            # Make sure both are sets for easy intersection/union
            if not isinstance(propsA, set):
                propsA = set(propsA)
            if not isinstance(propsB, set):
                propsB = set(propsB)

            intersection = propsA.intersection(propsB)
            union = propsA.union(propsB)

            if len(union) > 0:
                overlap_ratio = len(intersection) / len(union)
            else:
                overlap_ratio = 0.0

            # Check if at least 75% of their combined properties overlap
            if overlap_ratio >= similarity_coof:
                # Add an edge
                G.add_edge(objA, objB)

    return G
    # 3) Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # or any layout you prefer
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title("Objects Linked by ≥60% Matching Properties")
    plt.axis("off")
    plt.show()
    return G

# --- Usage Example ---
# build_and_plot_graph(objects_to_props)

def split_dict_into_n_chunks(original_dict, num_chunks):
    # Convert dictionary to list of key-value pairs
    items_list = list(original_dict.items())

    # Calculate how many items per chunk (rounded up)
    chunk_size = len(items_list) // num_chunks
    if len(items_list) % num_chunks != 0:
        chunk_size += 1

    # Split the list into chunks and convert each chunk back to a dictionary
    chunks = []
    for i in range(0, len(items_list), chunk_size):
        chunk = dict(items_list[i:i + chunk_size])
        chunks.append(chunk)

    return chunks



def get_connected_components_subgraphs(G):
    """
    Splits the input graph G into subgraphs (connected components).
    Returns a list of subgraphs.
    """
    # Check if G is a proper Graph object
    if G is None:
        raise ValueError("The graph is None. Make sure you've created a valid graph before calling this function.")
    # Get connected components as sets of nodes.
    components = nx.connected_components(G)
    # Build subgraphs for each connected set.
    subgraphs = [G.subgraph(c).copy() for c in components]
    return subgraphs

def plot_subgraphs(subgraphs):
    #%matplotlib widget
    """
    Plots each connected component (subgraph) separately.
    """
    it=0
    for i, subg in enumerate(subgraphs):
        if subg.number_of_nodes() != 1:

            #if it == 0:
                it+=1
                plt.figure(figsize=(5, 4))
                pos = nx.spring_layout(subg, seed=42)  # layout for node positions
                nx.draw_networkx_nodes(subg, pos, node_size=800, node_color='skyblue')
                nx.draw_networkx_edges(subg, pos)
                nx.draw_networkx_labels(subg, pos, font_size=10)
                plt.title(f"Component {i+1} ({len(subg.nodes())} nodes)")
                plt.axis("off")

                plt.show()
            #it += 1

def plot_graph_pyvis(graph, number):
    #key = list(graph.keys())[number]
    #subgraph = graph[key]
    subgraph = subgraphs[number]

    net = Network(notebook=True, height='800px', width='100%')
    net.from_nx(subgraph)  # Import the NetworkX graph into PyVis
    net.show(f"subgraph{number}.html")  # This creates and opens an HTML page with an interactive visualization


def export_subgraph_to_json(subg, filename):
    subg_nodes = list(subg.nodes())
    name_to_node = {}
    node_map = {}
    for idx, old_id in enumerate(subg_nodes):
        node_map[old_id] = idx + 1
        name_to_node[idx + 1] = old_id

    edges = []
    for (u, v) in subg.edges():
        edges.append([node_map[u], node_map[v]])

    # For simplicity, set all rest lengths to 1
    l0 = [1.0] * len(edges)

    data = {
        "nmz": len(subg_nodes),
        "edges": edges,
        "l0": l0,
        "Names_info": name_to_node
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def export_graphML(subgraph, fileName):
    nx.write_graphml(subgraph, fileName)

if __name__ == '__main__':



    with open("./data/cities_prop.json", "r") as file:
        cities_prop = json.load(file)

    #parts = split_dict_into_n_chunks(cities_prop, 40)

    graph = build_and_plot_graph(cities_prop, 0.7)
    #print(graph)
    subgraphs = get_connected_components_subgraphs(graph)
    print(f"Found {len(subgraphs)} connected components.")
    #plot_subgraphs(subgraphs)  # Plots each "island"
    with open(f"./data/graph.pickle", 'wb') as file:
        pickle.dump(graph, file)

    # cool graPH
    plot_graph_pyvis(subgraphs, 17)


    #save subgrapgs .json
    for i, subg in enumerate(subgraphs):
        if subg.number_of_nodes() > 1:
            file_path = f"./data/subgraphs/subgraph_{i}.json"
            export_subgraph_to_json(subg, file_path)
    #save subgrapgs .graphml
    for i, subg in enumerate(subgraphs):
        if subg.number_of_nodes() > 1:
            file_path = f"./data/subgraphs/cyto/cytofgraph{i}.graphml"
            export_graphML(subg, file_path)

    #input("Press Enter to exit...")