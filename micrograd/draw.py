from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
    nodes, edges = trace(root)
    
    # Mapăm nodurile la ID-uri pentru simplitate
    node_ids = {n: f"({n.label})" for n in nodes}

    # Construim structura arborelui
    from collections import defaultdict
    tree = defaultdict(list)

    for n1, n2 in edges:
        tree[n2].append((n1, n2._op))

    def build_ascii_tree(node, prefix=""):
        if node not in tree:
            return f"{prefix}{node_ids[node]} [data={node.data:.4f}, grad={node.grad:.4f}]\n"
        
        output = f"{prefix}{node_ids[node]} [data={node.data:.4f}, grad={node.grad:.4f}]\n"
        children = tree[node]
        
        for i, (child, op) in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            output += f"{prefix}{connector}({op})\n"
            output += build_ascii_tree(child, prefix + ("    " if i == len(children) - 1 else "│   "))
        
        return output

    print(build_ascii_tree(root))
