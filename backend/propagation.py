"""Delay propagation simulator utilities

Lightweight deterministic propagation algorithm and helpers for building
an event dependency graph from schedule data. This is intentionally small
and easy to unit test for the project extension.
"""

# Optional dependency: networkx - provide a lightweight fallback if not installed
try:
    import networkx as nx
    NX_AVAILABLE = True
except Exception:
    nx = None
    NX_AVAILABLE = False

from typing import Dict, Tuple, Any
# Optional plotting dependency: matplotlib - provide fallback when not installed
try:
    import matplotlib
    # Use non-interactive backend to avoid Tkinter/main-loop issues in server contexts
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    plt = None
    MPL_AVAILABLE = False
    print('⚠️ matplotlib not available: visualizations will be simplified. Install matplotlib for richer images.')

import numpy as np
import pandas as pd
import io
import base64

if not NX_AVAILABLE:
    print('⚠️ networkx not available: using simplified graph fallback (install networkx for improved visualizations)')


# Simple fallback DiGraph implementation when networkx is not available
class _SimpleDiGraph:
    def __init__(self):
        self._nodes = set()
        self._edges = []  # list of (u, v, attrs_dict)

    def add_node(self, n):
        self._nodes.add(n)

    def add_edge(self, u, v, **attrs):
        self._nodes.add(u)
        self._nodes.add(v)
        self._edges.append((u, v, attrs))

    @property
    def nodes(self):
        return list(self._nodes)

    def edges(self, data=False):
        if data:
            return list(self._edges)
        return [(u, v) for (u, v, a) in self._edges]

    def edges_data(self):
        return self._edges

    def __iter__(self):
        return iter(self._nodes)


def build_dependency_graph(edges):
    """Build a directed graph from a list of dependency edges.

    edges: list of (src_node, dst_node, attrs_dict)
    returns: DiGraph (networkx.DiGraph or fallback)
    """
    if NX_AVAILABLE:
        G = nx.DiGraph()
        for src, dst, attrs in edges:
            G.add_node(src)
            G.add_node(dst)
            G.add_edge(src, dst, **attrs)
        return G
    else:
        G = _SimpleDiGraph()
        for src, dst, attrs in edges:
            G.add_node(src)
            G.add_node(dst)
            G.add_edge(src, dst, **attrs)
        return G


def simulate_propagation(graph, init_delays: Dict[Any, float], recovery_margin: float = 5.0, max_iters: int = 10, max_delay: float = 24*60):
    """Run a deterministic propagation simulation with safety guards.

    - graph: DiGraph (networkx or fallback) with edge attributes 'transfer_time' (float)
    - init_delays: mapping node -> initial_delay (minutes)
    - recovery_margin: margin that reduces propagation (minutes)
    - max_iters: maximum iterations of relaxation
    - max_delay: per-node cap on delay (minutes)

    Returns: final_delays (dict), traces (list of tuples describing updates), warnings (list)
    """
    # Extract nodes and edges depending on graph implementation
    try:
        nodes_iter = list(graph.nodes)
    except Exception:
        # Fallback: try iterating graph
        nodes_iter = list(graph)

    final_delays = {n: float(init_delays.get(n, 0.0)) for n in nodes_iter}
    traces = []
    warnings = []

    # Track how many times a node gets updated to detect potential runaway
    update_counts = {n: 0 for n in nodes_iter}

    # Helper to iterate edges uniformly
    def iter_edges(g):
        if NX_AVAILABLE:
            return g.edges(data=True)
        else:
            return g.edges(data=True) if hasattr(g, 'edges') and g.edges.__code__.co_argcount > 0 else g.edges_data()

    # Repeated relaxation until no change or max_iters
    for it in range(max_iters):
        changed = False
        for edge in iter_edges(graph):
            if NX_AVAILABLE:
                u, v, data = edge
            else:
                u, v, data = edge
            t_transfer = float(data.get('transfer_time', 0.0))
            upstream = final_delays.get(u, 0.0)
            candidate = max(0.0, upstream + t_transfer - float(recovery_margin))
            # Enforce per-node cap to avoid runaway accumulation in cyclic graphs
            if max_delay is not None:
                if candidate > float(max_delay):
                    candidate = float(max_delay)
                    # record cap reached for diagnostics
                    warnings.append({'type': 'cap_reached', 'node': v, 'capped_value': candidate})
            if candidate > final_delays.get(v, 0.0) + 1e-6:
                old = final_delays.get(v, 0.0)
                final_delays[v] = candidate
                traces.append((it, u, v, old, candidate))
                update_counts[v] = update_counts.get(v, 0) + 1
                changed = True
                # If a node gets updated too many times, record a warning
                if update_counts[v] > max(3, int(max_iters / 2)):
                    warnings.append({'type': 'frequent_updates', 'node': v, 'updates': update_counts[v]})
        if not changed:
            break
    # Deduplicate warnings (kept for debugging) and log, but do not return them to preserve a 2-tuple API
    unique_warnings = []
    seen = set()
    for w in warnings:
        key = (w.get('type'), w.get('node'))
        if key not in seen:
            seen.add(key)
            unique_warnings.append(w)

    if unique_warnings:
        print(f"Propagation warnings: {unique_warnings}")

    # Keep a simple 2-tuple return value (final_delays, traces) for backward compatibility with callers and tests
    return final_delays, traces


def _fig_to_base64(fig, fmt='png'):
    """Convert Matplotlib figure to base64 string.

    If Matplotlib is not available, return a tiny transparent PNG placeholder.
    """
    # 1x1 transparent PNG (base64)
    TRANSPARENT_PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII='
    if not MPL_AVAILABLE or fig is None:
        return TRANSPARENT_PNG_B64
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    try:
        plt.close(fig)
    except Exception:
        pass
    return img_b64


def visualize_propagation(graph, final_delays: Dict[Any, float], traces=None, node_positions=None, figsize=(8, 6)):
    """Create a network visualization showing delays as node colors and traces as annotated edges.

    Supports both networkx graphs and fallback _SimpleDiGraph.
    Returns (fig, ax)
    """
    # Determine node list and positions
    if NX_AVAILABLE:
        nodes = list(graph.nodes)
        if node_positions is None:
            pos = nx.spring_layout(graph, seed=42)
        else:
            pos = node_positions
    else:
        nodes = list(graph.nodes)
        # simple circular layout
        theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        pos = {node: (np.cos(t), np.sin(t)) for node, t in zip(nodes, theta)}

    delays = np.array([final_delays.get(n, 0.0) for n in nodes])

    # If matplotlib is not available, we cannot draw a figure - return placeholders
    if not MPL_AVAILABLE:
        print('⚠️ matplotlib not available: returning placeholder visualization (install matplotlib for richer graphics)')
        return None, None

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.get_cmap('Reds')
    # Normalize colors - add small epsilon to avoid division by zero
    norm = plt.Normalize(vmin=delays.min() if len(delays)>0 else 0, vmax=delays.max() if len(delays)>0 else 1)

    node_colors = [cmap(norm(final_delays.get(n, 0.0))) for n in nodes]

    # Draw edges and nodes
    if NX_AVAILABLE:
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='gray', alpha=0.6)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=800, ax=ax)
        labels = {n: f"{n}\n{final_delays.get(n,0):.1f}m" for n in nodes}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9, ax=ax)
    else:
        # Draw edges manually
        for u, v, attrs in graph.edges_data():
            x1, y1 = pos.get(u, (0,0))
            x2, y2 = pos.get(v, (0,0))
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
        # Draw nodes
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        ax.scatter(xs, ys, c=node_colors, s=800, edgecolors='k')
        for n in nodes:
            x, y = pos[n]
            ax.text(x, y, f"{n}\n{final_delays.get(n,0):.1f}m", ha='center', va='center', fontsize=9)

    ax.set_title('Delay propagation - node delays (min)')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Delay (minutes)')

    # Optionally annotate traces
    if traces:
        for it, u, v, old, new in traces:
            if u in pos and v in pos:
                ax.annotate('', xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle='->', color='black', alpha=0.3))
    plt.axis('off')
    return fig, ax


def compute_backtest_metrics(observed_final: Dict[Any, float], simulated_final: Dict[Any, float]):
    """Compute simple error metrics comparing observed vs simulated final delays."""
    keys = set(observed_final.keys()).union(set(simulated_final.keys()))
    obs = np.array([float(observed_final.get(k, 0.0)) for k in keys])
    sim = np.array([float(simulated_final.get(k, 0.0)) for k in keys])
    mae = float(np.mean(np.abs(obs - sim)))
    rmse = float(np.sqrt(np.mean((obs - sim) ** 2)))
    # fraction of nodes where sign of delay (affected vs not) differs
    obs_aff = (obs > 0).astype(int)
    sim_aff = (sim > 0).astype(int)
    aff_mismatch = int(np.sum(obs_aff != sim_aff))
    return {
        'mae': mae,
        'rmse': rmse,
        'n_nodes': int(len(keys)),
        'aff_mismatch': aff_mismatch
    }


def backtest_propagation(graph, pred_init: Dict[Any, float], observed_final: Dict[Any, float], recovery_margin: float = 5.0, max_delay: float = 24*60, max_iters: int = 10):
    """Run propagation using predicted initial delays and compare with observed final delays.

    - graph: networkx.DiGraph or fallback _SimpleDiGraph
    - max_iters: maximum relaxation iterations to run in the simulator
    Returns simulated_final, traces, metrics
    """
    # simulate_propagation now returns (simulated_final, traces)
    simulated_final, traces = simulate_propagation(graph, pred_init, recovery_margin=recovery_margin, max_iters=max_iters, max_delay=max_delay)
    metrics = compute_backtest_metrics(observed_final, simulated_final)
    return simulated_final, traces, metrics


def build_historical_graph_from_trains(trains_df, max_transfer_minutes=180):
    """Construct a simple directed graph of train-to-train dependencies using terminal/source/destination times.

    - trains_df: DataFrame with columns ['train_id','source','destination','departure_time','arrival_time']
    - Creates an edge A -> B if A.destination == B.source and departure_time(B) >= arrival_time(A) and difference <= max_transfer_minutes

    Returns: graph (DiGraph) and a dict mapping node->train_id info
    """
    G = nx.DiGraph() if NX_AVAILABLE else _SimpleDiGraph()
    # Normalize times helper
    def _time_to_minutes(tstr):
        try:
            if not tstr or tstr == '--' or pd.isna(tstr):
                return None
            parts = str(tstr).split(':')
            h = int(parts[0])
            m = int(parts[1])
            return h * 60 + m
        except Exception:
            return None

    # Build index by station for quick lookup
    dest_map = {}
    src_map = {}
    for _, row in trains_df.iterrows():
        tid = str(row.get('train_id'))
        src = str(row.get('source')) if row.get('source') is not None else ''
        dst = str(row.get('destination')) if row.get('destination') is not None else ''
        arr = _time_to_minutes(row.get('arrival_time'))
        dep = _time_to_minutes(row.get('departure_time'))
        # Add nodes
        G.add_node(tid)
        dest_map.setdefault(dst, []).append({'train_id': tid, 'arrival': arr})
        src_map.setdefault(src, []).append({'train_id': tid, 'departure': dep})

    # Construct edges from dest_map -> src_map matches
    for station, dest_trains in dest_map.items():
        src_trains = src_map.get(station, [])
        if not src_trains:
            continue
        for d in dest_trains:
            for s in src_trains:
                if d['arrival'] is None or s['departure'] is None:
                    continue
                dt = s['departure'] - d['arrival']
                if dt >= 0 and dt <= max_transfer_minutes:
                    # Edge from earlier terminating train to departing train
                    transfer_time = float(max(0.0, dt))
                    G.add_edge(d['train_id'], s['train_id'], transfer_time=transfer_time)
    return G


if __name__ == '__main__':
    # simple smoke test
    edges = [('A', 'B', {'transfer_time': 10})]
    G = build_dependency_graph(edges)
    final, traces = simulate_propagation(G, {'A': 15}, recovery_margin=5)
    print('Final delays:', final)
    print('Traces:', traces)
