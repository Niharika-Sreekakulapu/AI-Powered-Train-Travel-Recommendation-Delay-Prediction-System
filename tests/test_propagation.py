from backend.propagation import build_dependency_graph, simulate_propagation


def test_simple_propagation():
    edges = [('A', 'B', {'transfer_time': 10})]
    G = build_dependency_graph(edges)
    final, traces = simulate_propagation(G, {'A': 15}, recovery_margin=5)
    assert 'B' in final
    # expected: B = max(0, 15 + 10 - 5) = 20
    assert abs(final['B'] - 20.0) < 1e-6


if __name__ == '__main__':
    print('Running simple propagation test...')
    test_simple_propagation()
    print('OK')