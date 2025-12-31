from backend.propagation import build_dependency_graph, simulate_propagation, backtest_propagation


def test_backtest_perfect_match():
    edges = [('A', 'B', {'transfer_time': 10}), ('B', 'C', {'transfer_time': 5})]
    G = build_dependency_graph(edges)
    # Predicted initial: A=15
    pred_init = {'A': 15}
    # Observed final expected if propagation correct: A=15, B=20, C=20
    observed_final = {'A': 15.0, 'B': 20.0, 'C': 20.0}

    simulated, traces, metrics = backtest_propagation(G, pred_init, observed_final, recovery_margin=5.0)
    assert abs(metrics['mae']) < 1e-6
    assert metrics['n_nodes'] == 3


if __name__ == '__main__':
    test_backtest_perfect_match()
    print('OK')