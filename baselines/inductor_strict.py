"""Exact original graph compiled with the installed MetaX TorchInductor backend."""
ROUTE = {'kind': 'inductor', 'capture': 'fullgraph', 'dynamic': False,
         'known_eager_regions': []}
