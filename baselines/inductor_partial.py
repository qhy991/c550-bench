"""Complete original call with explicitly partial TorchInductor capture."""
ROUTE = {'kind': 'inductor', 'capture': 'declared_partial', 'dynamic': False,
         'known_eager_regions': ['data-dependent Python control flow and host scalar/list conversion',
                                 'unsupported original operations reported by Dynamo graph breaks']}
