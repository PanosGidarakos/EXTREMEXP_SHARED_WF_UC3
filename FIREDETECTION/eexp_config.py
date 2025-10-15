# Main folders (mandatory) - paths are just examples
TASK_LIBRARY_PATH = 'tasks'
EXPERIMENT_LIBRARY_PATH = 'experiments'
# The ones below has to be a path relative to the client script
DATASET_LIBRARY_RELATIVE_PATH = 'datasets'
PYTHON_DEPENDENCIES_RELATIVE_PATH = 'dependencies'

EXECUTIONWARE = "PROACTIVE"  # other option: "LOCAL"
# Proactive credentials (only needed if EXECUTIONWARE = "PROACTIVE" above)
PROACTIVE_URL = ""
PROACTIVE_USERNAME = ""
PROACTIVE_PASSWORD = ""

MAX_WORKFLOWS_IN_PARALLEL_PER_NODE = 4

DATA_ABSTRACTION_BASE_URL = ""
DATA_ABSTRACTION_ACCESS_TOKEN = ''

PROACTIVE_PYTHON_VERSIONS = {"3.8": "/usr/bin/python3.8", "3.9": "/usr/bin/python3.9"}

DATASET_MANAGEMENT = "DDM"
DDM_URL = ""
PORTAL_USERNAME = ""
PORTAL_PASSWORD = ""

# logging configuration, optional; if not set, all loggers have INFO level
LOGGING_CONFIG = {
    'version': 1,
    'loggers': {
        'eexp_engine.functions': {
            'level': 'DEBUG'
        },
        'eexp_engine.functions.parsing': {
            'level': 'DEBUG',
        },
        'eexp_engine.functions.execution': {
            'level': 'DEBUG',
        },
        'eexp_engine.data_abstraction_layer': {
            'level': 'DEBUG'
        },
        'eexp_engine.models': {
            'level': 'DEBUG'
        },
        'eexp_engine.proactive_executionware': {
            'level': 'DEBUG'
        }
    }
}
