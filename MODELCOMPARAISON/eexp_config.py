# Main folders (mandatory) - paths are just examples
TASK_LIBRARY_PATH = 'tasks'
EXPERIMENT_LIBRARY_PATH = 'experiments'
# The ones below has to be a path relative to the client script
DATASET_LIBRARY_RELATIVE_PATH = 'datasets'
PYTHON_DEPENDENCIES_RELATIVE_PATH = 'dependencies'

EXECUTIONWARE = "PROACTIVE"  # other option: "LOCAL"
# Proactive credentials (only needed if EXECUTIONWARE = "PROACTIVE" above)
PROACTIVE_URL = "http://146.124.106.171:8880"
PROACTIVE_USERNAME="arc_user"
PROACTIVE_PASSWORD="cY2UY_5i8"

MAX_WORKFLOWS_IN_PARALLEL_PER_NODE = 4

DATA_ABSTRACTION_BASE_URL = "http://146.124.106.171:8445/api"
DATA_ABSTRACTION_ACCESS_TOKEN = 'b7eb45c87af9b4e32f2bdf38f9e605e4b2d20130'
PROACTIVE_PYTHON_VERSIONS = {
    "3.8": "/usr/bin/python3.8", "3.9": "/usr/bin/python3.9"}

DATASET_MANAGEMENT = "DDM"
DDM_URL = "https://ddm.extremexp-icom.intracom-telecom.com"
PORTAL_USERNAME = "ntinos"
PORTAL_PASSWORD = "ntinos123"

# logging configuration, optional; if not set, all loggers have INFO level
LOGGING_CONFIG = {
    'version': 1,
    'loggers': {
        'eexp_engine.functions': {
            'level': 'INFO'
        },
        'eexp_engine.functions.parsing': {
            'level': 'INFO',
        },
        'eexp_engine.functions.execution': {
            'level': 'INFO',
        },
        'eexp_engine.data_abstraction_layer': {
            'level': 'INFO'
        },
        'eexp_engine.models': {
            'level': 'INFO'
        },
        'eexp_engine.proactive_executionware': {
            'level': 'INFO'
        }
    }
}
