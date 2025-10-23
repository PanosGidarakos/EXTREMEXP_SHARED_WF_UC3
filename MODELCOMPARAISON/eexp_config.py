# Main folders (mandatory) - paths are just examples
TASK_LIBRARY_PATH = 'tasks'
EXPERIMENT_LIBRARY_PATH = 'experiments'
# The ones below has to be a path relative to the client script
DATASET_LIBRARY_RELATIVE_PATH = 'datasets'
PYTHON_DEPENDENCIES_RELATIVE_PATH = 'dependencies'

EXECUTIONWARE = "PROACTIVE"  # other option: "LOCAL"
# Proactive credentials (only needed if EXECUTIONWARE = "PROACTIVE" above)
PROACTIVE_URL = "https://proactive.extremexp-icom.intracom-telecom.com"
PROACTIVE_USERNAME = "ads_user"
PROACTIVE_PASSWORD = "}h18?7DhM"

MAX_WORKFLOWS_IN_PARALLEL_PER_NODE = 4

DATA_ABSTRACTION_BASE_URL = "https://api.dal.extremexp-icom.intracom-telecom.com/api"
DATA_ABSTRACTION_ACCESS_TOKEN = '72fcd93c8e91c9a4704f80754369c1dba25e2ecb'

PROACTIVE_PYTHON_VERSIONS = {
    "3.8": "/usr/bin/python3.8", "3.9": "/usr/bin/python3.9"}

DATASET_MANAGEMENT = "DDM"
DDM_URL = "https://ddm.extremexp-icom.intracom-telecom.com"
PORTAL_USERNAME = "drouglazet"
PORTAL_PASSWORD = "Alphazero1?"

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
