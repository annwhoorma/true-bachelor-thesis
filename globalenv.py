from enum import Enum

NUM_NODES = 81
NUM_REGIONS = 9
NUM_NODES_PER_REGION = 9
CONNECTED_REGIONS = 2 # [2, 9] for integration
INNERCONNECTED_REGIONS = 2 # [2, 9] for segregation
SEGREGATED_REGIONS = []
DS = range(6) # [0, 5]
MIN_VALUE = 0
MAX_VALUE = 1

class GenDataset(Enum):
    segregation = 'segregation'
    integration = 'integration'