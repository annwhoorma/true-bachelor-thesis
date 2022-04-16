from enum import Enum

NUM_NODES = 81
NUM_REGIONS = 9
NUM_NODES_PER_REGION = 9
CONNECTED_REGIONS = 2 # [2, 9] for integration
INNERCONNECTED_REGIONS = 2 # [2, 9] for segregation
SEGREGATED_REGIONS = []
DS = range(6) # [0, 5]
CS = range(2, 10)
MIN_VALUE = 0
MAX_VALUE = 1

class GenDataset(Enum):
    segregation = 'segregation'
    integration = 'integration'

class CalculateMetrics(Enum):
    segregation = 'segregation'
    integration = 'integration'
    neutral = 'neutral'

class WeightedMetric(Enum):
    GlobalEfficiency = 'g_efficiency'
    GlobalClustering = 'g_clustering'
    GlobalModularity = 'g_modularity'
    GlobalParticipation = 'g_participation'