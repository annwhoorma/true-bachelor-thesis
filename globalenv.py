from enum import Enum

NUM_REGIONS = 12
NUM_NODES_PER_REGION = 5
NUM_NODES = NUM_REGIONS * NUM_NODES_PER_REGION
CONNECTED_REGIONS = None
INNERCONNECTED_REGIONS = None

class DistributionType(Enum):
    Normal = 'normal'
    Uniform = 'uniform'

DIST_TYPE = DistributionType.Uniform

REGIONS_RANGE = [int(ratio * NUM_REGIONS) for ratio in [0.25, 0.5, 0.75, 1]]
NS = range(6) # [0, 5]

SEGREGATED_REGIONS = []
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
    GlobalEfficiency = 'efficiency'
    GlobalClustering = 'clustering'
    GlobalModularity = 'modularity'
    GlobalParticipation = 'participation'
