import enum


# This was introduced in python 3.8
class StrEnum(str, enum.Enum):
    """An enumeration that is also a string and can be compared to strings."""

    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return str(self.value)


class TaskType(StrEnum):
    regression = enum.auto()
    binary = enum.auto()
    # multiclass = enum.auto()


class Dataset(StrEnum):
    # regression
    alkane = enum.auto()
    delaney = enum.auto()
    freesolv = enum.auto()
    lipo = enum.auto()
    opera_AOH = enum.auto()
    opera_BCF = enum.auto()
    opera_BioHL = enum.auto()
    opera_BP = enum.auto()
    opera_HL = enum.auto()
    opera_KM = enum.auto()
    opera_KOA = enum.auto()
    opera_KOC = enum.auto()
    opera_logP = enum.auto()
    opera_MP = enum.auto()
    opera_VP = enum.auto()
    opera_WS = enum.auto()
    bergstrom = enum.auto()

    # binary
    clintox = enum.auto()
    sider = enum.auto()
    bace = enum.auto()
    bbbp = enum.auto()
    opera_RBioDeg = enum.auto()


class Models(StrEnum):
    sngp = enum.auto()
    gp = enum.auto()
    bnn = enum.auto()
    ngboost = enum.auto()
    gnngp = enum.auto()


class FeatureType(StrEnum):
    mfp = enum.auto()
    mordred = enum.auto()
    graphnet = enum.auto()
    graphembed = enum.auto()


GRAPH_FEATURES = [FeatureType.graphnet]
VECTOR_FEATURES = [f for f in FeatureType if f not in GRAPH_FEATURES]


class GNNBlock(enum.Enum):
    gcn = 'gcn'
    mpnn = 'mpnn'
    graphnet = 'graphnet'
