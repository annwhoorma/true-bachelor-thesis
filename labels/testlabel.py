import globalenv
from labels.label_interface import LabelInterface

class TestLabel(LabelInterface):
    def __init__(self, mask, label_name, segr):
        self.segr = segr
        super(TestLabel, self).__init__(mask, label_name)
        self.distinguished = self._choose_random_regions(range(self.num_regions),
            globalenv.INNERCONNECTED_REGIONS if segr else globalenv.CONNECTED_REGIONS)
        self._generate_regions()
        self._generate_patterns()
        self.A *= self.mask

    def _generate_regions(self):
        n = self.num_regions
        super()._generate_regions()
        if self.segr:
            # segregation - global clustering coefficient
            self.connections = {
                'low': [(i, j) for i in range(0, n) for j in range(i, n)],
                'high': [(i, i) for i in range(0, n) if i in self.distinguished]
                }
        else:
            # integration - global efficiency coefficient
            self.connections = {
                'low': [(i, j) for i in range(0, n) for j in range(i, n) if not (i in self.distinguished and j in self.distinguished) or i == j],
                'high': [(i, j) for i in range(0, n) for j in range(i+1, n) if (i in self.distinguished and j in self.distinguished)]
                }