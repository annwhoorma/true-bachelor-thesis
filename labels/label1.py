import globalenv
from labels.label_interface import LabelInterface

class Label1(LabelInterface):
    '''integration'''
    def __init__(self, mask, label_name, low_dist, high_dist):
        super(Label1, self).__init__(mask, label_name, low_dist, high_dist)
        self.distinguished = self._choose_random_regions(range(self.num_regions), globalenv.CONNECTED_REGIONS)
        self._generate_regions()
        self._generate_patterns()
        # self.A = self._make_symmetric(self.A) * self.mask
        self.A *= self.mask

    def _generate_regions(self):
        n = self.num_regions
        super()._generate_regions()
        self.connections = {
            'low': [(i, j) for i in range(0, n) for j in range(i, n) if not (i in self.distinguished and j in self.distinguished) or i == j],
            'high': [(i, j) for i in range(0, n) for j in range(i+1, n) if (i in self.distinguished and j in self.distinguished)]
            }
