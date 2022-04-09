import globalenv
from labels.label_interface import LabelInterface

class Label3(LabelInterface):
    '''segregation'''
    def __init__(self, num_nodes, num_edges, mask, label_name):
        super(Label3, self).__init__(num_nodes, num_edges, mask, label_name)
        self.distinguished = self._choose_random_regions(range(self.num_regions), globalenv.INNERCONNECTED_REGIONS)
        self._generate_regions()
        self._generate_patterns()
        # self.A = self._make_symmetric(self.A) * self.mask
        self.A *= self.mask

    def _generate_regions(self):
        n = self.num_regions
        super()._generate_regions()
        self.connections = {
            'low': [(i, j) for i in range(0, n) for j in range(i, n)],
            'high': [(i, i) for i in range(0, n) if i in self.distinguished]
            }