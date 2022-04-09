from labels.label_interface import LabelInterface

class Label2(LabelInterface):
    '''neutral'''
    def __init__(self, num_nodes, num_edges, mask, label_name):
        super(Label2, self).__init__(num_nodes, num_edges, mask, label_name)
        self._generate_regions()
        self._generate_patterns()
        self.A *= self.mask

    def _generate_regions(self):
        n = self.num_regions
        super()._generate_regions()
        self.connections = {
            'low': [(i, j) for i in range(0, n) for j in range(i, n)],
            }