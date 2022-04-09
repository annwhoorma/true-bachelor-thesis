from labels.label_interface import LabelInterface

class Label2(LabelInterface):
    '''neutral'''
    def __init__(self, mask, label_name, low_dist, high_dist):
        super(Label2, self).__init__(mask, label_name, low_dist, high_dist)
        self._generate_regions()
        self._generate_patterns()
        self.A *= self.mask

    def _generate_regions(self):
        n = self.num_regions
        super()._generate_regions()
        self.connections = {
            'low': [(i, j) for i in range(0, n) for j in range(i, n)],
            }