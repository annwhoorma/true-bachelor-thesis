#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

class Explainer:
    mu = 0.509187
    d = 0.077008
    sigma = 0.19252
    steps = 5
    def __init__(self):
        self._generate_mus()

    def _generate_mus(self):
        self.mus = [(self.mu, self.mu)]
        for i in range(self.steps):
            mu_l = self.mus[i][0] - 0.5 * self.d
            mu_h = self.mus[i][1] + 0.5 * self.d
            self.mus.append((mu_l, mu_h))

    def get_l_distribution(self, d):
        '''
        at d=0 l distribution and h distribution are the same
        '''
        assert 0 <= d <= self.steps
        return {'mu': self.mus[d][0], 'sigma': self.sigma}
    
    def get_h_distribution(self, d):
        assert 0 <= d <= self.steps
        return {'mu': self.mus[d][1], 'sigma': self.sigma}
    
    def draw_distribtutions(self):
        for mu_l, mu_h in self.mus:
            x_l = np.linspace(mu_l - 10*self.sigma, mu_l + 10*self.sigma, 100)
            x_h = np.linspace(mu_h - 10*self.sigma, mu_h + 10*self.sigma, 100)
            fig = plt.figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot()
            ax.plot(x_l, stats.norm.pdf(x_l, mu_l, self.sigma))
            ax.plot(x_h, stats.norm.pdf(x_h, mu_h, self.sigma))
            ax.axvline(x=0, c='black')
            ax.axvline(x=1, c='black')
            plt.show()
    
# explainer = Explainer()
# print(np.round(explainer.mus, 6), explainer.sigma)
# explainer.draw_distribtutions()
# %%
