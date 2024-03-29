#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go


class ExplainerInterface:
    def __init__(self):
        self._generate_mus()
    def _generate_mus(self):
        pass
    def get_l_distribution(self):
        pass
    def get_h_distribution(self):
        pass
    def draw_distribtutions(self):
        pass


class UniformExplainer(ExplainerInterface):
    num_steps = 5
    mu = 0.5
    len_range = 0.5
    final_overlap = 0.1
    d = (len_range - final_overlap) / (2 * num_steps)
    def __init__(self):
        super().__init__()
        print(self.d)

    def _generate_mus(self):
        self.mus = [(self.mu, self.mu)] # loc, scale => [loc, loc + scale]
        for _ in range(self.num_steps):
            mu_l = self.mus[-1][0] - self.d
            mu_h = self.mus[-1][1] + self.d
            self.mus.append((mu_l, mu_h))
        print(self.mus)

    def get_l_distribution(self, d):
        assert 0 <= d <= self.num_steps
        return {'loc': self.mus[d][0], 'scale': self.len_range}
    
    def get_h_distribution(self, d):
        assert 0 <= d <= self.num_steps
        return {'loc': self.mus[d][1], 'scale': self.len_range}

    def draw_distribtutions(self):
        for mu_l, mu_h in self.mus:
            x_l = np.linspace(0, 1, 500)
            x_h = np.linspace(0, 1, 500)
            fig = plt.figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot()
            print(mu_l, mu_h)
            ax.plot(x_l, stats.uniform.pdf(x_l, mu_l - self.len_range / 2, self.len_range))
            ax.plot(x_h, stats.uniform.pdf(x_h, mu_h - self.len_range / 2, self.len_range))
            ax.axvline(x=0, c='black')
            ax.axvline(x=1, c='black')
            plt.show()

    def draw_distributions_plotly(self):
        for mu_l, mu_h in self.mus:
            x_l = np.linspace(0, 1, 1000)
            x_h = np.linspace(0, 1, 1000)
            y_l = stats.uniform.pdf(x_l, mu_l - self.len_range / 2, self.len_range)
            y_h = stats.uniform.pdf(x_h, mu_h - self.len_range / 2, self.len_range)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_l, y=y_l, mode='lines', name='Low', line=dict(width=7)))
            fig.add_trace(go.Scatter(x=x_h, y=y_h, mode='lines', name='High', line=dict(width=7)))
            fig.add_vline(x=0, line_width=3), fig.add_vline(x=1, line_width=3)
            fig.add_vline(x=mu_l, line_width=5, line_dash="dash", line_color="blue")
            fig.add_vline(x=mu_h, line_width=5, line_dash="dash", line_color="red")
            fig.update_layout(width=1000, height=500, xaxis_range=[-0.1, 1.1],
                            font=dict(size=50))
            fig.show()


class GaussianExplainer(ExplainerInterface):
    num_steps = 5
    mu = 0.5
    d = 0.0675684
    sigma = 0.168921
    # sigma = sigma ** 2
    # mul_5 = 0.331079
    # muh_5 = 0.668921
    def __init__(self):
        super().__init__()

    def _generate_mus(self):
        self.mus = [(self.mu, self.mu)]
        for n in range(self.num_steps):
            mu_l = self.mus[n][0] - 0.5 * self.d
            mu_h = self.mus[n][1] + 0.5 * self.d
            self.mus.append((mu_l, mu_h))

    def get_l_distribution(self, d):
        '''
        at d=0 l distribution and h distribution are the same
        '''
        assert 0 <= d <= self.num_steps
        return {'mu': self.mus[d][0], 'sigma': self.sigma}

    def get_h_distribution(self, d):
        assert 0 <= d <= self.num_steps
        return {'mu': self.mus[d][1], 'sigma': self.sigma}

    def draw_distribtutions(self):
        for mu_l, mu_h in self.mus:
            x_l = np.linspace(mu_l - 10*self.sigma, mu_l + 10*self.sigma, 100)
            x_h = np.linspace(mu_h - 10*self.sigma, mu_h + 10*self.sigma, 100)
            fig = plt.figure(figsize=(10, 5), dpi=100)
            ax = fig.add_subplot()
            ax.plot(x_l, stats.norm.pdf(x_l, mu_l, self.sigma))
            ax.plot(x_h, stats.norm.pdf(x_h, mu_h, self.sigma))
            ax.axvline(x=0, c='black')
            ax.axvline(x=1, c='black')
            plt.show()
    
    def draw_distributions_plotly(self):
        for mu_l, mu_h in self.mus:
            x_l = np.linspace(mu_l - 10*self.sigma, mu_l + 10*self.sigma, 2000)
            y_l = stats.norm.pdf(x_l, mu_l, self.sigma)
            x_h = np.linspace(mu_h - 10*self.sigma, mu_h + 10*self.sigma, 2000)
            y_h = stats.norm.pdf(x_h, mu_h, self.sigma)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_l, y=y_l, mode='lines', name='Low', line=dict(width=7)))
            fig.add_trace(go.Scatter(x=x_h, y=y_h, mode='lines', name='High', line=dict(width=7)))
            fig.add_vline(x=0, line_width=3), fig.add_vline(x=1, line_width=3)
            fig.add_vline(x=mu_l, line_width=5, line_dash="dash", line_color="blue")
            fig.add_vline(x=mu_h, line_width=5, line_dash="dash", line_color="red")
            fig.update_layout(width=2000, height=1000, xaxis_range=[-0.1, 1.1],
                            font=dict(size=50))
            fig.show()

if __name__ == '__main__':
    explainer = UniformExplainer()
    # print(np.round(explainer.mus, 6))
    explainer.draw_distributions_plotly()

# %%
