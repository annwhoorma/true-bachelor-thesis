#%%
import pandas as pd
import numpy as np
import plotly.express as px
#%%
def visualize_heatmap(df, metric_name, title):
    nr, nnpr, z = df['num_regions'], df['num_nodes_per_region'], np.array(df[metric_name])
    data = np.array(z)
    data = z.reshape(len(set(nr)), len(set(nnpr)))
    fig = px.imshow(data,
                    labels=dict(x="x: # nodes per region", y="y: # regions", color=metric_name),
                    y=sorted(set(nr)),
                    x=sorted(set(nnpr)),
                    color_continuous_scale='Blues',
                    # zmin=-1, zmax=1
                )
    fig.update_layout(title=title, height=500)
    return fig

def visualize_all_coeffs(df, metric_name, color, title):
    fig = px.line(df, x='num_regions', y=metric_name, color=color)
    fig.update_layout(title=title, height=500)
    return fig

#%%
df = pd.read_csv('modul_particip_nn_nnpr.csv')
int_df = df[df['net_type'] == 'integration'].drop(['net_type'], axis=1)
seg_df = df[df['net_type'] == 'segregation'].drop(['net_type'], axis=1)

#%%
# HEATMAPS
all_dfs = {
    'integration': {
        'bct_modularity': int_df.drop(['bct_participation'], axis=1),
        'bct_participation': int_df.drop(['bct_modularity'], axis=1)
    },
    'segregation': {
        'bct_modularity': seg_df.drop(['bct_participation'], axis=1),
        'bct_participation': seg_df.drop(['bct_modularity'], axis=1)
    }
}
for net_type in all_dfs:
    print(net_type)
    for metric_name, df in all_dfs[net_type].items():
        fig = visualize_heatmap(df, metric_name, f'{metric_name} for {net_type}')
        fig.show()

# %%

