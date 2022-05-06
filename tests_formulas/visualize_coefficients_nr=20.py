#%%
from turtle import width
from matplotlib import markers
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from os import listdir

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
    fig.update_layout(title=title, height=500, template='plotly_dark')
    return fig

def visualize_coeff(df, metric_name, title):
    fig = px.line(df, x='num_added_conns', y=metric_name, color='net_type')
    fig.update_layout(title=title, height=300, font=dict(size=15))
    return fig

# visualize integrated -> fully-connected -> segregated
def visualize_spectrum(df, metric_name, vline, title):
    # add vertical lines at when it's FC
    fig = px.line(df, x='step', y=metric_name)
    fig.add_vline(x=vline, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(title=title, height=500, width=1000, font=dict(size=15))
    return fig

def visualize_spectra(dfs, metric_name, vline, titles):
    '''
    for NR/NNPR = 0.5, 1, 2
    '''
    fig = go.Figure()
    for i, df in enumerate(dfs):
        fig.add_trace(go.Scatter(x=df['step'], y=df[metric_name], mode='lines', name=titles[i]))
    fig.update_layout(title=metric_name, height=500, width=1000, font=dict(size=15))
    fig.add_vline(x=vline, line_width=2, line_dash="dash", line_color="red")
    return fig


def visualize_for_fixed_nnpr(df: pd.DataFrame, metric_name, vline, df_name):
    title = f'{metric_name} for {df_name}'
    fig = go.Figure()
    unique_nr = df['nr'].unique()
    for nr in unique_nr:
        ldf = df[df['nr'] == nr]
        fig.add_trace(go.Scatter(x=ldf['step'], y=ldf[metric_name], mode='lines', name=f'nr = {nr}'))
    fig.update_layout(title=title, height=500, width=1000, font=dict(size=15))
    fig.add_vline(x=vline, line_width=2, line_dash="dash", line_color="red")
    return fig

def visualize_parametric_dep(df: pd.DataFrame, metric1, metric2, title):
    fig = go.Figure()
    unique_nr = df['nr'].unique()
    for nr in unique_nr:
        ldf = df[df['nr'] == nr]
        fig.add_trace(go.Scatter(x=ldf[metric1], y=ldf[metric2], mode='lines+markers', name=f'nr = {nr}'))
    fig.update_layout(title=title,
                    xaxis_title=metric1,
                    yaxis_title=metric2)
    return fig

#%%
df = pd.read_csv('coefficients_nr=20.csv')
int_df = df[df['net_type'] == 'integration'].drop(['net_type'], axis=1)
seg_df = df[df['net_type'] == 'segregation'].drop(['net_type'], axis=1)

visualize_coeff(int_df, 'bct_modularity', 'modularity for integration').show()
visualize_coeff(int_df, 'bct_participation', 'participation for integration').show()
visualize_coeff(seg_df, 'bct_modularity', 'modularity for segregation').show()
visualize_coeff(seg_df, 'bct_participation', 'participation for segregation').show()
# %%
df = pd.read_csv('coeffs_i2s.csv')

fc_line = df[df['net_type'] == 'fully-connected']
_, step_num, fc_modul, fc_particip = fc_line.values[0].tolist()

visualize_spectrum(df, 'bct_modularity', step_num, '1').show()
visualize_spectrum(df, 'bct_participation', step_num, '2').show()

# %%
df0 = pd.read_csv('coeffs_i2s_nr_nnpr_0.25.csv')
df1 = pd.read_csv('coeffs_i2s_nr_nnpr_0.5.csv')
df2 = pd.read_csv('coeffs_i2s_nr_nnpr_1.0.csv')
df3 = pd.read_csv('coeffs_i2s_nr_nnpr_2.0.csv')

fc_line = df0[df0['net_type'] == 'fully-connected']
_, step_num, _, _ = fc_line.values[0].tolist()

visualize_spectra([df0, df1, df2, df3], 'bct_modularity', step_num, ['NR/NNPR=0.25', 'NR/NNPR=0.5', 'NR/NNPR=1', 'NR/NNPR=2']).show()
visualize_spectra([df0, df1, df2, df3], 'bct_participation', step_num, ['NR/NNPR=0.25', 'NR/NNPR=0.5', 'NR/NNPR=1', 'NR/NNPR=2']).show()
# %%
folder = 'for_fixed_nnpr_with_old'
csvs = listdir(folder)
for csv_file in csvs:
    df_name = csv_file.rstrip('.csv')
    df = pd.read_csv(f'{folder}/{csv_file}')
    fc_line = df[df['net_type'] == 'fully-connected']
    _, _, step_num, _, _, _, _ = fc_line.values[0].tolist()
    visualize_for_fixed_nnpr(df, 'bct_modularity', step_num, df_name).show()
    visualize_for_fixed_nnpr(df, 'bct_participation', step_num, df_name).show()
    visualize_for_fixed_nnpr(df, 'clustering', step_num, df_name).show()
    visualize_for_fixed_nnpr(df, 'efficiency', step_num, df_name).show()
# %%

def visualize_parametric_dep(df: pd.DataFrame, metric1, metric2, title):
    colors = ['blue', 'red', 'green']
    fig = go.Figure()
    unique_nr = df['nr'].unique()
    for nr, color in zip(unique_nr, colors):
        ldf = df[df['nr'] == nr]
        fig.add_trace(go.Scatter(x=ldf[metric1], y=ldf[metric2], mode='lines+markers', name=f'nr = {nr}', marker_color=color))
        midpoint = ldf[ldf['net_type'] == 'fully-connected']
        print(midpoint[metric1].values[0], midpoint[metric2].values[0])
        x, y = midpoint[metric1].values[0], midpoint[metric2].values[0]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker_color=color, marker_size=15, name=f'nr = {nr}: FC'))
    fig.update_layout(title=title,
                    xaxis_title=metric1,
                    yaxis_title=metric2)
    return fig

folder = 'for_fixed_nnpr_with_old'
csv_file = 'nnpr=18.csv'
df = pd.read_csv(f'{folder}/{csv_file}')
visualize_parametric_dep(df, 'modularity', 'clustering', csv_file).show()
visualize_parametric_dep(df, 'participation', 'efficiency', csv_file).show()
# %%
