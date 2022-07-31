import pickle
import plotly.figure_factory as ff
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from truncatedGaussianMixture import TruncatedGaussianMixture
from scipy import linalg
import math


def ellipse(x_center, y_center, sigma):
    # x_center, y_center the coordinates of ellipse center
    # ax1 ax2 two orthonormal vectors representing the ellipse axis directions
    # a, b the ellipse parameters
    v, w = linalg.eigh(sigma)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    a = v[1]
    b = v[0]
    x_origin = x_center
    y_origin = y_center
    x_ = []
    y_ = []
    for t in range(0, 1000, 1):
        x = a * (math.cos(math.radians(t))) + x_origin
        x_.append(x)
        y = b * (math.sin(math.radians(t))) + y_origin
        y_.append(y)

    return x_, y_


mu, sigma, s, t, pweights, k, n = [-3, 15, 7], [20, 10, 20], 0, 25, [0.4, 0.3, 0.3], 3, 1000


def summary_plots(mu, sigma, s, t, pweights, k, n):
    with open(
            "".join(("vysl_", str(mu), "_", str(sigma), "_", str(pweights), "_", str(s), "_", str(t), str(n), ".pkl")),
            "rb") as f:
        res = pickle.load(f)
    hist_data = [res['data'][j] for j in range(k)]
    group_labels = ["".join(('cluster ', str(j + 1))) for j in range(k)]

    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.1)
    for i in range(k):
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=res['correct_EM'][i],
                y0=-0.02,
                x1=res['correct_EM'][i],
                y1=0.5,
                line=dict(
                    color="RoyalBlue",
                    width=3
                )
            ))
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=mu[i],
                y0=-0.02,
                x1=mu[i],
                y1=0.5,
                line=dict(
                    color="Black",
                    width=3
                )
            ))
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=res['article_EM'][i],
                y0=-0.02,
                x1=res['article_EM'][i],
                y1=0.5,
                line=dict(
                    color="Red",
                    width=3
                )
            ))
    fig.layout.title = 'EM algorithm'
    fw = go.FigureWidget(fig)
    fw.show()


def summary_plots2(mu, sigma, s, t, pweights, k, n):
    with open(
            'C:/Users/rk621fq/Documents/GitHub/Diploma Thesis/em/vysl_[[-3, 3], [10, -1], [20, 20]]_[[[20, 0], [0, 5]], [[5, 0], [0, 20]], [[20, 0], [0, 20]]]_[0.5, 0.2, 0.3]_[0, 0]_[25, 25]1000.pkl',
            "rb") as f:
        res = pickle.load(f)
    df_pom = {}

    for i in range(k):
        df_pom[i] = pd.DataFrame(res['data'][i], columns=['x', 'y'])
        df_pom[i]['cluster'] = str(i)

    df = pd.concat([df_pom[i] for i in range(k)], ignore_index=True)

    fig = px.scatter(df, x='x', y='y', color='cluster')

    fig.layout.title = 'EM algorithm'
    fw = go.FigureWidget(fig)
    fw.show()


mu, sigma, s, t, pweight, k, n, dim = [[1, 1], [10, 10], [2, 10]], [[[1, 0], [0, 1]], [[1, 0], [0, 1]],
                                                                    [[1, 0], [0, 1]]], [-5, -5], [25, 25], [1 / 3,
                                                                                                            1 / 3,
                                                                                                            1 / 3], 3, 1000, 2


# summary_plots2(mu, sigma, s, t, pweights, k, n)


def summary_plots3(mu, sigma, s, t, pweights, k, n):
    with open(
            'C:/Users/rk621fq/Documents/GitHub/Diploma Thesis/em/vysl_[[-3, 3], [10, -1], [20, 20]]_[[[20, 0], [0, 5]], [[5, 0], [0, 20]], [[20, 0], [0, 20]]]_[0.5, 0.2, 0.3]_[0, 0]_[25, 25]1000.pkl',
            "rb") as f:
        res = pickle.load(f)
    pom = {}
    data = pd.DataFrame(np.concatenate(([res['data'][j] for j in range(k)])), columns=['x', 'y'])
    for step, value in res['correct_EM'].items():
        d = TruncatedGaussianMixture(value['mu'], value['sigma'], s, t, value['weight'], k, n, 2)
        pdf = d.pdf(data)
        p = pd.DataFrame(((pdf * value['weight']) / (np.sum(pdf * value['weight'], axis=1)).reshape(1000, 1)),
                         columns=[''.join(('cluster_', str(j))) for j in range(k)])
        pom[step] = pd.concat([p, data], axis=1)
        pom[step]['step'] = step
    df = pd.concat([pom[step] for step in res['correct_EM']], ignore_index=True)
    df['cluster'] = df['cluster_1'] + 2 * df['cluster_2'] + 1
    fig = px.scatter(df, x='x', y='y', color='cluster', animation_frame="step")

    pom2 = {}
    # for step, value in res['correct_EM'].items():
    #    pom2[step] = pd.DataFrame(value['mu'])
    pom2 = {}

    s = []
    fig2 = {}
    for j in range(k):
        x_ = []
        y_ = []
        s = []
        for step, value in res['correct_EM'].items():
            v, w = linalg.eigh(value['sigma'][j])
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            a = v[1]
            b = v[0]
            x_origin = value['mu'][j][0]
            y_origin = value['mu'][j][1]

            for t in range(0, 360, 1):
                x = a * (math.cos(math.radians(t))) + x_origin
                x_.append(x)
                y = b * (math.sin(math.radians(t))) + y_origin
                y_.append(y)
                s.append(step)
        df2 = pd.DataFrame({'x': np.array(x_), 'y': np.array(y_), 'step': np.array(s)})
        df2['cluster'] = j + 1
        # df = df.append(df2, sort=False)

        # fig2[j] = px.scatter(df2, x = 'x', y = 'y',  animation_frame = "step").update_traces(mode='lines')
    # f.add_trace(fig)
    # f.add_trace(fig2[0])
    # fig.update_traces([fig[j] for j in range(k)])
    fig.layout.title = 'EM algorithm'
    fig.layout.yaxis = dict(range=[-10, 35])
    fig.layout.xaxis = dict(range=[-20, 35])

    xx, yy = ellipse(res['real_params'][0][0][0], res['real_params'][0][0][1], res['real_params'][1][0])
    fig.add_scatter(x=xx, y=yy, mode='lines')
    xx, yy = ellipse(res['real_params'][0][1][0], res['real_params'][0][1][1], res['real_params'][1][1])
    fig.add_scatter(x=xx, y=yy, mode='lines')
    xx, yy = ellipse(res['real_params'][0][2][0], res['real_params'][0][2][1], res['real_params'][1][2])
    fig.add_scatter(x=xx, y=yy, mode='lines')

    fig.show()


mu, sigma, s, t, pweight, k, n, dim = [[1, 1], [10, 10], [2, 10]], [[[1, 0], [0, 1]], [[1, 0], [0, 1]],
                                                                    [[1, 0], [0, 1]]], [-5, -5], [25, 25], [1 / 3,
                                                                                                            1 / 3,
                                                                                                            1 / 3], 3, 1000, 2
summary_plots3(mu, sigma, s, t, pweights, k, n)
