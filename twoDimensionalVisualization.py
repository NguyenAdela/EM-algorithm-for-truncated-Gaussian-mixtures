import pickle
import pylab as pl
import matplotlib as mpl
import plotly
import plotly.figure_factory as ff
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy import linalg
import math
import matplotlib as mpl
from scipy.stats import norm, chi2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import multivariate_normal

import numpy as np

from truncatedGaussianMixture import TruncatedGaussianMixture


class SummaryPlot2D:

    def __init__(self, results):
        self.s, self.t = results['real_params']['s'], results['real_params']['t']
        self.k = results['real_params']['k']
        self.n = len(results['correct_EM'])
        self.nalg = len(results['article_EM'])
        self.order_correct = {}
        self.order_article = {}
        nodes_correct = results['correct_EM'][len(results['correct_EM']) - 1]['mean']
        nodes_article = results['article_EM'][len(results['article_EM']) - 1]['mean']
        for k in range(self.k):
            self.order_correct[k] = np.argmin([np.dot(
                (nodes_correct[l] - results['real_params']['mean'][k]).T,
                nodes_correct[l] - results['real_params']['mean'][k]
            ) for l in range(self.k)])
            self.order_article[k] = np.argmin([np.dot(
                (nodes_article[l] - results['real_params']['mean'][k]).T,
                nodes_article[l] - results['real_params']['mean'][k]
            ) for l in range(self.k)])

        self.mean_true, self.variance_true, self.weights_true = results['real_params']['mean'], results['real_params'][
            'variance'], results['real_params']['weight']
        self.pweight_true = results['real_params']['pweight']
        self.mean_em, self.variance_em, self.weights_em = [results['correct_EM'][i]['mean'] for i in range(self.n)], [
            results['correct_EM'][i]['variance'] for i in range(self.n)], [results['correct_EM'][i]['weight'] for i in
                                                                           range(self.n)]
        self.mean_article, self.variance_article, self.weights_article = [results['article_EM'][i]['mean'] for i in
                                                                          range(self.nalg)], [
                                                                             results['article_EM'][i]['variance'] for i
                                                                             in
                                                                             range(self.nalg)], [
                                                                             results['article_EM'][i]['weight'] for i in
                                                                             range(self.nalg)]
        self.log_likelihood = [results['correct_EM'][i]['log-likelihood'] for i in range(len(results['correct_EM']))]
        self.log_likelihood_article = [results['article_EM'][i]['log-likelihood'] for i in
                                       range(len(results['article_EM']))]

        self.data = [
            pd.DataFrame([[results['data'][j][k].tolist()] for k in range(len(results['data'][j]))], columns=['x']) for
            j in range(self.k)]
        self.flatten_data = []
        for k in range(self.k):
            for i in range(len(results['data'][k])):
                self.flatten_data.append(results['data'][k][i].tolist())

        self.flatten_data = self.flatten_data
        self.time = results['time']
        self.time_article = results['time_article']
        self.n_points = len(self.flatten_data)

    def classify(self, x, mean, variance, weight):
        pom = []
        for k in range(self.k):
            pom.append((1 / (multivariate_normal.cdf(self.t, mean[k],
                                                     variance[k]) - multivariate_normal.cdf(self.s,
                                                                                            mean[k],
                                                                                            variance[
                                                                                                k]))) * (
                               weight[k] * multivariate_normal.pdf(x, mean[k],
                                                                   variance[k])))
        return np.argmax(np.array(pom))

    def data_plot(self):
        fig = plt.figure()
        ax1 = plt.subplot(131)
        picked_colors = np.array([
            [[220, 179, 15], [37, 117, 74], [192, 57, 43], [187, 187, 187], [27, 163, 156]]
            [k] for k in range(self.k)
        ]) / 255.0
        for j in range(self.k):
            ax1.scatter(
                x=[self.data[j].x[i][0] for i in range(len(self.data[j]))],
                y=[self.data[j].x[i][1] for i in range(len(self.data[j]))],
                color=picked_colors[j],
                zorder=1
            )

            ax1.scatter(
                x=[self.mean_true[j][0]],
                y=[self.mean_true[j][1]],
                color='black', marker="x", linewidths=3, s=100,
                zorder=3, label=None
            )

            vals, vecs = np.linalg.eigh(self.variance_true[j])
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]

            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            r2 = chi2.ppf(0.95, 1)
            width, height = 2 * np.sqrt(vals * r2)
            ell = mpl.patches.Ellipse(self.mean_true[j], width, height, 180 + angle, color=picked_colors[j])
            #plt.xlim([-1.5, 26.5])
            #plt.ylim([-1.5, 26.5])
            ell.set_clip_box(fig.bbox)
            ell.set_alpha(0.35)
            ax1.add_artist(ell)
        plt.title(label='True distribution', fontsize=20)
        left, bottom, width, height = (self.s[0], self.s[1], self.t[0] - self.s[0], self.t[1] - self.s[1])
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  fill=False,
                                  color="black",
                                  linewidth=1)
        plt.gca().add_patch(rect)
        ax1.set_facecolor('white')
        ax2 = plt.subplot(132)
        data_correct = {}
        for k in range(self.k):
            data_correct[k] = []
        for x in self.flatten_data:
            cluster = self.classify(x, [self.mean_em[-1][self.order_correct[j]] for j in range(self.k)],
                                    [self.variance_em[-1][self.order_correct[j]] for j in range(self.k)],
                                    [self.weights_em[-1][self.order_correct[j]] for j in range(self.k)])

            data_correct[cluster].append(x)
        data_correct = [pd.DataFrame([[data_correct[j][k]] for k in range(len(data_correct[j]))], columns=['x']) for j
                        in range(self.k)]

        for j in range(self.k):
            ax2.scatter(

                x=[data_correct[self.order_correct[j]].x[i][0] for i in
                   range(len(data_correct[self.order_correct[j]]))],
                y=[data_correct[self.order_correct[j]].x[i][1] for i in
                   range(len(data_correct[self.order_correct[j]]))],

                color=picked_colors[self.order_correct[j]], zorder=1

            )

            vals, vecs = np.linalg.eigh(self.variance_em[-1][self.order_correct[j]])
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]

            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            r2 = chi2.ppf(0.95, 1)
            width, height = 2 * np.sqrt(vals * r2)
            ell = mpl.patches.Ellipse(self.mean_em[-1][self.order_correct[j]], width, height, 180 + angle,
                                      color=picked_colors[j])
            #plt.xlim([-1.5, 26.5])
            #plt.ylim([-1.5, 26.5])

            ell.set_clip_box(fig.bbox)
            ell.set_alpha(0.35)
            ax2.add_artist(ell)
            ax2.scatter(
                x=[self.mean_em[-1][self.order_correct[j]][0]],
                y=[self.mean_em[-1][self.order_correct[j]][1]],

                color='black', marker="x", linewidths=3, s=100, zorder=3, label=None
            )
        plt.title(label='Thesis algorithm', fontsize=20)
        left, bottom, width, height = (self.s[0], self.s[1], self.t[0] - self.s[0], self.t[1] - self.s[1])
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  fill=False,
                                  color="black",
                                  linewidth=1)

        plt.gca().add_patch(rect)
        ax2.set_facecolor('white')
        ax3 = plt.subplot(133)
        #plt.xlim([-1.5, 26.5])
        #plt.ylim([-1.5, 26.5])
        data_article = {}
        for k in range(self.k):
            data_article[k] = []
        for x in self.flatten_data:
            cluster = self.classify(x, [self.mean_article[-1][self.order_correct[j]] for j in range(self.k)],
                                    [self.variance_article[-1][self.order_correct[j]] for j in range(self.k)],
                                    [self.weights_article[-1][self.order_correct[j]] for j in range(self.k)])

            data_article[cluster].append(x)
        data_article = [pd.DataFrame([[data_article[j][k]] for k in range(len(data_article[j]))], columns=['x']) for j
                        in range(self.k)]

        for j in range(self.k):
            ax3.scatter(

                x=[data_article[self.order_correct[j]].x[i][0] for i in
                   range(len(data_article[self.order_correct[j]]))],
                y=[data_article[self.order_correct[j]].x[i][1] for i in
                   range(len(data_article[self.order_correct[j]]))],

                color=picked_colors[self.order_correct[j]], zorder=1

            )
            vals, vecs = np.linalg.eigh(self.variance_article[-1][self.order_correct[j]])
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            r2 = chi2.ppf(0.95, 1)
            width, height = 2 * np.sqrt(vals * r2)
            ell = mpl.patches.Ellipse(self.mean_article[-1][self.order_correct[j]], width, height, 180 + angle,
                                      color=picked_colors[j])
            ell.set_clip_box(fig.bbox)
            ell.set_alpha(0.35)
            ax3.add_artist(ell)
            ax3.scatter(
                x=[self.mean_article[-1][self.order_correct[j]][0]],
                y=[self.mean_article[-1][self.order_correct[j]][1]],
                color='black', marker="x", linewidths=3, s=100, zorder=3, label=None
            )
        plt.title(label='Article algorithm', fontsize=20)
        left, bottom, width, height = (self.s[0], self.s[1], self.t[0] - self.s[0], self.t[1] - self.s[1])
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  fill=False,
                                  color="black",
                                  linewidth=1)
        plt.gca().add_patch(rect)
        ax3.set_facecolor('white')
        ax2.sharey(ax1)
        ax3.sharey(ax1)
        ax2.sharex(ax1)
        ax3.sharex(ax1)
        plt.show()
        plt.tight_layout()
        fig.figure.savefig("images/hist_2d" + str(self.k) + "c_" + "_".join(
            [str(self.mean_true[k]) for k in range(self.k)]) + "_" + str(self.n_points) + ".png",
                           width=2, height=1)

    def log_likelihood_plot(self):
        titles = ['Thesis algorithm', 'Article algorithm']
        self.fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_titles=titles, x_title='Iteration',
                                 horizontal_spacing=0.03)
        self.fig.add_trace(go.Scatter(
            y=self.log_likelihood,
            x=np.array(range(0, len(self.log_likelihood))),
            line=dict(
                color='red',
                width=1,
            ),
            showlegend=False),
            row=1, col=1)
        self.fig.add_trace(go.Scatter(
            y=self.log_likelihood_article,
            x=np.array(range(0, len(self.log_likelihood_article))),
            line=dict(
                color='red',
                width=1,
            ),
            showlegend=False),
            row=1, col=2)
        self.fig.update_layout(title_text="Convergence of log-likelihood function", title_font=dict(size=22))
        self.fig.update_annotations(font_size=22)
        self.fig.show()
        self.fig.write_image(
            "images/ll_2d" + str(self.k) + "c_" + "_".join([str(self.mean_true[k]) for k in range(self.k)]) + "_" + str(
                self.n_points) + ".png",
            width=2, height=1)

    def output_table(self):
        pokus = TruncatedGaussianMixture(
            self.mean_true, self.variance_true, self.s, self.t, self.pweight_true, self.k, 2000, 2, seed=23)
        data, weights = pokus.simulate()
        KL_divergence_em = self.kl_divergence_2d(data, self.weights_em[-1], self.mean_em[-1], self.variance_em[-1])
        KL_divergence_article = self.kl_divergence_2d(data, self.weights_article[-1], self.mean_article[-1], self.variance_article[-1])

        print('===============================================================')
        print('===============================================================')
        print()
        print('Summary table')
        print()
        print('===============================================================')
        print('===============================================================')
        print()
        print()
        print(f'Number of clusters:            {str(self.k)}')
        print(f'Number of data points:         {str(len(self.flatten_data))}')
        print(f'Real parameters:')
        for k in range(self.k):
            print(f'    mean {str(k + 1)}:                    {self.mean_true[k]}')
            print(f'    variance {str(k + 1)}:                {[self.variance_true[k][0], self.variance_true[k][0]]}')
            print(f'    weight {str(k + 1)}:                {self.weights_true[k]}')
        print('===============================================================')
        print('Thesis algorithm')
        print()
        print(f'Thesis estimated parameters:')
        for k in range(self.k):
            print(f'    mean {str(k + 1)}:                    {self.mean_em[-1][self.order_correct[k]]}')
            print(
                f'    variance {str(k + 1)}:                {[self.variance_em[-1][self.order_correct[k]][0].tolist(), self.variance_em[-1][self.order_correct[k]][1].tolist()]}')
            print(f'    weight {str(k + 1)}:                {self.weights_em[-1][self.order_correct[k]]}')
        print(f'Number of steps:               {self.n-1}')
        print(f'Log-likelihood:                {self.log_likelihood[-1]}')
        print(f'Calculation time:              {self.time}')
        print(f'KL score:                      {KL_divergence_em}')
        print('===============================================================')
        print('Article algorithm')
        print()
        print(f'Article estimated parameters:')
        for k in range(self.k):
            print(f'    mean {str(k + 1)}:                    {self.mean_article[-1][self.order_correct[k]]}')
            print(
                f'    variance {str(k + 1)}:                {[self.variance_article[-1][self.order_correct[k]][0].tolist(), self.variance_article[-1][self.order_correct[k]][1].tolist()]}')
            print(f'    weight {str(k + 1)}:                {self.weights_article[-1][self.order_correct[k]]}')
        print(f'Number of steps:               {self.nalg-1}')
        print(f'Log-likelihood:                {self.log_likelihood_article[-1]}')
        print(f'Calculation time:              {self.time_article}')
        print(f'KL score:                      {KL_divergence_article}')
        print('===============================================================')


    def pdf_2d(self, x, weight, mu, sigma):
        pom=[]
        for k in range(len(weight)):
            pom.append(
                weight[k] * multivariate_normal.pdf(
                    x, mu[k], sigma[k]
                )/(multivariate_normal.cdf(
                    self.t, mu[k],sigma[k]
                )-multivariate_normal.cdf(
                    self.s, mu[k],sigma[k])
                   ))
        return np.sum(pom)


    def kl_divergence_2d(self, data, weight, mu, sigma):

        kl = (1/2000)*np.sum([(np.log(self.pdf_2d(x_i, self.weights_true, self.mean_true, self.variance_true))-np.log(self.pdf_2d(x_i, weight, mu, sigma))) for x_i in data])
        return kl





class SummaryPlot2Dreal():

    def __init__(self, results, k, data, classic):
        self.mean_c = classic['mu']
        self.variance_c = classic['sigma']
        self.weights_c = classic['weights']
        self.s, self.t = [0, 0], [1, 1]
        self.k = k
        self.n = len(results['correct_EM'])
        self.nalg = len(results['article_EM'])

        self.order_correct = {}
        self.order_article = {}
        nodes_classic = classic['mu']
        nodes_correct = results['correct_EM'][len(results['correct_EM']) - 1]['mean']
        nodes_article = results['article_EM'][len(results['article_EM']) - 1]['mean']

        for k in range(self.k):
            self.order_correct[k] = np.argmin([np.dot(
                (nodes_correct[l]-nodes_classic[k]).T, nodes_correct[l]-nodes_classic[k]
            ) for l in range(self.k)])

            self.order_article[k] = np.argmin([np.dot(
                (nodes_article[l]-nodes_classic[k]).T, nodes_article[l]-nodes_classic[k]
            ) for l in range(self.k)])


        self.mean_em, self.variance_em, self.weights_em = [results['correct_EM'][i]['mean'] for i in range(self.n)], [
            results['correct_EM'][i]['variance'] for i in range(self.n)], [results['correct_EM'][i]['weight'] for i in
                                                                           range(self.n)]
        self.mean_article, self.variance_article, self.weights_article = [results['article_EM'][i]['mean'] for i in
                                                                          range(self.nalg)], [
                                                                             results['article_EM'][i]['variance'] for i
                                                                             in
                                                                             range(self.nalg)], [
                                                                             results['article_EM'][i]['weight'] for i in
                                                                             range(self.nalg)]
        self.log_likelihood = [results['correct_EM'][i]['log-likelihood'] for i in range(len(results['correct_EM']))]
        self.log_likelihood_article = [results['article_EM'][i]['log-likelihood'] for i in
                                       range(len(results['article_EM']))]
        self.data = data

    def classify(self, x, mean, variance, weight):
        pom = []
        for k in range(self.k):
            pom.append((1 / (multivariate_normal.cdf(self.t, mean[k],
                                                     variance[k]) - multivariate_normal.cdf(self.s,
                                                                                            mean[k],
                                                                                            variance[
                                                                                                k]))) * (
                               weight[k] * multivariate_normal.pdf(x, mean[k],
                                                                   variance[k])))
        return np.argmax(np.array(pom))

    def output_plot(self):
        fig = plt.figure()

        ax0 = plt.subplot(131)
        picked_colors = np.array([
            [[220, 179, 15], [37, 117, 74], [192, 57, 43], [187, 187, 187], [27, 163, 156],[2,48,71],[197,216,26],
             [220, 179, 15], [37, 117, 74], [192, 57, 43], [187, 187, 187], [27, 163, 156],[2,48,71],[197,216,26]]
            [k] for k in range(self.k)
        ]) / 255.0

        data_classic = {}
        for j in range(self.k):
            data_classic[j] = []
        for index, x in self.data.iterrows():
            cluster = self.classify(x.values[0], [self.mean_c[j] for j in range(self.k)],
                               [self.variance_c[j] for j in range(self.k)],
                               [self.weights_c[j] for j in range(self.k)])

            data_classic[cluster].append(x.values[0])
        data_classic = [pd.DataFrame([[data_classic[j][k]] for k in range(len(data_classic[j]))], columns=['x']) for j
                        in range(self.k)]

        for j in range(self.k):
            ax0.scatter(

                x=[data_classic[j].x[i][0] for i in range(len(data_classic[j]))],
                y=[data_classic[j].x[i][1] for i in range(len(data_classic[j]))],

                color=picked_colors[j], zorder=1

            )

            vals, vecs = np.linalg.eigh(self.variance_c[j])
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]

            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            r2 = chi2.ppf(0.95, 1)

            width, height = 2 * np.sqrt(vals * r2)

            ell = mpl.patches.Ellipse(self.mean_c[j], width, height, 180 + angle, color=picked_colors[j])

            ell.set_clip_box(fig.bbox)
            ell.set_alpha(0.35)
            ax0.add_artist(ell)
            ax0.scatter(
                x=[self.mean_c[j][0]],
                y=[self.mean_c[j][1]],

                color='black', marker="x", linewidths=3, s=100, zorder=3
            )
        plt.title(label='Classic algorithm', fontsize=20)

        left, bottom, width, height = (self.s[0], self.s[1], self.t[0] - self.s[0], self.t[1] - self.s[1])
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  fill=False,
                                  color="black",
                                  linewidth=1)
        plt.gca().add_patch(rect)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])


        ax1 = plt.subplot(132)
        picked_colors = np.array([
            [[220, 179, 15], [37, 117, 74], [192, 57, 43], [187, 187, 187], [27, 163, 156],[2,48,71],[197,216,26],
             [220, 179, 15], [37, 117, 74], [192, 57, 43], [187, 187, 187], [27, 163, 156],[2,48,71],[197,216,26]]
            [k] for k in range(self.k)
        ]) / 255.0
        titles = ['True distribution', 'Thesis algorithm', 'Article algorithm']


        data_correct = {}
        for k in range(self.k):
            data_correct[k] = []
        for index, x in self.data.iterrows():
            cluster = self.classify(x.values[0], [self.mean_em[-1][j] for j in range(self.k)],
                                    [self.variance_em[-1][j] for j in range(self.k)],
                                    [self.weights_em[-1][j] for j in range(self.k)])

            data_correct[cluster].append(x.values[0])
        data_correct = [pd.DataFrame([[data_correct[j][k]] for k in range(len(data_correct[j]))], columns=['x']) for j
                        in range(self.k)]

        for j in range(self.k):
            ax1.scatter(

                x=[data_correct[j].x[i][0] for i in range(len(data_correct[j]))],
                y=[data_correct[j].x[i][1] for i in range(len(data_correct[j]))],

                color=picked_colors[j], zorder=1

            )

            vals, vecs = np.linalg.eigh(self.variance_em[-1][j])
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]

            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            r2 = chi2.ppf(0.95, 1)

            width, height = 2 * np.sqrt(vals * r2)

            ell = mpl.patches.Ellipse(self.mean_em[-1][j], width, height, 180 + angle, color=picked_colors[j])

            ell.set_clip_box(fig.bbox)
            ell.set_alpha(0.35)
            ax1.add_artist(ell)
            ax1.scatter(
                x=[self.mean_em[-1][j][0]],
                y=[self.mean_em[-1][j][1]],

                color='black', marker="x", linewidths=3, s=100, zorder=3
            )
        plt.title(label='Thesis algorithm', fontsize=20)

        left, bottom, width, height = (self.s[0], self.s[1], self.t[0] - self.s[0], self.t[1] - self.s[1])
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  fill=False,
                                  color="black",
                                  linewidth=1)
        plt.gca().add_patch(rect)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])


        ax2 = plt.subplot(133)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])

        data_article = {}
        for k in range(self.k):
            data_article[k] = []
        for index, x in self.data.iterrows():
            cluster = self.classify(x.values[0], [self.mean_article[-1][j] for j in range(self.k)],
                                    [self.variance_article[-1][j] for j in range(self.k)],
                                    [self.weights_article[-1][j] for j in range(self.k)])

            data_article[cluster].append(x.values[0])
        data_article = [pd.DataFrame([[data_article[j][k]] for k in range(len(data_article[j]))], columns=['x']) for j
                        in range(self.k)]

        for j in range(self.k):
            ax2.scatter(

                x=[data_article[j].x[i][0] for i in range(len(data_article[j]))],
                y=[data_article[j].x[i][1] for i in range(len(data_article[j]))],

                color=picked_colors[j], zorder=1

            )

            vals, vecs = np.linalg.eigh(self.variance_article[-1][j])
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]

            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            r2 = chi2.ppf(0.95, 1)

            width, height = 2 * np.sqrt(vals * r2)

            ell = mpl.patches.Ellipse(self.mean_article[-1][j], width, height, 180 + angle, color=picked_colors[j])

            ell.set_clip_box(fig.bbox)
            ell.set_alpha(0.35)
            ax2.add_artist(ell)
            ax2.scatter(
                x=[self.mean_article[-1][j][0]],
                y=[self.mean_article[-1][j][1]],

                color='black', marker="x", linewidths=3, s=100, zorder=3
            )
        plt.title(label='Article algorithm', fontsize=20)

        left, bottom, width, height = (self.s[0], self.s[1], self.t[0] - self.s[0], self.t[1] - self.s[1])
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  fill=False,
                                  color="black",
                                  linewidth=1)

        plt.gca().add_patch(rect)
        fig.patch.set_facecolor('white')
        ax0.set_facecolor('white')
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')
        plt.show()
        fig.figure.savefig("images/hist_real_data_" + str(self.k) + ".png",
                           width=3, height=1)

        titles = ['Thesis algorithm', 'Article algorithm']
        self.fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_titles=titles, x_title='Iteration',
                                 horizontal_spacing=0.03)
        self.fig.add_trace(go.Scatter(
            y=self.log_likelihood,
            x=np.array(range(1, len(self.log_likelihood) + 1)),
            line=dict(
                color='red',
                width=1,
            ),
            showlegend=False),
            row=1, col=1)
        self.fig.add_trace(go.Scatter(
            y=self.log_likelihood_article,
            x=np.array(range(1, len(self.log_likelihood_article) + 1)),
            line=dict(
                color='red',
                width=1,
            ),
            showlegend=False),
            row=1, col=2)
        self.fig.update_layout(title_text="Convergence of log-likelihood function", title_font=dict(size=22))
        self.fig.update_annotations(font_size=22)
        self.fig.show()
        #self.fig.write_image("images/ll_real_data_" + str(self.k) + ".png",
        #                   width=2, height=1)
