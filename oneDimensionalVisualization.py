import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
from matplotlib import pyplot as plt

from plotly.subplots import make_subplots
from scipy.stats import norm

from truncatedGaussianMixture import TruncatedGaussianMixture


class SummaryPlot1D:

    def __init__(self, results):
        self.s, self.t = results['real_params']['s'], results['real_params']['t']
        self.k = results['real_params']['k']
        self.pweight_true = results['real_params']['pweight']
        self.n = len(results['correct_EM'])-1
        self.nalg = len(results['article_EM'])-1
        self.n_points = np.sum([len(results['data'][k]) for k in range(self.k)])
        self.time = results['time']
        self.time_article = results['time_article']
        self.order_correct = {}
        self.order_article = {}
        nodes_correct = results['correct_EM'][len(results['correct_EM']) - 1]['mean']
        nodes_article = results['article_EM'][len(results['article_EM']) - 1]['mean']
        for k in range(self.k):
            self.order_correct[k] = np.argmin((nodes_correct - results['real_params']['mean'][k]) ** 2)
            self.order_article[k] = np.argmin((nodes_article - results['real_params']['mean'][k]) ** 2)

        self.mean_true, self.variance_true, self.weights_true = results['real_params']['mean'], results['real_params'][
            'variance'], results['real_params']['weight']
        self.mean_em, self.variance_em, self.weights_em = [results['correct_EM'][i]['mean'] for i in range(self.n+1)], [
            results['correct_EM'][i]['variance'] for i in range(self.n+1)], [results['correct_EM'][i]['weight'] for i in
                                                                        range(self.n+1)]
        self.mean_article, self.variance_article, self.weights_article = [results['article_EM'][i]['mean'] for i in
                                                                     range(self.nalg+1)], [
                                                                        results['article_EM'][i]['variance'] for i in
                                                                        range(self.nalg+1)], [
                                                                        results['article_EM'][i]['weight'] for i in
                                                                        range(self.nalg+1)]
        self.log_likelihood = [results['correct_EM'][i]['log-likelihood'] for i in range(len(results['correct_EM']))]
        self.log_likelihood_article = [results['article_EM'][i]['log-likelihood'] for i in
                                       range(len(results['article_EM']))]

        self.data = results['data']



    def histogram_plot(self, bin_size=1, xaxis_range=None):

        picked_colors = [
            ["rgba(220,179,15,0.85)", "rgba(27, 163, 156,0.85)","rgba(192,57,43,0.85)","rgba(187,187,187,0.85)","rgba(27,163,156,0.85)"]
            [k] for k in range(self.k)
        ]
        hist_data = [[self.data[self.k-j-1][i][0] for i in range(len(self.data[self.k-j-1]))] for j in range(self.k)]
        pom_plot = [

            np.max( plt.hist(
                hist_data[k],
                density=True,
                bins=np.arange(
                    min(hist_data[k]),
                    max(hist_data[k]) + 1,
                    bin_size
                )
            )[0])
            for k in range(self.k)
        ]
        dmax = np.round(np.max(pom_plot),2)
        plt.clf()
        grouplabels = ['Cluster ' + str(self.k-k) for k in range(self.k)]
        fig = ff.create_distplot(hist_data, group_labels=grouplabels, bin_size=bin_size, show_curve=False, show_rug=False,
                                 colors=picked_colors)
        #print(fig)

        for j in range(self.k):
            fig.add_trace(go.Scatter(x=[self.mean_true[j], self.mean_true[j]],
                                     y=[0, dmax],
                                     mode='lines',
                                     line=dict(color='green', width=1.5),
                                     name='True mean',
                                     legendgroup='group1',
                                     showlegend=j == 0))

            fig.add_trace(go.Scatter(x=[self.mean_em[-1][self.order_correct[j]], self.mean_em[-1][self.order_correct[j]]],
                                     y=[0, dmax],
                                     mode='lines',
                                     line=dict(color='#000075', width=1.5, dash='dot'),
                                     name='Thesis algorithm mean',
                                     legendgroup='group2',
                                     showlegend=j == 0))

            fig.add_trace(
                go.Scatter(x=[self.mean_article[-1][self.order_article[j]], self.mean_article[-1][self.order_article[j]]],
                           y=[0, dmax],
                           mode='lines',
                           line=dict(color='#d16f6f', width=1.5, dash='dash'),
                           name='Article algorithm mean',
                           legendgroup='group3',
                           showlegend=j == 0))

        fig.update_layout(title_text="Histogram of data with estimated means", showlegend=True,
                          title_font=dict(size=22),plot_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(legend=dict(
            font=dict(size=18)
        ))
        fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', ticks="outside")
        fig.update_yaxes(showline=True, linewidth=1, linecolor='gray')
        fig.update_layout(yaxis_range=[0,dmax])
        '''
        fig.update_layout(xaxis_range=[-0.1,1.1])
        '''
        if xaxis_range is None:
            fig.update_layout(xaxis_range=[
                min(0,min(
                    [
                        np.min(self.mean_true)-2,
                        np.min([self.mean_em[-1][self.order_correct[j]] for j in range(self.k)])-2,
                        np.min([self.mean_em[-1][self.order_correct[j]] for j in range(self.k)])-2])),
                    40])
        else:
            fig.update_layout(xaxis_range=xaxis_range)

        fig.show()
        fig.write_image("images/hist_1d"+str(self.k)+"c_"+"_".join([str(self.mean_true[k]) for k in range(self.k)])+"_"+str(self.n_points)+".png",
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
            x=np.array(range(0, len(self.log_likelihood_article) )),
            line=dict(
                color='red',
                width=1,
            ),
            showlegend=False),
            row=1, col=2)
        self.fig.update_layout(title_text="Convergence of log-likelihood function", title_font=dict(size=22))
        self.fig.update_annotations(font_size=22)
        self.fig.show()
        self.fig.write_image("images/ll_1d"+str(self.k)+"c_"+"_".join([str(self.mean_true[k]) for k in range(self.k)])+"_"+str(self.n_points)+".png",
                             width=2, height=1)



    def parameters_plot(self):
        titles = ['Thesis algorithm', 'Article algorithm']
        self.fig_2 = make_subplots(rows=self.k, cols=2, shared_yaxes=True, x_title='Iteration',
                                   column_titles=titles, horizontal_spacing=0.03)
        print(self.mean_em)
        for k in range(self.k):
            print(len(self.mean_em))
            y=[np.abs(self.mean_em[i][self.order_correct[k]] - self.mean_true[k]) ** 2 for i in
               range(len(self.mean_em))]
            x=np.array(range(0, len(self.mean_em)+1 ))
            print(x)
            print(y)
            self.fig_2.add_trace(
                go.Scatter(
                    y=[np.abs(self.mean_em[i][self.order_correct[k]] - self.mean_true[k]) ** 2 for i in
                       range(len(self.mean_em))],
                    x=np.array(range(0, len(self.mean_em)+1 )),
                    line=dict(
                        color='black',
                        width=1
                    ),
                    showlegend=False),
                row=k + 1, col=1
            )

            self.fig_2.add_trace(
                go.Scatter(
                    y=[np.abs(self.mean_article[i][self.order_article[k]] - self.mean_true[k]) ** 2 for i in
                       range(len(self.mean_article))],
                    x=np.array(range(0, len(self.mean_article) + 1)),
                    line=dict(
                        color='black',
                        width=1
                    ),
                    showlegend=False),
                row=k + 1, col=2
            )
        self.fig_2.update_layout(title_text="Convergence of cluster means", title_font=dict(size=22))
        self.fig_2.update_annotations(font_size=22)
        self.fig_2.show()
        self.fig_3 = make_subplots(rows=self.k, cols=2, shared_yaxes=True, x_title='Iteration', column_titles=titles,
                                   horizontal_spacing=0.03)
        for k in range(self.k):
            self.fig_3.add_trace(
                go.Scatter(
                    y=[np.abs(self.variance_em[i][self.order_correct[k]] - self.variance_true[k]) ** 2 for i in
                       range(len(self.variance_em))],
                    x=np.array(range(0, len(self.variance_em)+1)),
                    line=dict(
                        color='black',
                        width=1
                    ),
                    showlegend=False),
                row=k + 1, col=1
            )

            self.fig_3.add_trace(
                go.Scatter(
                    y=[np.abs(self.variance_article[i][self.order_article[k]] - self.variance_true[k]) ** 2 for i in
                       range(len(self.variance_article))],
                    x=np.array(range(0, len(self.variance_article) +1)),
                    line=dict(
                        color='black',
                        width=1
                    ),
                    showlegend=False),
                row=k + 1, col=2
            )
        self.fig_3.update_layout(title_text="Convergence of cluster variance", title_font=dict(size=22))
        self.fig_3.update_annotations(font_size=22)

        self.fig_3.show()

        self.fig_4 = make_subplots(rows=self.k, cols=2, shared_yaxes=True, x_title='Iteration', column_titles=titles,
                                   horizontal_spacing=0.03)
        for k in range(self.k):
            self.fig_4.add_trace(
                go.Scatter(
                    y=[np.abs(self.weights_em[i].flatten()[self.order_correct[k]] - self.weights_true[k]) ** 2 for i in
                       range(len(self.weights_em))],
                    x=np.array(range(0, len(self.weights_em) +1)),
                    line=dict(
                        color='black',
                        width=1
                    ),
                    showlegend=False),
                row=k + 1, col=1
            )

            self.fig_4.add_trace(
                go.Scatter(
                    y=[np.abs(self.weights_article[i].flatten()[self.order_article[k]] - self.weights_true[k]) ** 2 for
                       i in range(len(self.weights_article))],
                    x=np.array(range(0, len(self.weights_article)+1 )),
                    line=dict(
                        color='black',
                        width=1
                    ),
                    showlegend=False),
                row=k + 1, col=2
            )
        self.fig_4.update_layout(title_text="Convergence of cluster weights", title_font=dict(size=22))
        self.fig_4.update_annotations(font_size=22)

        self.fig_4.show()


    def pdf_1d(self, x, weight, mu, sigma):
        pom=[]
        for k in range(self.k):
            pom.append(weight[k] * norm.pdf(x, mu[k], sigma[k]**0.5)/(norm.cdf(self.t, mu[k],sigma[k]**0.5)-norm.cdf(self.s, mu[k],sigma[k]**0.5)))
        return np.sum(pom)


    def kl_divergence_1d(self, data, weight, mu, sigma):


        kl = (1/2000)*np.sum(
            [
                np.sum([np.log(self.pdf_1d(x_i, self.weights_true, self.mean_true, self.variance_true))-np.log(self.pdf_1d(x_i, weight, mu, sigma)) for x_i in data[j]
                 ]) for j in range(self.k)
            ])
        return kl

    def output_table(self):
        pokus = TruncatedGaussianMixture(
            self.mean_true, self.variance_true, self.s, self.t, self.pweight_true, self.k, 2000, 1, seed=23)
        data, weights = pokus.simulate()
        KL_divergence_em = self.kl_divergence_1d(data, self.weights_em[-1], self.mean_em[-1], self.variance_em[-1])
        KL_divergence_article = self.kl_divergence_1d(data, self.weights_article[-1], self.mean_article[-1], self.variance_article[-1])
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
        print(f'Number of data points:         {str(self.n_points)}')
        print(f'Real parameters:')
        for k in range(self.k):
            print(f'    mean {str(k+1)}:                    {self.mean_true[k]}')
            print(f'    variance {str(k+1)}:                {self.variance_true[k]}')
            print(f'    weight {str(k+1)}:                  {self.weights_true[k]}')
        print('===============================================================')
        print('Thesis algorithm')
        print()
        print(f'Thesis estimated parameters:')
        for k in range(self.k):
            print(f'    mean {str(k+1)}:                    {self.mean_em[-1][self.order_correct[k]]}')
            print(f'    variance {str(k+1)}:                {self.variance_em[-1][self.order_correct[k]]}')
            print(f'    weight {str(k+1)}:                {self.weights_em[-1][self.order_correct[k]]}')
        print(f'Number of steps:               {self.n}')
        print(f'Log-likelihood:                {self.log_likelihood[-1]}')
        print(f'Calculation time:              {self.time}')
        print(f'KL score:                      {KL_divergence_em}')
        print('===============================================================')
        print('Article algorithm')
        print()
        print(f'Article estimated parameters:')
        for k in range(self.k):
            print(f'    mean {str(k+1)}:                    {self.mean_article[-1][self.order_correct[k]]}')
            print(f'    variance {str(k+1)}:                {self.variance_article[-1][self.order_correct[k]]}')
            print(f'    weight {str(k+1)}:                {self.weights_article[-1][self.order_correct[k]]}')
        print(f'Number of steps:               {self.nalg}')
        print(f'Log-likelihood:                {self.log_likelihood_article[-1]}')
        print(f'Calculation time:              {self.time_article}')
        print(f'KL score:                      {KL_divergence_article}')
        print('===============================================================')
