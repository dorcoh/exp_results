import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class Response(Enum):
	conflicts = 1,
	runtime = 2

class Exploratory(Enum):
	entropy = 1,
	solution_density = 2,
	#backbone = 3

def compare_histograms(dfdict):
	# choose some experiments and solver to extract all entropies,sols
	fig, axes = plt.subplots(2, 2)
	fig.suptitle('Occurences compare')
	i,j = 0, 0
	for name, df in dfdict.iteritems():
		df = df[(df['experiment'] == 'exp1') & (df['solver'] == 'glucose')]
		title = '%s benchmark sets' % str(name)
		df['entropy'].hist(ax=axes[i][j], bins=100)
		axes[i][j].set_title('Entropies -' + title)
		axes[i][j].set_xlabel('Entropy')
		axes[i][j].set_ylabel('Occurences')
		#tempdf_e['num_sols'].apply(lambda x: math.log(x,2) ).hist(ax=ax2, bins=10)
		df['solution_density'].hist(ax=axes[i][j+1], bins=100)
		axes[i][j+1].set_title('Solution densities - ' + title)
		axes[i][j+1].set_xlabel('Solution density')
		i = i+1

	plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
	plt.show()

def delta_beta_single_benchmark(df, response):
	"""
		In: df - sub data frame
			response - response variable (conflicts/runtime)
		Out:
			plot
	"""
	fig, axes = plt.subplots(1, 2)
	fig.suptitle('Delta beta test')

	title = '%s benchmark sets' % str(name)
	df['entropy'].hist(ax=axes[i][j], bins=100)
	axes[0][0].set_title('Entropies -' + title)
	axes[0][0].set_xlabel('Entropy')
	axes[0][0].set_ylabel('Occurences')
	#tempdf_e['num_sols'].apply(lambda x: math.log(x,2) ).hist(ax=ax2, bins=10)
	df['solution_density'].hist(ax=axes[i][j+1], bins=100)
	axes[0][1].set_title('Solution densities - ' + title)
	axes[0][1].set_xlabel('Solution density')
	i = i+1

	plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
	plt.show()

def plotter_both_runtimes(df, response, solvers, x_lim=None, y_lim=None):
	"""
		In: df - sub data frame
			response - response variable (conflicts/runtime)
			solvers - list of solvers to compare
		Out:
			plot
	"""
	
	fig, axes = plt.subplots(1, 2)
	fig.suptitle('Delta beta test')

	y_a_col = response.name + '_a'
	y_b_col = response.name + '_b'

	y_a = df[y_a_col]
	y_b = df[y_b_col]
	
	i = 0
	for var in Exploratory:
		# scatter
		x = df[var.name]

		axes[i].plot(x , y_a, 'ro')
		m, b = np.polyfit(x, y_a, 1)
		axes[i].plot(x, m*x + b, '-', c='r')

		axes[i].plot(x, y_b, 'bx')
		m, b = np.polyfit(x, y_b, 1)
		axes[i].plot(x, m*x + b, '-', c='b')

		#axes[i].set_xlim((-0.1, 0.8))
		#axes[i].set_ylim((-1000, max(y_a.max(), y_b.max())+1000))
		axes[i].autoscale_view()

		axes[i].set_xlabel(str(var.name))
		axes[i].set_ylabel('Runtime (seconds)')
		title = 'Runtime: ' + str(solvers[0]) + '-' + str(solvers[1])
		axes[i].set_title(title)
		axes[i].legend([solvers[0], solvers[0], solvers[1], solvers[1]])
		i +=1

	plt.show()

def aggregate(x, y, i):
	if i==1:
		return x, y
	temp = pd.DataFrame()
	temp['x'] = x
	temp['y'] = y
	agg = temp.groupby(by='x').mean()
	agg = agg.reset_index()
	return agg['x'], agg['y']


def plotter_compare(dfa, dfb, response_type_a, response_type_b, solvers, x_lim=None, y_lim=None):

	fig, axes = plt.subplots(2, 2)
	fig.suptitle('Delta beta test')

	# assume dfa is exact (need to round and aggregate)
	# round to to 2 decimal points

	for var in Exploratory:
		col = var.name
		dfa[col] = dfa[col].apply(lambda x: round(x, 2))

	
	for i, df in enumerate([dfa, dfb]):
		if i == 0:
			response = response_type_a
		else:
			response = response_type_b
		for j,var in enumerate(Exploratory):
			# scatter

			x = df[var.name]
			
			y_a_col = response.name + '_a'
			y_b_col = response.name + '_b'

			y_a = df[y_a_col]
			y_b = df[y_b_col]

			x_agg, y_a_agg = aggregate(x, y_a, i)
			axes[i][j].plot(x_agg, y_a_agg, 'ro')

			m, b = np.polyfit(x, y_a, 1)
			axes[i][j].plot(x, m*x + b, '-', c='r')

			x_agg, y_b_agg = aggregate(x, y_b, i)
			axes[i][j].plot(x_agg, y_b_agg, 'bx')
			m, b = np.polyfit(x, y_b, 1)
			axes[i][j].plot(x, m*x + b, '-', c='b')

			#axes[i].set_xlim((-0.1, 0.8))
			#axes[i].set_ylim((-1000, max(y_a.max(), y_b.max())+1000))
			axes[i][j].autoscale_view()

			axes[i][j].set_xlabel(str(var.name))


			axes[i][j].set_ylabel(str(response.name))
			title = str(response.name) + ' ' + str(solvers[0]) + '-' + str(solvers[1])
			axes[i][j].set_title(title)
			axes[i][j].legend([solvers[0], solvers[0], solvers[1], solvers[1]])

	plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
	plt.show()