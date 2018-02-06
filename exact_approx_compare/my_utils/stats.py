import pandas as pd
import numpy as np
from enum import Enum
import statsmodels.formula.api as sm
from sklearn import preprocessing
from sklearn.utils import resample
from scipy.stats import norm
import time
import sys
from copy import deepcopy
import math

class Response(Enum):
	conflicts = 1,
	runtime = 2

class Exploratory(Enum):
	entropy = 1,
	solution_density = 2,
	#backbone = 3

def get_formulas(df):
	"""
		Iterate over dataframe and return formulas set
	"""
	formulas_set = set()
	for index, row in df.iterrows():
		formulas_set.add(row['formula'])

	return formulas_set


def get_experiment_solver(dfdict, dfname, exp, solver):
	df = dfdict[dfname]
	return df[(df['experiment'] == exp) & (df['solver'] == solver)]


def unpack_dict_to_df(sub_dict, formulas_set, solvers):
	columns = ['formula', 'entropy', 'solution_density', 'backbone', 
			   'runtime_a', 'runtime_b', 'conflicts_a', 'conflicts_b',
			   'diff_runtime', 'diff_conflicts']
	out_dict = {col: list() for col in columns} 
	
	for f in formulas_set:
		unpack = sub_dict[(f, solvers[0])]
		unpackb = sub_dict[(f, solvers[1])]
		entropy = unpack[0]
		solution_density = unpack[1]
		backbone = unpack[2]
		runtime_a = unpack[3]
		conflicts_a = unpack[4]
		runtime_b = unpackb[3]
		conflicts_b = unpackb[4]
		diff_runtime = runtime_a - runtime_b
		diff_conflicts = conflicts_a - conflicts_b
		out_dict['formula'].append(f)
		out_dict['entropy'].append(entropy)
		out_dict['solution_density'].append(solution_density)
		out_dict['backbone'].append(backbone)
		out_dict['runtime_a'].append(runtime_a)
		out_dict['runtime_b'].append(runtime_b)
		out_dict['conflicts_a'].append(conflicts_a)
		out_dict['conflicts_b'].append(conflicts_b)
		out_dict['diff_runtime'].append(diff_runtime)
		out_dict['diff_conflicts'].append(diff_conflicts)
	
	out_df = pd.DataFrame(out_dict, columns=columns)
	return out_df

def add_scaled_variables(out_df, scale_y='standard'):
	# add scaled columns for stats test
	scaler = preprocessing.StandardScaler()
	out_df['entropy_scaled'] = scaler.fit_transform(out_df['entropy'])
	out_df['solution_density_scaled'] = scaler.fit_transform(out_df['solution_density'])
	# runtime
	out_df['diff_runtime_scaled'] = scaler.fit_transform(out_df['diff_runtime'])
	out_df['runtime_a_scaled'] = scaler.fit_transform(out_df['runtime_a'])
	out_df['runtime_b_scaled'] = scaler.fit_transform(out_df['runtime_b'])
	# conflicts
	out_df['diff_conflicts_scaled'] = scaler.fit_transform(out_df['diff_conflicts'])
	out_df['conflicts_a_scaled'] = scaler.fit_transform(out_df['conflicts_a'])
	out_df['conflicts_b_scaled'] = scaler.fit_transform(out_df['conflicts_b'])
	return out_df

def sub_dataframe(dfdict, dfname, exp, solvers, add_scaled_vars=True):
	t = time.time()
	df = pd.DataFrame()

	formulas_set = set()
	sub_dict = {}
	
	# collect solvers data for this experiment
	for solver in solvers:
		df = get_experiment_solver(dfdict, dfname, exp, solver)
		# collect data from solver
		for index, row in df.iterrows():
			formula = row['formula']
			entropy = row['entropy']
			solution_density = row['solution_density']
			backbone = row['backbone']
			runtime = row['runtime']
			conflicts = row['conflicts']
			sub_dict[(formula,solver)] = (entropy, solution_density, backbone, runtime, conflicts)
			formulas_set.add(formula)

	out_df = unpack_dict_to_df(sub_dict, formulas_set, solvers)

	# add scaled variables
	if add_scaled_vars:
		out_df = add_scaled_variables(out_df)

	print 'Time elapsed: %lf' % (time.time() - t)

	return out_df

def perform_lm_bootstrap(df, X, y, samples):
	"""
		Perform bootstrap resampling 
		returns linear regression coefs dataframe (index is sample num)
	"""
	t = time.time()
	formula = str(y) + '~' + str(X)
	sampled_params_df = {'slope': list(), 'intercept': list()}
	for i in range(samples):
		df_resample = resample(df)
		res = sm.ols(formula = formula, data=df_resample).fit()
		slope = res.params.iloc[1]
		intercept = res.params.iloc[0]
		sampled_params_df['slope'].append(slope)
		sampled_params_df['intercept'].append(intercept)

	#print 'Performed bootstrap'
	#print 'Num samples: %d, Time elapsed: %lf' % (samples, time.time() - t)
	return pd.DataFrame(sampled_params_df)

def perform_lm_bootstrap_test(df, X, y, samples):
	"""
		Perform bootstrap resampling 
		returns linear regression coefs dataframe (index is sample num)
	"""
	t = time.time()
	formula = str(y) + '~' + str(X)
	sampled_params_df = {'slope': list(), 'intercept': list()}
	for i in range(samples):
		df_resample = resample(df)
		res = sm.ols(formula = formula, data=df_resample).fit()
		slope = res.params.iloc[1]
		intercept = res.params.iloc[0]
		#slope_std_err = res.bse.iloc[0]
		#intercept_std_err = res.bse.iloc[0]
		sampled_params_df['slope'].append(slope)
		sampled_params_df['intercept'].append(intercept)

	#print 'Performed bootstrap'
	#print 'Num samples: %d, Time elapsed: %lf' % (samples, time.time() - t)
	sampled_params_df = pd.DataFrame(sampled_params_df)
	print 
	return res.summary()

def perform_lm_bootstrap_generic(df, formulas, samples):
	"""
	IN:
		df: dataframe filtered by experiment-solver
		formulas: list of formulas e.g. ['colM ~ colN', 'colJ ~ colN' ]
		samples: # of boot strap samples

		Perform bootstrap resampling for delta beta test (2 linear models, difference between coefs)
		returns linear regression coefs dataframe (index is sample num)
	"""
	t = time.time()

	generic_result = {'slope': list(), 'intercept': list() }
	diff_dict = {'slope_diff': list(), 'intercept_diff': list()} # a-b

	res = {f: None for f in formulas}
	slope = {f: None for f in formulas}
	intercept = {f: None for f in formulas}
	res_dict = {f: dict(deepcopy(generic_result)) for f in formulas}

	for i in range(samples):
		df_resample = resample(df)

		for f in formulas:
			res[f] = sm.ols(formula = f, data=df_resample).fit()

			slope[f] = res[f].params.iloc[1]
			intercept[f] = res[f].params.iloc[0]

			res_dict[f]['slope'].append(slope[f])
			res_dict[f]['intercept'].append(intercept[f])

		# currently works for 2 formulas
		# append diffs
		diff_dict['slope_diff'].append(slope[formulas[0]] - slope[formulas[1]])
		diff_dict['intercept_diff'].append(intercept[formulas[0]] - intercept[formulas[1]])

	# concat all
	out_df = pd.DataFrame()
	a = pd.DataFrame(res_dict[formulas[0]])
	b = pd.DataFrame(res_dict[formulas[1]])

	# handle output
	for f in formulas:
		if f == formulas[0]:
			a.rename(columns={'slope': 'slope_a', 'intercept': 'intercept_a'}, inplace=True)
			out_df = pd.concat([out_df, a], axis=1)
		else:
			b.rename(columns={'slope': 'slope_b', 'intercept': 'intercept_b'}, inplace=True)
			out_df = pd.concat([out_df, b], axis=1)

	out_df = pd.concat([out_df, pd.DataFrame(diff_dict)], axis=1)
	#print 'Performed bootstrap'
	#print 'Num samples: %d, Time elapsed: %lf' % (samples, time.time() - t)
	
	return out_df	

def get_statistics_bootstrap(bootstrap_df):
	"""
		Unpack statistics
	"""
	# cols in delta beta test
	diff_cols = ['intercept_diff', 'slope_diff']

	stats = {}
	for col in bootstrap_df.columns:
		mean = bootstrap_df[col].mean() 
		std = bootstrap_df[col].std()
		n = bootstrap_df.shape[0]
		# 95% confidence interval
		#a_ci = mean - (1.96 * std / (n**0.5))
		b_ci = 2*mean - bootstrap_df[col].quantile(0.025)
		a_ci = 2*mean - bootstrap_df[col].quantile(0.975)
		#b_ci = mean + (1.96 * std / (n**0.5))
		ci = (round(a_ci, 4), round(b_ci, 4)) 
		# stats tests: intercept, slope != 0 
		if col not in diff_cols:
			# H0: param = 0
			Z = mean / std
			# two tailed test
			p_val = norm.sf(abs(Z))*2
		else:
			# H0: b1 - b2 = 0
			col_a = col.replace('_diff','_a')
			col_b = col.replace('_diff','_b')
			Z_denum = math.sqrt( (bootstrap_df[col_a].std())**2 + (bootstrap_df[col_b].std())**2 ) 
			Z = mean / Z_denum
			#Z = mean / std
			p_val = norm.sf(abs(Z))*2
		

		stats[col] = {'mean': round(mean,4), 'std': std, 'Z-score': round(Z, 4), 'ci': ci, 'p_val': round(p_val, 4)}

	return stats

def get_stats_delta_test(df, var, response, samples=1000):
	"""
		IN: df, exploratory var, response var, # bootstrap samples
		OUT: mean, confidence-interval, z-statistic, p_val, dataframe with sampled params
	"""
	t = time.time()

	# x,y
	X = str(var.name) + '_scaled' # only x should be scaled
	y = 'diff_' + str(response.name)

	# to dataframe
	df_sampled_params = perform_lm_bootstrap(df, X, y, samples)

	return get_statistics_bootstrap(df_sampled_params)


def delta_test(df, response, samples=1000):
	"""
		Perform bootstrap to find delta_test statistics
		User is responsible for correct dataframe (using sub_dataframe function)
	"""
	t = time.time()

	columns = {'exploratory-var', 'response', 'intercept', 
			   'intercept_confidence_interval', 'intercept_p_val', 
			   'slope', 'slope_confidence_interval', 'slope_p_val'}
	stats_dict = {i: list() for i in columns}

	for var in Exploratory:
		stats = get_stats_delta_test(df, var, response, samples)
		stats_dict['exploratory-var'].append(var.name)
		stats_dict['response'].append(response.name)
		stats_dict['intercept'].append(stats['intercept']['mean'])
		stats_dict['intercept_confidence_interval'].append(stats['intercept']['ci'])
		stats_dict['intercept_p_val'].append(stats['intercept']['p_val'])
		stats_dict['slope'].append(stats['slope']['mean'])
		stats_dict['slope_confidence_interval'].append(stats['slope']['ci'])
		stats_dict['slope_p_val'].append(stats['slope']['p_val'])


	out_df = pd.DataFrame(stats_dict, columns=columns)
	out_df.set_index(keys=['exploratory-var', 'response'], inplace=True)
	#out_df.drop(['exploratory-var', 'response'], axis=1, inplace=True)
	print 'Time elapsed: %lf' % (time.time() - t)
	return out_df

def get_stats_delta_beta_test(df, var, response, samples, scale_y=False):
	t = time.time()

	temp_df = pd.DataFrame()

	formulas = list()

	# x,y
	for solver in ['a', 'b']:
		X = str(var.name) + '_scaled'
		if scale_y:
			y = str(response.name) + '_' + solver + '_scaled'
		else:
			y = str(response.name) + '_' + solver
		formula = str(y) + '~' + str(X)
		formulas.append(formula)
		

	temp_df = perform_lm_bootstrap_generic(df, formulas, samples)

	return get_statistics_bootstrap(temp_df)
	#return temp_df

def delta_beta_test(df, response, samples=1000, scale_y=False):
	"""
		Perform bootstrap to find delta beta test statistics
		User is responsible for correct dataframe (using sub_dataframe function)
	"""
	t = time.time()

	columns = {'exploratory-var', 'response', 'test', 'intercept', 
			   'intercept_confidence_interval', 'intercept_p_val', 
			   'slope', 'slope_confidence_interval', 'slope_p_val'}
	stats_dict = {i: list() for i in columns}

	for var in Exploratory:
		stats = get_stats_delta_beta_test(df, var, response, samples, scale_y)
		for solver in ['_a', '_b', '_diff']:
			stats_dict['test'].append(solver.replace('_',''))
			stats_dict['exploratory-var'].append(var.name)
			stats_dict['response'].append(response.name)
			stats_dict['intercept'].append(stats['intercept' + solver]['mean'])
			stats_dict['intercept_confidence_interval'].append(stats['intercept' + solver]['ci'])
			stats_dict['intercept_p_val'].append(stats['intercept' + solver]['p_val'])
			stats_dict['slope'].append(stats['slope' + solver]['mean'])
			stats_dict['slope_confidence_interval'].append(stats['slope' + solver]['ci'])
			stats_dict['slope_p_val'].append(stats['slope' + solver]['p_val'])


	out_df = pd.DataFrame(stats_dict, columns=columns)
	out_df.set_index(keys=['test', 'exploratory-var', 'response'], inplace=True)
	#out_df.drop(['exploratory-var', 'response'], axis=1, inplace=True)
	print 'Time elapsed: %lf' % (time.time() - t)
	return out_df

def tests(dfdict, exp, a_val, b_val):
	"""
		Perform delta_test and delta_beta test

		OUT: dict 
	"""
	# dataframes
	exact, approx = None, None
	if exp != 'exp1':
		exact = create_secondary_dataframe(dfdict['exact'], exp, a_val, b_val)
	approx = create_secondary_dataframe(dfdict['exact'], exp, a_val, b_val)

	delta_test(exact, dfname)


def compare_delta(exact, approx, response_exact, response_approx, samples=1000):
	"""
		Compare two benchmark sets by appending their tests to one table
	"""
	a = delta_test(exact, response_exact, samples)
	b = delta_test(approx, response_approx, samples)
	a['benchmark'] = 'exact'
	b['benchmark'] = 'approx'

	out_df = pd.DataFrame()
	out_df = pd.concat([a.reset_index(),b.reset_index()], axis=0)
	out_df.set_index(keys=['benchmark', 'exploratory-var'], inplace=True)
	return out_df

def compare_delta_beta(exact, approx, response_exact, response_approx, samples=1000):
	"""
		Compare two benchmark sets by appending their tests to one table
	"""
	a = delta_beta_test(exact, response_exact, samples)
	b = delta_beta_test(approx, response_approx, samples)
	a['benchmark'] = 'exact'
	b['benchmark'] = 'approx'

	out_df = pd.DataFrame()
	out_df = pd.concat([a.reset_index(),b.reset_index()], axis=0)
	out_df.set_index(keys=['benchmark', 'exploratory-var'], inplace=True)
	return out_df

def check_directly(df, response, exploratory, scale_y=False):
	"""
		check directly p-values for slope difference
	"""
	if not scale_y:
		y_a = str(response.name) + '_a'
		y_b = str(response.name) + '_b'
	else:
		y_a = str(response.name) + '_a_scaled'
		y_b = str(response.name) + '_b_scaled'
	X = str(exploratory.name) + '_scaled'

	a_conf = df[['formula', y_a, X]]
	a_conf['condition'] = 'a'
	a_conf.rename(columns = {y_a: 'conflicts'}, inplace = True)
	b_conf = df[['formula', y_b, X]]
	b_conf['condition'] = 'b'
	b_conf.rename(columns = {y_b: 'conflicts'}, inplace = True)

	newdf = pd.DataFrame()
	newdf = pd.concat([a_conf, b_conf], axis=0)
	newdf

	formula = "conflicts ~ " + str(X) + "*C(condition)"
	res = sm.ols(formula=formula, data=newdf).fit()
	return res.summary()
"""
# check directly p-values for slope difference
a_conf = exact[['formula','conflicts_a','solution_density_scaled']]
a_conf['condition'] = 'a'
a_conf.rename(columns = {'conflicts_a': 'conflicts'}, inplace = True)
b_conf = exact[['formula','conflicts_b','solution_density_scaled']]
b_conf['condition'] = 'b'
b_conf.rename(columns = {'conflicts_b': 'conflicts'}, inplace = True)

newdf = pd.DataFrame()
newdf = pd.concat([a_conf, b_conf], axis=0)
newdf

import statsmodels.formula.api as sm
formula = "conflicts ~ solution_density_scaled*C(condition)"
res = sm.ols(formula=formula, data=newdf).fit()
res.summary()
"""