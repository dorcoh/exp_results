import pandas as pd
import math

def find_timeout_formulas(df):
	"""
		Find timeout formulas 
		(only for the approx dataframe)
		IN: pd DataFrame
		OUT: 
			tuple:
			(python set with formulas, 
		 	 python set with indices)
	"""
	# find timeout formulas
	terminated_formulas = set()
	terminated_indices = list() 
	for index, row in df[df['variables'] <= 0].iterrows():
	    terminated_formulas.add(row['formula'])
	    terminated_indices.append(index)
	return terminated_formulas, terminated_indices

def show_timeout_formulas(df):
	"""
		Shows timeout formulas (all their engines) in approx df
	"""
	terminated_formulas, indices = find_timeout_formulas(df)
	return df[df['formula'].isin(terminated_formulas)].sort_values(by=['formula','experiment','solver'])

def get_stats_df(df):
	"""
	#TODO
	"""
	variables = df.groupby('formula')['variables'].mean()
	entropy = df.groupby('formula')['entropy'].mean()
	num_sols = df.groupby('formula')['num_sols'].mean()
	means = df.groupby('formula')['runtime'].mean()
	medians = df.groupby('formula')['runtime'].median()
	stds = df.groupby('formula')['runtime'].std()
	series_list = [ entropy,num_sols, variables, means, medians, stds]
	statsdf = pd.concat(series_list, axis=1)
	statsdf.columns=['entropy', 'num_sols', 'variables','runtime-mean', 'median', 'std']
	return statsdf

def remove_formulas(df):
	"""
		Approx dataframe only
		Remove formulas with avg time < 15 (on our engines)
	"""
	statsdf = get_stats_df(df)
	formulas_to_remove = list(statsdf[(statsdf['runtime-mean'] < 15)].index)
	#formulas_to_remove += list(terminated_formulas)
	print('formulas_to_remove', len(formulas_to_remove))
	print('expected #rows', len(formulas_to_remove)*21)
	indices = list()
	for index, row in df.iterrows():
	    if row['formula'] in formulas_to_remove:
	        indices.append(index)
	print('actual #rows', len(indices))
	print('before ', df.shape)
	df.drop(indices, inplace=True)
	print('after ', df.shape)

def add_density(df):
	"""
		add solution density to df
	"""
	df['solution_density'] = df.apply(lambda x: (math.log(x['num_sols'], 2) / x['variables']) if x['variables'] > 0 else None , axis=1)

