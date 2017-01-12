import MediFor_bn_model_generator_13 as reasoning_system

rows = 4
cols = 4
training_data_file = 'NIST_training_data_v17.csv'
regional_manipulations = ['removal', 'splice', 'copyclone']
global_manipulations = ['lighting']
schema_dict = {'removal': 2, 'splice': 2, 'copyclone': 2, 'lighting': 2, 'se-removal': 2, 'se-splice': 2, 'se-copyclone': 2, 'se-lighting': 2, 're': 2}
algorithms = ['abb01', 'block01', 'block02', 'cfa01', 'cfa02', 'combo01', 'copymove01', 'dct01', 'dct02', 'dct03_A', 'dct03_NA', 'ela01', 'noise01', 'noise02']

#Create base Bayesian Network model.
reasoning_system.train_model(training_data_file, algorithms, regional_manipulations, global_manipulations, schema_dict, rows, cols)
