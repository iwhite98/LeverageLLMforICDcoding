import pandas as pd
class Vocab:
	def __init__(self, data_path):
		total_code = set()
		df_train = pd.read_csv(data_path + '/train.csv')
		df_valid = pd.read_csv(data_path + '/valid.csv')
		df_test = pd.read_csv(data_path + '/test.csv')
		
		self.total_output = df_train['output'].tolist() + df_valid['output'].tolist() + df_test['output'].tolist()

		self.label2index, self.index2label = self.make_label_dict()
		self.n_label = len(self.label2index.keys())
	
	def make_label_dict(self):
		codes = set()
		for output in self.total_output:
			for code in output.split(','):
				codes.add(code)
		label2index = dict()
		index2label = dict()
		for i, code in enumerate(list(codes)):
			label2index[code] = i
			index2label[i] = code

		return label2index, index2label
