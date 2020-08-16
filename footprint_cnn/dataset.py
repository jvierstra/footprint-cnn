from kipoi.data import Dataset
from kipoiseq.transforms import ReorderedOneHot

from genome_tools import genomic_interval, genomic_interval_set, bed3_iterator
from genome_tools.helpers import open_file
from pyfaidx import Fasta

from footprint_tools import bamfile
from footprint_tools.modeling import bias, prediction

class DataLoader(Dataset):
	def __init__(self, intervals_file, fasta_file, bam_file, bias_model_file, shuffle=True):

		self.intervals_file = intervals_file
		self.fasta_file = fasta_file
		self.bam_file = bam_file
		self.bias_model_file = bias_model_file

		intervals_filehandle=open_file(intervals_file)
		self.intervals=genomic_interval_set(bed3_iterator(intervals_filehandle))
		
		self.seq_transform = ReorderedOneHot(alphabet="ACGT")

		self.fasta_extractor=None
		self.bm=None
		self.cutcounts=None


	def __len__(self):
		return len(self.intervals)

	def __getitem__(self, idx):

		interval=self.intervals[idx]

		if self.fasta is None:
			self.fasta_extractor = Fasta(self.fasta_file)
			self.cutcounts = bamfile(self.bam_file)
			self.bm=bias.kmer_model(self.bias_model_file)


		# Set up the footprint-tools class to predict cleavages
		pred=prediction(self.cutcounts, 
									self.fasta_extractor, 
									interval, 
									self.bm,
									half_window_width = 5, 
									smoothing_half_window_width = 50, 
									smoothing_clip = 0.01)


		# one hot encode DNA
		one_hot_seq = self.seq_transform(pred.seq)

		# compute the observed expected DNase I data
		obs, exp, win = pred.compute()

		inputs=[one_hot_seq, np.vstack([obs['+'][1:], obs['-'][:-1], exp['+'][1:], exp['-'][:-1]])]

		ret = {"inputs": inputs,
			   "targets": outputs}

		return ret
