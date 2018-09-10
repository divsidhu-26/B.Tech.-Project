import tensorflow as tf 
import time 
import argparse
import random
import numpy as np 
import os.path
import math
import timeit
from multiprocessing import JoinableQueue, Queue, Process
from collections import defaultdict

class TransE:
	@property
	def variables(self):
		return self.__variables

	@property
	def num_triple_train(self):
		return self.__num_triple_train

	@property 
	def num_triple_test(self):
		return self.__num_triple_test

	@property
	def testing_data(self):
		return self.__triple_test

	@property 
	def num_entity(self):
		return self.__num_entity

	@property
	def embedding_entity(self):
		return self.__embedding_entity


	@property
	def embedding_relation(self):
		return self.__embedding_relation

	@property
	def hr_t(self):
		return self.__hr_t

	@property 
	def tr_h(self):
		return self.__tr_h


	def training_data_batch(self, batch_size = 5120):
		n_triple = len(self.__triple_train)
		rand_idx = np.random.permutation(n_triple)
		start = 0
		while start < n_triple:
			start_t = timeit.default_timer()
			end = min(start+batch_size, n_triple)
			size = end - start 
			train_triple_positive = np.asarray([ self.__triple_train[x] for x in  rand_idx[start:end]])
			train_triple_negative = []
			for t in train_triple_positive:
				replace_entity_id = np.random.randint(self.__num_entity)
				random_num = np.random.random()

				if self.__negative_sampling == 'unif':
					replace_head_probability = 0.5
				elif self.__negative_sampling == 'bern':
					replace_head_probability = self.__relation_property[t[1]]
				else:
					raise NotImplementedError("Dose not support %s negative_sampling" %negative_sampling)

				if random_num<replace_head_probability:
					train_triple_negative.append((replace_entity_id, t[1],t[2]))
				else:
					train_triple_negative.append((t[0], t[1], replace_entity_id))
				
			start = end
			prepare_t = timeit.default_timer()-start_t

			yield train_triple_positive, train_triple_negative, prepare_t


	def __init__(self, data_dir, negative_sampling,learning_rate, 
				 batch_size, max_iter, margin, dimension, norm, evaluation_size, regularizer_weight):
		# this part for data prepare
		self.__data_dir=data_dir
		self.__negative_sampling=negative_sampling
		self.__regularizer_weight = regularizer_weight
		self.__norm = norm

		self.__entity2id={}
		self.__id2entity={}
		# self.__e2e = {}
		self.__relation2id={}
		self.__id2relation={}

		self.__triple_train=[] #[(head_id, relation_id, tail_id),...]
		self.__triple_test=[]
		self.__triple_valid=[]
		self.__triple = []

		self.__num_entity=0
		self.__num_relation=0
		self.__num_triple_train=0
		self.__num_triple_test=0
		self.__num_triple_valid=0

		# load all the file: entity2id.txt, relation2id.txt, train.txt, test.txt, valid.txt
		self.load_data()
		print('finish preparing data. ')


		# this part for the model:
		self.__learning_rate = learning_rate
		self.__batch_size = batch_size
		self.__max_iter = max_iter
		self.__margin = margin
		self.__dimension = dimension
		self.__variables= []
		#self.__norm = norm
		self.__evaluation_size = evaluation_size
		bound = 6 / math.sqrt(self.__dimension)
		with tf.device('/cpu'):
			self.__embedding_entity = tf.get_variable('embedding_entity', [self.__num_entity, self.__dimension],
													   initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound, seed = 123))
			self.__embedding_relation = tf.get_variable('embedding_relation', [self.__num_relation, self.__dimension],
														 initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound, seed =124))
			self.__normal_vectors = tf.get_variable('normal_vectors', [self.__num_relation,self.__dimension],
														initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound, seed=125))
			self.__variables.append(self.__embedding_entity)
			self.__variables.append(self.__embedding_relation)
			self.__variables.append(self.__normal_vectors)
			print('finishing initializing')
		


	def load_data(self):
		print('loading entity2id.txt ...')
		with open(os.path.join(self.__data_dir, 'entity2id.txt')) as f:
			self.__entity2id = {line.split(" ")[0]: int(line.split(' ')[1]) for line in f.readlines()}				# check if line.strip() is needed here - removed need in data processing
			# self.__e2e = {line.strip().split('}{|')[0].split("-;-")[0]: line.strip().split('}{|')[0] for line in f.readlines()}
			self.__id2entity = {value:key for key,value in self.__entity2id.items()}


		with open(os.path.join(self.__data_dir,'relation2id.txt')) as f:
			self.__relation2id = {line.split(' ')[0]: int(line.split(' ')[1]) for line in f.readlines()}
			self.__id2relation = {value:key for key, value in self.__relation2id.items()}

		def load_triple(self, triplefile):
			triple_list = [] #[(head_id, relation_id, tail_id),...]
			with open(os.path.join(self.__data_dir, triplefile)) as f:
				for line in f.readlines():
					line_list = line.split(' ')
					assert len(line_list) == 3
					# print(line_list)
					headid,relationid,tailid = 0,0,0
					try:
						headid = int(line_list[0])
						relationid = int(line_list[1])
						tailid = int(line_list[2])
					except:
						continue
					triple_list.append((headid, relationid, tailid))
					self.__hr_t[(headid, relationid)].add(tailid)
					self.__tr_h[(tailid, relationid)].add(headid)
			return triple_list

		self.__hr_t = defaultdict(set)
		self.__tr_h = defaultdict(set)
		self.__triple_train = load_triple(self, 'train.txt')
		self.__triple_test = load_triple(self, 'test.txt')
		self.__triple_valid = load_triple(self, 'valid.txt')
		self.__triple = np.concatenate([self.__triple_train, self.__triple_test, self.__triple_valid], axis = 0 )

		self.__num_relation = len(self.__relation2id)
		self.__num_entity = len(self.__entity2id)
		self.__num_triple_train = len(self.__triple_train)
		self.__num_triple_test = len(self.__triple_test)
		self.__num_triple_valid = len(self.__triple_valid)

		print('entity number: ' + str(self.__num_entity))
		print('relation number: ' + str(self.__num_relation))
		print('training triple number: ' + str(self.__num_triple_train))
		print('testing triple number: ' + str(self.__num_triple_test))
		print('valid triple number: ' + str(self.__num_triple_valid))


		if self.__negative_sampling == 'bern':
			self.__relation_property_head = {x:[] for x in range(self.__num_relation)} #{relation_id:[headid1, headid2,...]}
			self.__relation_property_tail = {x:[] for x in range(self.__num_relation)} #{relation_id:[tailid1, tailid2,...]}
			for t in self.__triple_train:
				#print(t)
				self.__relation_property_head[t[1]].append(t[0])
				self.__relation_property_tail[t[1]].append(t[2])
			self.__relation_property = {x:(len(set(self.__relation_property_tail[x])))/(len(set(self.__relation_property_head[x]))+ len(set(self.__relation_property_tail[x]))) \
										 for x in self.__relation_property_head.keys()} # {relation_id: p, ...} 0< num <1, and for relation replace head entity with the property p
		else: 
			print("unif set do'n need to calculate hpt and tph")

	

	def train(self, inputs):
		embedding_relation = self.__embedding_relation
		embedding_entity = self.__embedding_entity
		normal_vector = self.__normal_vectors

		triple_positive, triple_negative = inputs # triple_positive:(head_id,relation_id,tail_id)

		norm_entity = tf.nn.l2_normalize(embedding_entity, dim = 1)
		norm_relation = tf.nn.l2_normalize(embedding_relation, dim = 1)
		norm_entity_l2sum = tf.sqrt(tf.reduce_sum(norm_entity**2, axis = 1))

		embedding_positive_head = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 0])
		embedding_positive_tail = tf.nn.embedding_lookup(norm_entity, triple_positive[:, 2])
		embedding_positive_relation = tf.nn.embedding_lookup(norm_relation, triple_positive[:, 1])

		embedding_negative_head = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 0])
		embedding_negative_tail = tf.nn.embedding_lookup(norm_entity, triple_negative[:, 2])
		embedding_negative_relation = tf.nn.embedding_lookup(norm_relation, triple_negative[:, 1])

		pos_norm = tf.nn.embedding_lookup(normal_vector,triple_positive[:, 1])
		neg_norm = tf.nn.embedding_lookup(normal_vector,triple_negative[:, 1])

		def transfer( e, n):
			norm = n
			return e - tf.reduce_sum(e * norm, 1, keep_dims = True) * norm

		e_p_h = transfer(embedding_positive_head,pos_norm)
		e_p_t = transfer(embedding_positive_tail,pos_norm)
		e_n_h = transfer(embedding_negative_head,neg_norm)
		e_n_t = transfer(embedding_negative_tail,neg_norm)
		

		score_positive = tf.reduce_sum(tf.abs(e_p_h + embedding_positive_relation - e_p_t), axis = 1)
		score_negative = tf.reduce_sum(tf.abs(e_n_h + embedding_negative_relation - e_n_t), axis = 1)

		loss_every = tf.maximum(0., score_positive + self.__margin - score_negative)
		loss_triple = tf.reduce_sum(tf.maximum(0., score_positive + self.__margin - score_negative))
		self.__loss_regularizer = loss_regularizer = tf.reduce_sum(tf.abs(self.__embedding_relation)) + tf.reduce_sum(tf.abs(self.__embedding_entity)) + tf.reduce_sum(tf.abs(self.__normal_vectors))
		return loss_triple, loss_every, norm_entity_l2sum #+ loss_regularizer*self.__regularizer_weight



	def test(self, inputs):
		embedding_relation = self.__embedding_relation
		embedding_entity = self.__embedding_entity
		normal_vector = self.__normal_vectors
		norm_normal_vector = tf.nn.l2_normalize(normal_vector, dim = 1)
		
		triple_test = inputs # (headid, tailid, tailid)
		def mult(e,n):
			print(n.get_shape())
			norm = tf.reshape(n,[self.__dimension,1])
			# print(e.get_shape(),(tf.reshape(n,[self.__dimension,1])).get_shape())
			return e - tf.matmul(tf.reduce_sum(tf.matmul(e,norm),1,keep_dims=True), tf.transpose(norm))						

		normal_vec = tf.nn.embedding_lookup(normal_vector,triple_test[1])
		norm_normal_vec = tf.nn.embedding_lookup(norm_normal_vector,triple_test[1])

		print(embedding_entity.get_shape())
		embedding_entity = mult(embedding_entity,normal_vec)
		print(embedding_entity.get_shape())

		head_vec = tf.nn.embedding_lookup(embedding_entity, triple_test[0])
		rel_vec = tf.nn.embedding_lookup(embedding_relation, triple_test[1])
		tail_vec = tf.nn.embedding_lookup(embedding_entity, triple_test[2])

		norm_embedding_entity = mult(tf.nn.l2_normalize(embedding_entity, dim =1 ),norm_normal_vec)
		norm_embedding_relation = tf.nn.l2_normalize(embedding_relation, dim = 1)
		norm_head_vec = tf.nn.embedding_lookup(norm_embedding_entity, triple_test[0])
		norm_rel_vec = tf.nn.embedding_lookup(norm_embedding_relation, triple_test[1])
		norm_tail_vec = tf.nn.embedding_lookup(norm_embedding_entity, triple_test[2])

	
		def transfer( e, n):
			norm = tf.transpose(n)
			return e - tf.reduce_sum(e * norm, 1, keep_dims = True) * norm
			# return (e - tf.multiply(e,norm))

		# e_h = transfer([head_vec],[normal_vec])
		# e_t = transfer([tail_vec],[normal_vec])
		# n_e_h = transfer([norm_head_vec],[norm_normal_vec])
		# n_e_t = transfer([norm_tail_vec],[norm_normal_vec])
		
		
		# print(embedding_entity.get_shape(),normal_vec.get_shape())
		_, id_replace_head = tf.nn.top_k(tf.reduce_sum(tf.abs(embedding_entity + rel_vec - tail_vec), axis=1), k=self.__num_entity)
		_, id_replace_tail = tf.nn.top_k(tf.reduce_sum(tf.abs(head_vec + rel_vec - embedding_entity), axis=1), k=self.__num_entity)

		_, norm_id_replace_head = tf.nn.top_k(tf.reduce_sum(tf.abs(norm_embedding_entity + norm_rel_vec - tail_vec), axis=1), k=self.__num_entity)
		_, norm_id_replace_tail = tf.nn.top_k(tf.reduce_sum(tf.abs(head_vec + norm_rel_vec - norm_embedding_entity), axis=1), k=self.__num_entity)

		return id_replace_head, id_replace_tail, norm_id_replace_head, norm_id_replace_tail

		
def train_operation(model, learning_rate=0.01, margin=1.0, optimizer_str = 'gradient'):
	with tf.device('/cpu'):
		train_triple_positive_input = tf.placeholder(tf.int32, [None, 3])
		train_triple_negative_input = tf.placeholder(tf.int32, [None, 3])

		loss, loss_every, norm_entity = model.train([train_triple_positive_input, train_triple_negative_input])
		if optimizer_str == 'gradient':
			optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
		elif optimizer_str == 'rms':
			optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
		elif optimizer_str == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		else:
			raise NotImplementedError("Dose not support %s optimizer" %optimizer_str)

		grads = optimizer.compute_gradients(loss, model.variables)
		op_train = optimizer.apply_gradients(grads)

		return train_triple_positive_input, train_triple_negative_input, loss, op_train, loss_every, norm_entity

def test_operation(model):
	with tf.device('/cpu'):
		test_triple = tf.placeholder(tf.int32, [3])
		print('finish palceholder')
		head_rank, tail_rank, norm_head_rank, norm_tail_rank = model.test(test_triple)
		print('finish model.test')
		return test_triple, head_rank, tail_rank, norm_head_rank, norm_tail_rank


def main():
	parser = argparse.ArgumentParser(description = "TransE")
	parser.add_argument('--data_dir', dest='data_dir', type=str, help='the directory of dataset', default='./')
	parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning rate', default=0.01)
	parser.add_argument('--batch_size', dest='batch_size', type=int, help="batch size", default=4096)
	parser.add_argument('--max_iter', dest='max_iter', type=int, help='maximum interation', default=100)
	parser.add_argument('--optimizer', dest='optimizer', type=str, help='optimizer', default='adam')
	parser.add_argument('--dimension', dest='dimension', type=int, help='embedding dimension', default=100)
	parser.add_argument('--margin', dest='margin', type=float, help='margin', default=1.0)
	parser.add_argument('--norm', dest='norm', type=str, help='L1 or L2 norm', default='L1')
	parser.add_argument('--evaluation_size', dest='evaluation_size', type=int, help='batchsize for evaluation', default=500)
	parser.add_argument('--save_dir', dest='save_dir', type=str, help='directory to save tensorflow checkpoint directory', default='output/')
	parser.add_argument('--negative_sampling', dest='negative_sampling', type=str, help='choose unit or bern to generate negative examples', default='bern')
	parser.add_argument('--evaluate_per_iteration', dest='evaluate_per_iteration', type=int, help='evaluate the training result per x iteration', default=10)
	parser.add_argument('--evaluate_worker', dest='evaluate_worker', type=int, help='number of evaluate workers', default=4)
	parser.add_argument('--regularizer_weight', dest='regularizer_weight', type=float, help='regularization weight', default=1e-5)
	parser.add_argument('--n_test', dest = 'n_test', type=int, help='number of triples for test during the training', default = 300)
	args = parser.parse_args()
	print(args)
	model = TransE(negative_sampling=args.negative_sampling, data_dir=args.data_dir,
				   learning_rate=args.learning_rate, batch_size=args.batch_size,
				   max_iter=args.max_iter, margin=args.margin, 
				   dimension=args.dimension, norm=args.norm, evaluation_size=args.evaluation_size, 
				   regularizer_weight = args.regularizer_weight)

	train_triple_positive_input, train_triple_negative_input, loss, op_train, loss_every, norm_entity = train_operation(model, learning_rate = args.learning_rate, margin = args.margin, optimizer_str = args.optimizer)
	test_triple, head_rank, tail_rank , norm_head_rank, norm_tail_rank= test_operation(model)

	saver = tf.train.Saver()
	with tf.Session() as session:
		saver.restore(session,'./data_0-transH-100-1000')
		# tf.initialize_all_variables().run()


		#print("[iter %d] after l2 normalization the entity vectors: %s"%(n_iter, str(norm_e[:10])))
		#print("[iter %d] after training the entity vectors: %s"%(n_iter, str(session.run(tf.sqrt(tf.reduce_sum(model.embedding_entity**2, axis = 1))[:10]))))

		rank_head = []
		rank_tail = []
		filter_rank_head = []
		filter_rank_tail = []

		norm_rank_head = []
		norm_rank_tail = []
		norm_filter_rank_head = []
		norm_filter_rank_tail = []

		start = timeit.default_timer()
		testing_data = model.testing_data
		hr_t = model.hr_t
		tr_h = model.tr_h
		n_test = model.__num_triple_test
		
		for i in range(n_test):
			print('[%.2f sec] --- testing[%d/%d]' %(timeit.default_timer()-start, i+1, n_test), end='\r')
			t = testing_data[i]
			id_replace_head , id_replace_tail, norm_id_replace_head , norm_id_replace_tail  = session.run([head_rank, tail_rank, norm_head_rank, norm_tail_rank], {test_triple:t})
			
			hrank = 0
			fhrank = 0
			for i in range(len(id_replace_head)):
				val = id_replace_head[-i-1]
				if val == t[0]:
					break						
				else: 
					hrank += 1
					fhrank += 1 
					if val in tr_h[(t[2],t[1])]:
						fhrank -= 1

			norm_hrank = 0
			norm_fhrank = 0
			for i in range(len(norm_id_replace_head)):
				val = norm_id_replace_head[-i-1]
				if val == t[0]:
					break						
				else: 
					norm_hrank += 1
					norm_fhrank += 1 
					if val in tr_h[(t[2],t[1])]:
						norm_fhrank -= 1
							

			trank = 0
			ftrank = 0
			for i in range(len(id_replace_tail)):
				val = id_replace_tail[-i-1]
				if val == t[2]:
					break
				else:
					trank += 1
					ftrank += 1
					if val in hr_t[(t[0], t[1])]:
						ftrank -= 1

			norm_trank = 0
			norm_ftrank = 0
			for i in range(len(norm_id_replace_tail)):
				val = norm_id_replace_tail[-i-1]
				if val == t[2]:
					break
				else:
					norm_trank += 1
					norm_ftrank += 1
					if val in hr_t[(t[0], t[1])]:
						norm_ftrank -= 1

			rank_head.append(hrank)
			rank_tail.append(trank)
			filter_rank_head.append(fhrank)
			filter_rank_tail.append(ftrank)

			norm_rank_head.append(norm_hrank)
			norm_rank_tail.append(norm_trank)
			norm_filter_rank_head.append(norm_fhrank)
			norm_filter_rank_tail.append(norm_ftrank)

		# mean_rank_head = np.sum(rank_head, dtype=np.float32)/n_test
		# mean_rank_tail = np.sum(rank_tail, dtype=np.float32)/n_test
		filter_mean_rank_head = np.sum(filter_rank_head, dtype=np.float32)/n_test
		filter_mean_rank_tail = np.sum(filter_rank_tail, dtype=np.float32)/n_test

		# norm_mean_rank_head = np.sum(norm_rank_head, dtype=np.float32)/n_test
		# norm_mean_rank_tail = np.sum(norm_rank_tail, dtype=np.float32)/n_test
		norm_filter_mean_rank_head = np.sum(norm_filter_rank_head, dtype=np.float32)/n_test
		norm_filter_mean_rank_tail = np.sum(norm_filter_rank_tail, dtype=np.float32)/n_test

		# hit10_head = np.sum(np.asarray(np.asarray(rank_head)<10 , dtype=np.float32))/n_test
		# hit10_tail = np.sum(np.asarray(np.asarray(rank_tail)<10 , dtype=np.float32))/n_test
		norm_filter_hit10_head = np.sum(np.asarray(np.asarray(norm_filter_rank_head)<10 , dtype=np.float32))/n_test
		norm_filter_hit10_tail = np.sum(np.asarray(np.asarray(norm_filter_rank_tail)<10 , dtype=np.float32))/n_test
		norm_filter_hit3_head = np.sum(np.asarray(np.asarray(norm_filter_rank_head)<3 , dtype=np.float32))/n_test
		norm_filter_hit3_tail = np.sum(np.asarray(np.asarray(norm_filter_rank_tail)<3 , dtype=np.float32))/n_test
		norm_filter_hit1_head = np.sum(np.asarray(np.asarray(norm_filter_rank_head)<1 , dtype=np.float32))/n_test
		norm_filter_hit1_tail = np.sum(np.asarray(np.asarray(norm_filter_rank_tail)<1 , dtype=np.float32))/n_test

		# norm_hit10_head = np.sum(np.asarray(np.asarray(norm_rank_head)<10 , dtype=np.float32))/n_test
		# norm_hit10_tail = np.sum(np.asarray(np.asarray(norm_rank_tail)<10 , dtype=np.float32))/n_test
		# norm_filter_hit10_head = np.sum(np.asarray(np.asarray(norm_filter_rank_head)<10 , dtype=np.float32))/n_test
		# norm_filter_hit10_tail = np.sum(np.asarray(np.asarray(norm_filter_rank_tail)<10 , dtype=np.float32))/n_test

		# print('iter:%d --mean rank: %.2f --hit@10: %.2f' %(n_iter, (mean_rank_head+ mean_rank_tail)/2, (hit10_tail+hit10_head)/2))
		print('norm filter mean rank: %.2f  hit@10: %.2f hit@3: %.2f hit@1: %.2f' %(n_iter, (norm_filter_mean_rank_head+ norm_filter_mean_rank_tail)/2, (norm_filter_hit10_tail+norm_filter_hit10_head)/2),(norm_filter_hit3_tail+norm_filter_hit3_head)/2,(norm_filter_hit1_tail+norm_filter_hit1_head)/2)

		# print('iter:%d --norm mean rank: %.2f --norm hit@10: %.2f' %(n_iter, (norm_mean_rank_head+ norm_mean_rank_tail)/2, (norm_hit10_tail+norm_hit10_head)/2))
		# print('iter:%d --norm filter mean rank: %.2f --norm filter hit@10: %.2f' %(n_iter, (norm_filter_mean_rank_head+ norm_filter_mean_rank_tail)/2, (norm_filter_hit10_tail+norm_filter_hit10_head)/2))
			

if __name__ =="__main__":
	main()