from collections import defaultdict
import numpy as np

class Result(object):
	def __init__(self, ranks, raw_ranks):
		self.ranks = ranks
		self.raw_ranks = raw_ranks
		self.mrr = np.mean(1.0 / ranks)
		self.raw_mrr = np.mean(1.0 / raw_ranks)

		cnt = float(len(ranks))

		self.hits_at1 = np.sum(ranks <= 1) / cnt
		self.hits_at3 = np.sum(ranks <= 3) / cnt
		self.hits_at10 = np.sum(ranks <= 10) / cnt

		self.raw_hits_at1 = np.sum(raw_ranks <= 1) / cnt
		self.raw_hits_at3 = np.sum(raw_ranks <= 3) / cnt
		self.raw_hits_at10 = np.sum(raw_ranks <= 10) / cnt

class Scorer(object):
	def __init__(self, train, valid, test, n_entities):
		self.known_obj_triples = defaultdict(list)
		self.known_sub_triples = defaultdict(list)
		self.n_entities = n_entities
		self.types = []
		with open("./data/data_25/etypes.txt","r") as file:
			for line in file:
				self.types.append(int(line.strip()))
		self.update_known_triples(train[:100])
		self.update_known_triples(test[:100])

		if valid is not None:
			self.update_known_triples(valid[:100])

	
	def update_known_triples(self, triples):
		for i, j, k in triples:
			self.known_obj_triples[(i, j)].append(k)
			self.known_sub_triples[(j, k)].append(i)
	
	def compute_scores(self, predict_func, eval_set):
		# preds = predict_func(eval_set)

		nb_test = len(eval_set)
		ranks = np.empty(2*nb_test)
		raw_ranks = np.empty(2*nb_test)


		idx_obj_mat = np.empty((self.n_entities, 3), dtype=np.int64)
		idx_sub_mat = np.empty((self.n_entities, 3), dtype=np.int64)
		idx_obj_mat[:,2] = np.arange(self.n_entities)
		idx_sub_mat[:,0] = np.arange(self.n_entities)

		def eval_o(i, j):
			idx_obj_mat[:,:2] = np.tile((i,j), (self.n_entities,1))
			return predict_func(idx_obj_mat)
		def eval_s(j, k):
			idx_sub_mat[:,1:] = np.tile((j,k), (self.n_entities,1))
			return predict_func(idx_sub_mat)
		l,x = [[] for i in range(14)],[[] for i in range(14)]
		m = [1 for i in range(14)]
		o_test,o_train,o_percent = 0,0,0.0
		s_test,s_train,s_percent = 0,0,0.0

		f1 = open("s_test.txt","a")
		f2 = open("s_typ.txt","a")
		f3 = open("o_test.txt","a")
		f4 = open("o_typ.txt","a")
		flag = 1
		
		for a, (i,j,k) in enumerate(eval_set):
			res_obj = eval_o(i, j)
			raw_ranks[a] = np.sum(res_obj >= res_obj[k])
			ranks[a] = raw_ranks[a] - np.sum(res_obj[self.known_obj_triples[(i,j)]] >= res_obj[k]) + 1
			ranks[a] = max(1, ranks[a])
			raw_ranks[nb_test+a] = ranks[a]
			# if ranks[a]<10:
			# 	f3.write(str(i)+" "+str(j)+" "+str(k)+" "+str(ranks[a])+"\n")
			# elif ranks[a]>100:
			# 	f4.write(str(i)+" "+str(j)+" "+str(k)+" "+str(ranks[a])+"\n")
			#x = np.sum(res_obj[self.types] >= res_obj[k])
			# raw_ranks[a] = ranks[a]
			dum = ranks[a]
			for k in self.known_obj_triples[(i,j)]:
				y = np.sum(res_obj >= res_obj[k])
				if y < ranks[a]:
					ranks[a] = y
			if flag == 1 and j == 12 and ranks[a] == dum:	
				o_test += 1
				f3.write(str(i)+" "+str(j)+" "+str(k)+" "+str(ranks[a])+"\n")
			elif j == 12 and flag == 1:	
				o_train += 1
			if j == 12 and flag == 1:
				y = np.argsort(res_sub)[-10:]
				for i in y:
					if i in self.types:
						o_percent += 1
		


			res_sub = eval_s(j, k)
			ranks[a] = np.sum(res_sub >= res_sub[i])
			ranks[nb_test+a] = ranks[a] - np.sum(res_sub[self.known_sub_triples[(j,k)]] >= res_sub[i]) + 1
			ranks[nb_test+a] = max(1, ranks[nb_test+a])
			if flag == 1 and j == 12 and ranks[nb_test+a] == dum:	
				s_test += 1
				f1.write(str(i)+" "+str(j)+" "+str(k)+" "+str(ranks[nb_test+a])+"\n")
			elif j == 12 and flag == 1:	
				s_train += 1
			if j == 12 and flag == 1:
				y = np.argsort(res_sub)[-10:]
				for i in y:
					if i in self.types:
						s_percent += 1
			
			#ranks[a] = x
			# if int(ranks[nb_test+a])<10:
			# 	f1.write(str(i)+" "+str(j)+" "+str(k)+" "+str(ranks[nb_test+a])+"\n")
			# elif int(ranks[nb_test+a]>100):
			# 	f2.write(str(i)+" "+str(j)+" "+str(k)+" "+str(ranks[nb_test+a])+"\n")
			#y = 0
			#for m in self.types:
			#	if(res_obj[m] >= res_obj[k]):
			#		y+= 1
			#ranks[a] = y
			#if(x == y):	print("D",end='')
			#raw_ranks[nb_test+a] = ranks[nb_test+a]
			for k in self.known_sub_triples[(j,k)]:
				y = np.sum(res_sub >= res_sub[k])
				if y < ranks[nb_test+a]:
					ranks[nb_test+a] = y
			#ranks[nb_test+a] = y
			l[j].append(ranks[a])
			x[j].append(raw_ranks[a])
			x[j].append(raw_ranks[nb_test+a])
			m[j] += 2
			l[j].append(ranks[nb_test+a])
		h1,h3,h10,mrr = [],[],[],[]
		i = -1
		print(s_test,s_train,s_percent,s_percent*10.0/((s_train+s_test)))
		print(o_test,o_train,o_percent,o_percent*10.0/((o_train+o_test)))
		
		f1.write("\n\n\n\n\n")
		f1.close()
		f2.write("\n\n\n\n\n")
		f2.close()
		f3.write("\n\n\n\n\n")
		f3.close()
		f4.write("\n\n\n\n\n")
		f4.close()
		for lis in l:
			i += 1
			x1,x3,x10 = 0,0,0
			for k in lis:
				if k<=1:	x1 += 1
				if k<=3:	x3 += 1
				if k<=10:	x10 += 1
			h1.append(x1/m[i])
			h3.append(x3/m[i])
			h10.append(x10/m[i])
			mrr.append(np.mean(1.0/np.array(lis)))
		print(h1)
		print(h3)
		print(h10)
		print(mrr)
		for i in range(10):
			x1,x3,x10 = 0,0,0
			for k in x[i]:
				if k <= 1:	x1 += 1
				if k <= 3:	x3 += 1
				if k <= 10:	x10 += 1
			h1[i],h3[i],h10[i] = x1/m[i],x3/m[i],x10/m[i]
			mrr[i] = np.mean(1.0/np.array(x[i]))
		print("raw")
		print(h1)
		print(h3)
		print(h10)
		print(mrr)
		return Result(ranks, raw_ranks)

class RelationScorer(object):
	def __init__(self, train, valid, test, n_relations):
		self.known_rel_triples = defaultdict(list)
		self.n_relations = n_relations
		self.types = []
		#with open("../data/data_25/etypes.txt","r") as file:
		#	for line in file:
		#		self.types.append(int(line.strip()))
		self.update_known_triples(train)
		self.update_known_triples(test)
		if valid is not None:
			self.update_known_triples(valid)
	
	def update_known_triples(self, triples):
		for i, j, k in triples:
			self.known_rel_triples[(i,k)].append(j)
	
	def compute_scores(self, predict_func, eval_set):
		# preds = predict_func(eval_set)

		nb_test = len(eval_set)
		ranks = np.empty(nb_test)
		raw_ranks = np.empty(nb_test)

		idx_rel_mat = np.empty((self.n_relations, 3), dtype=np.int64)
		idx_rel_mat[:,1] = np.arange(self.n_relations)

		def eval_r(i, j):
			idx_rel_mat[:,0] = i*np.ones(self.n_relations) 
			idx_rel_mat[:,2] = j*np.ones(self.n_relations)
			return predict_func(idx_rel_mat)

		for a, (i,j,k) in enumerate(eval_set):
			res = eval_r(i, k)
			raw_ranks[a] = np.sum(res >= res[j])
			ranks[a] = raw_ranks[a] - np.sum(res[self.known_rel_triples[(i, k)]] >= res[j]) + 1
			ranks[a] = max(1, ranks[a])

		return Result(ranks, raw_ranks)
