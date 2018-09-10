import re

entities = set()
relations = set()
tuples = []

files = ["Film_dataset/film.yago.graph","Film_dataset/actor.yago.graph","Film_dataset/mix.yago.graph"]
for name in files:
	with open(name,"r") as file:
		for line in file:
			line = re.sub('[^(-)A-Za-z0-9_: \t]+', '', line)
			row = line.split("	")
			h,r,t = row[0],row[1],row[2]
			tuples.append((h,r,t))
			entities.add(h)
			entities.add(t)
			relations.add(r)

print(len(entities),len(relations),len(tuples))
entities = list(entities)
relations = list(relations)

entity2id = {entities[i]:str(i) for i in range(len(entities))}
relation2id = {relations[i]:str(i) for i in range(len(relations))}

with open("entity2id.txt","w") as file:
	for x in entities:
		file.write(x+" "+entity2id[x]+"\n")

with open("relation2id.txt","w") as file:
	for x in relations:
		file.write(x+" "+relation2id[x]+"\n")

with open("test.txt","w") as file2:
	with open("train.txt","w") as file:
		i = 0
		for x in tuples:
			if(x[1] == "rdf:type"):
				i += 1
				if(i%4 == 0):	#10%
					file2.write(entity2id[x[0]]+" "+relation2id[x[1]]+" "+entity2id[x[2]]+"\n")
				
				else:	file.write(entity2id[x[0]]+" "+relation2id[x[1]]+" "+entity2id[x[2]]+"\n")
		
			else:	file.write(entity2id[x[0]]+" "+relation2id[x[1]]+" "+entity2id[x[2]]+"\n")
