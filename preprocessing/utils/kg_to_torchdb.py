import torch
from rdflib import Graph, URIRef
from tqdm import tqdm
print('Loading Dict')
nodes = open('node_dict').read().splitlines()
node_dict = {k:v for (v,k) in enumerate(nodes)}
edges = open('edge_dict').read().splitlines()
edge_dict = {k:v for (v,k) in enumerate(edges)}
print('Loading KG')
kg = Graph()
kg.parse('mimic_sparqlstar_kg.xml', format='xml', publicID='/')
print('Build DB')
DB = torch.zeros(len(kg),3)
#nodes = open('node_dict').read().splitlines()
#edges = open('edge_dict').read().splitlines()
i=0
for triple in tqdm(kg):
    DB[i,:] = torch.tensor([node_dict[triple[0]],edge_dict[triple[1]],node_dict[triple[2]]])
    i+=1
    #nodes, edges = build_dict(str_triple, nodes, edges)
    #matching = [s for s in tqdm(list(nodes.keys())) if "hadm_id" in list(nodes.keys())]
print('Save DB')
torch.save(DB,'triple_DB')
