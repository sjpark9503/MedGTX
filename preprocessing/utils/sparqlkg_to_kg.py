from rdflib import Graph, URIRef
from tqdm import tqdm
import pickle

# kg 로드 1분걸려요
kg = Graph()
kg.parse('mimic_sparqlstar_kg.xml', format='xml', publicID='/')

def build_dict(triples, nodes, edges):
    h, r, t = triples
    #for (h,r,t) in triples:
    if h not in nodes:
        nodes[h]=1
    else:
        nodes[h]+=1
    if t not in nodes:
        nodes[t]=1
    else:
        nodes[t]+=1
    if r not in edges:
        edges[r]=1
    else:
        edges[r]+=1
    return nodes, edges

# triple 확인
nodes = dict()
edges = dict()
for triple in tqdm(kg):
    triples = [x.n3() for x in triple]
    #print(triples)
    nodes, edges = build_dict(triples, nodes, edges)
#matching = [s for s in tqdm(list(nodes.keys())) if "hadm_id" in list(nodes.keys())]
print(len(nodes))
print(len(edges))

f = open('node_dict','w')
g = open('edge_dict','w')
for node in list(nodes.keys()):
    f.write('{}\n'.format(node))
for edge in list(edges.keys()):
    g.write('{}\n'.format(edge))
