
import os
import natsort
import string
import numpy as np
import numpy.linalg as LA
import pandas as pd 
from nltk. tokenize import word_tokenize
from nltk.stem import porter
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

#os.chdir('/Users/sayo9/COSC 329/A2/alldocs')
#print(os.getcwd())
#print(os.listdir())
#filenames = os.listdir()
all_txt_files = []
all_doc = []
def preprocess ():
    for file in Path("alldocs").rglob("*.txt"):
        all_txt_files.append(file.parent/file.name)
        # sorting files into natural order t1, t2 ...
        natsort.natsorted(all_txt_files)
    for txt_file in all_txt_files:     
        with open(txt_file, 'r', encoding = "utf8") as f:
            txt_file_as_string = f.read()   
            #txt_file_as_string = txt_file_as_string.lower()   
            txt_file_as_string = re.sub( r'[^a-z\n ]','', txt_file_as_string )   
            txt_file_as_string =word_tokenize(txt_file_as_string)   
            txt_file_as_string = [w for w in txt_file_as_string if len(w) >= 3 if not w in stopwords.words('english')]
            stemmer = porter.PorterStemmer()           
            for i in range(len(txt_file_as_string)):
                txt_file_as_string[i] = stemmer.stem(txt_file_as_string[i])
                
            txt_file_as_string = " ".join(txt_file_as_string) 

            all_doc.append(txt_file_as_string)
    return all_doc

cleaned_docs = preprocess()
#txt1 = cleaned_docs[0]
document_vectore = pd.DataFrame(cleaned_docs)
print(document_vectore)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([cleaned_docs.lower() for cleaned_docs in all_doc])

df2 = pd.DataFrame(tfidf_matrix)
print(df2)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(cosine_sim)

df = cosine_sim
df = df[:,0:28]

Z = linkage(df, method = 'ward')

with open("preprocess.txt", "w", encoding="utf-8") as f:
    for row in all_doc:
        f.write(str(row))
        f.write('\n')
    
document_vectore.to_csv("vector.txt")
with open("cosine.txt", "w", encoding="utf-8") as f:

    for row in cosine_sim:
        f.write(str(row))
        f.write('\n')

#plotting dendrogram
dendro = dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')
plt.show()

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=2)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
