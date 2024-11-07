# if __name__=='__main__' --> 이건 무조건 치고 들어가야 함. (그냥 손에서 바로바로 나올 정도로)
# 부수적인 함수들은 따로 빼서 코드 작성 def
# 다른 파일에 함수나 전처리 코드 등을 저장시켜놓고 from import 써서 불러오기(정리하는 방법)
# 코딩에서는 디버깅 하는 것이 가장 중요하다. -> 디버깅하는 습관 들여놓기.

from contextlib import suppress
from xml.dom import HierarchyRequestErr

import numpy as np
import nltk
from nltk.corpus import stopwords
import os, re
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# from text0924 import filtered_content, NN_words

#nltk.download('all')

def load_data(path):
    onlyfiles = [f for f in os.listdir(path) if isfile(join(mypath, f))]
    onlyfiles.sort()

    total_docs = []
    for file in onlyfiles:
        file_path = mypath + file
        with open(file_path, 'r', encoding='utf8') as f:
            content = f.read()
        total_docs.append(content)

    return total_docs

def En_preprocessing(text):
    #1. 불필요한 symbols과 marks 제거하기
    filtered_content = re.sub(r'[^\s\d\w]', '',text)

    #2. Case conversion: 대문자를 소문자로
    filtered_content = filtered_content.lower()

    #3. Word tokenization
    word_tokens = nltk.word_tokenize(filtered_content)

    #4. POS tagging
    tokens_pos = nltk.pos_tag(word_tokens)

    #5. Select Noun words
    NN_words = []
    for word, pos in tokens_pos:
        if 'NN' in pos:
            NN_words.append(word)

    #6. Stopwords removal
    stopwords_list = stopwords.words('english')
    # print('stopwords: ', stopwords_list)
    unique_NN_words = set(NN_words)
    final_NN_words = NN_words

    for word in unique_NN_words:
        if word in stopwords_list:
            while word in final_NN_words: final_NN_words.remove(word)

    return final_NN_words

def plot_dend(feature):
    np.set_printoptions(precision=2, suppress=True)
    Z = linkage(feature, 'ward')
    plt.figure(figsize=(15,8))
    plt.title('Hierarchical Clustering Dendrogram', fontsize=18)
    plt.xlabel('Document ID', fontsize = 18)
    plt.ylabel('Distance', fontsize = 18)
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=14.,
    )
    plt.show()

if __name__== "__main__":
    mypath = './example_clustering/'
    total_docs = load_data(mypath)

    docs_nouns = [En_preprocessing(doc) for doc in total_docs]
    documents_filtered = [' '.join(doc) for doc in docs_nouns]

# 컴퓨터가 알아들을 수 있게 벡터화
    tfidf_vectorizer = TfidfVectorizer()
    DTM_tfidf = tfidf_vectorizer.fit_transform(documents_filtered)
    DTM_TFIDF = np.array(DTM_tfidf.todense())

# 차원 축소
    pca = PCA(n_components=10)
    pca_result_tfidf = pca.fit_transform(DTM_TFIDF)

 # 클러스터링
#AgglomerativeClustering = 위계적 군집 분석
    agg = AgglomerativeClustering(linkage='average', affinity='cosine' ,n_clusters=5)
    clusters = agg.fit_predict(pca_result_tfidf)

    # agg_ward = AgglomerativeClustering(linkage='ward', n_clusters=5)
    # clusters_ward = agg_ward.fit_predict(pca_result_tfidf)

    print(clusters)

    plot_dend(pca_result_tfidf)