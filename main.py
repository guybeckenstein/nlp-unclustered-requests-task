import json
from collections import Counter

import nltk
import pandas as pd
import numpy as np
from nltk import ngrams
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
# from compare_clustering_solutions import evaluate_clustering

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stop_words: set = set(stopwords.words('english'))

THRESHOLD: float = 0.631


def extract_data_set(data_file: str) -> np.ndarray:
    """
    :param data_file: String of the relative path of our CSV dataset. We open it using Pandas package
    :return: A numpy vector, each value of it represents a sentence from the dataset. We filtered the file and used
    only the 'request' column.
    """

    print(f'Step 1\t->\tFetching features-vector data (unrecognized requests) from {data_file}')
    df: pd = pd.read_csv(data_file)
    X = df['request'].values  # all requests
    return X


def get_sbert_model(X):
    """
    :param X: A numpy vector, each value of it represents a sentence from the dataset
    :param is_step_two: A boolean value, which is true only if the current step is step #2
    :return: A SciPy sparse matrix, each line vector of it represents a sentence from the dataset
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    feature_vector = model.encode(X)  # Sentences we want to encode, are encoded by calling model.encode()

    return feature_vector


def find_clusters(X: np.ndarray, min_size: str) -> tuple:
    """
    :param X: X of the dataset - all requests we got. NDArray of strings
    :param min_size: a hyperparameter we got from the 'analyze_unrecognized_requests' function. Minimum size of cluster
    :return: All requests' clusters we found using our clustering algorithm, and all unclustered requests.
    """
    print("Step 2\t->\tUsing the SentenceTransformer's community detection algorithm for finding clusters")
    print(f'\t\t\tOur distance threshold for SentenceTransformer: community detection = {THRESHOLD}')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Each sentence s_k is represented with an index i_k. This means s_k=X[i_k]
    corpus_embeddings = model.encode(X, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
    clusters = util.community_detection(corpus_embeddings, min_community_size=int(min_size), threshold=THRESHOLD)

    print(f'\t\t\tSentenceTransformer: community detection has found {len(clusters)} clusters!')

    sentences_idx_lst: list = list(range(X.size))

    # We create a set of all clustered indexes (for complexity reasons)
    clustered_idx_set = set()
    for cluster in clusters:
        cluster_set = set(cluster)
        clustered_idx_set = clustered_idx_set.union(cluster_set)
    # Now we get all unclustered indexes
    unclustered: list = [idx for idx in sentences_idx_lst if idx not in clustered_idx_set]
    print(f'\t\t\tThere are {len(unclustered)} unclustered requests, i.e., unresolved')

    return clusters, unclustered


def get_cluster_representatives(X_cluster, k):
    """
    :param X_cluster: X filtered to our current cluster only.
    :param k: a hyperparameter we got from the 'analyze_unrecognized_requests' function. K representatives we pick.
    :return: K representatives we chose of this cluster, using the algorithm below.
    """
    cluster = get_sbert_model(X_cluster)
    size = cluster.shape[0]
    # We get the distances between each coordinate in the cluster matrix
    distances_matrix = np.zeros((size, size))  # Distances matrix INIT
    for idx1 in range(size):
        for idx2 in range(size):
            if idx1 != idx2:
                distances_matrix[idx1, idx2] = np.linalg.norm(cluster[idx1] - cluster[idx2])
    rep_list = list()
    # We stop only when we get k representatives
    while len(rep_list) < int(k):
        if rep_list == list():
            # INIT
            rep_list.append(distances_matrix.argmax() // size)
            rep_list.append(distances_matrix.argmax() % size)
        else:
            distances_vector = np.zeros(size)
            # We find the minimal distance for each coordinate
            for idx1 in range(size):
                if idx1 not in rep_list:
                    min_rep = distances_matrix[rep_list[0], idx1]
                    for rep_idx in range(1, len(rep_list)):
                        curr_rep = distances_matrix[rep_list[rep_idx], idx1]
                        if min_rep > curr_rep:
                            min_rep = curr_rep
                    distances_vector[idx1] = min_rep
                else:
                    distances_vector[idx1] = 0
            # Now we pick the maximum distance (maximum of minimal)
            rep_list.append(distances_vector.argmax())

    return X_cluster[rep_list]


def get_all_clusters_representatives(X, clusters, k):
    """
    :param X: X of the dataset - all requests we got. NDArray of strings
    :param clusters: A list L of lists l_i. Each l_i contains indexes of strings in X that belongs to cluster i.
    :param k: a hyperparameter we got from the 'analyze_unrecognized_requests' function. K representatives we pick.
    :return: A matrix of |L|xK. Each line i contains the k representatives of cluster i.
    """
    print('Step 3\t->\tFor each cluster, achieving K cluster representatives satisfying the property of diversity')
    total_valid_clusters = len(clusters)
    clusters_representatives_matrix = np.empty((total_valid_clusters, int(k)), dtype='U175')
    # For each row in the 2d matrix of ours, we make a placement for the k-length vector of the k-representatives.
    # The total iterations are as the total clusters we achieved so far
    print(f'\t\t\tTotal of {total_valid_clusters} novel clusters')
    for cluster_idx in range(total_valid_clusters):
        print(f'\t\t\tCluster #{(cluster_idx + 1)} size: {len(clusters[cluster_idx])}')
        curr_cluster_representatives = get_cluster_representatives(X[clusters[cluster_idx]], k)
        clusters_representatives_matrix[cluster_idx] = curr_cluster_representatives
    return clusters_representatives_matrix


def get_clusters_names(X, clusters) -> list:
    """
    :param X: X of the dataset - all requests we got. NDArray of strings
    :param clusters: A list L of lists l_i. Each l_i contains indexes of strings in X that belongs to cluster i.
    :return: A list L that contains |clusters| labels. L[i] is the chosen label for cluster i.
    """
    print('Step 4\t->\tPerforming cluster naming (labeling)')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')


    clusters_names_lst = []

    possible_seq = [
        ['NNS', 'NN'], ['NN', 'NN'], ['JJ', 'NNS'], ['VB', 'DT', 'NN'], ['VBG', 'NN'], ['VBN', 'NN'],
        ['NN', 'NNS'], ['VBN', 'IN', 'NN'], ['IN', 'DT', 'NN'], ['VB', 'DT', 'NN', 'NN'],
        ['NN', 'IN', 'NN'], ['VB', 'RB', 'TO', 'NN'], ['RB', 'NNS'], ['IN', 'DT', 'NN'], ['VBN', 'NN'],
        ['JJ', 'NNS'], ['JJ', 'VBP', 'DT'], ['NN', 'DT', 'NN'], ['VBN', 'IN', 'DT', 'NN'],
        ['VBN', 'TO', 'NN'], ['VB', 'IN', 'DT', 'NN'], ['VB', 'NNS'], ['VB', 'TO', 'DT', 'NN'],
        ['WP', 'PRP', 'VBP'], ['NN', 'TO', 'DT', 'NNS'], ['VB', 'DT', 'NN']
    ]

    for cluster_idx in range(len(clusters)):
        name = choose_cluster_label(X[clusters[cluster_idx]], cluster_idx)
        clusters_names_lst.append(name)
    return clusters_names_lst


def preprocess_text(text) -> list[str]:
    """
    Preprocesses a string of text by converting it to lowercase, tokenizing it into individual words,
    removing stop words and non-alphabetic characters, and lemmatizing the remaining words.
    :param text: The text to preprocess.
    :return: List[str]: The preprocessed list of words.
    """
    # Tokenize the text into individual words
    words = word_tokenize(text.lower())

    # Remove any stop words or non-alphabetic characters, and lemmatize the remaining words
    preprocessed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]

    return preprocessed_words


def choose_cluster_label(X_cluster, cluster_idx):
    """
    :param X_cluster: X_cluster: X filtered to our current cluster only.
    :param cluster_idx: The index of X_cluster
    :return: The most frequent n-gram that has a reasonable POS sequence or the result that LDA algorithm returns
    """
    print(f'\t\t\tFinding the best label for cluster #{(cluster_idx + 1)}')

    lst = []
    for sentence in X_cluster:
        lst.append(preprocess_text(sentence))  # pre-process the data of the model

    # If the pre-process function removed all relevant information from the sentences
    if all(not sublist for sublist in lst):
        ngram_freq = Counter()
        for i in range(2, 5):
            for sentence in X_cluster:
                words = nltk.word_tokenize(sentence)
                filtered_words = [w.lower() for w in words if w.lower() and w.isalpha()]
                for gram in ngrams(filtered_words, i):
                    ngram_freq[gram] += 1

        sorted_counter = sorted(ngram_freq.items(), key=lambda x: len(x[0]), reverse=True)
        top_ngrams = sorted(sorted_counter, key=lambda x: x[1], reverse=True) #[:num_names]
        name = ' '.join(top_ngrams[0][0])
        return name

    # Otherwise, we will use LDA model
    dict1 = corpora.Dictionary(lst)
    doc_term_matrix = [dict1.doc2bow(sentence) for sentence in lst]

    lda_model = LdaModel(doc_term_matrix, num_topics=1, id2word=dict1, passes=10, eta=0.1)
    top_topics = lda_model.show_topics(num_topics=1, num_words=10)

    word_weight_pairs = top_topics[0][1].split(" + ")
    word_weight_dict = {}
    for pair in word_weight_pairs:
        # Split the pair into the word and weight
        word_weight = pair.split("*")
        word = word_weight[1].replace('"', '')
        weight = float(word_weight[0])

        # Add the word and weight to the dictionary for the current topic
        word_weight_dict[word] = weight

    all_ngrams = []

    for sentence in X_cluster:
        words = nltk.word_tokenize(sentence.lower())  # Convert sentence to lowercase and tokenize into words
        for n in range(2, 5):  # Extract n-grams where 2<=n<=4
            all_ngrams.extend(list(ngrams(words, n)))
    ngram_score_dict = {}
    for ngram in all_ngrams:
        ngram_score = 0

        ngram_words = list(ngram)
        for word in ngram_words:
            if word in word_weight_dict.keys():  # make a score for each ngram according to it word
                ngram_score = ngram_score + word_weight_dict[word]  # and the words with the highest probability for this topic of the cluster
        if ngram_score != 0:
            ngram_score_dict[ngram] = ngram_score

    g = max(ngram_score_dict, key=ngram_score_dict.get)
    my_sentence = ' '.join(g)  # choose the ngram with that its word a most associated with the topic of the cluster
    return my_sentence


def create_final_json(X, clusters, unclustered, clusters_representatives_matrix, clusters_names_lst, output_file: str) \
        -> None:
    """
    Creates a new JSON file. Its name is 'output_file', and it's hierarchy is equal to the hierarchy of the ground
    truth JSON file.
    :param X: X of the dataset - all requests we got. NDArray of strings
    :param clusters: A list L of lists l_i. Each l_i contains indexes of strings in X that belongs to cluster i.
    :param unclustered: A list L of all unclustered requests. Each value in the list is an index of unclustered reuqest.
    :param clusters_representatives_matrix: The output of 'get_all_clusters_representatives' function
    :param clusters_names_lst: The output of 'get_clusters_names' function
    :param output_file: A parameter we got from 'analyze_unrecognized_requests' function, for our relative
    full-qualified name, of our JSON output file.
    :return: None.
    """
    print(f"Step 5\t->\tCreating JSON file (full name) {output_file}, then comparing it to the example solution file")
    # Final result is a two-key dictionary
    dict_to_json = {'cluster_list': [], 'unclustered': X[unclustered].tolist()}
    # Then the clustered ones
    for cluster_idx in range(len(clusters)):
        # In each iteration, we create a new dictionary for the current cluster
        cluster_name = clusters_names_lst[cluster_idx]  # cluster name
        cluster_representative_sentences = clusters_representatives_matrix[cluster_idx]  # representative sentences
        cluster_requests = X[clusters[cluster_idx]]  # requests
        curr_dict = {
            'cluster_name': cluster_name,
            'representative_sentences': cluster_representative_sentences.tolist(),
            'requests': cluster_requests.tolist()
        }
        dict_to_json['cluster_list'].append(curr_dict)
    # We now write the dictionary into a new JSON object, then into a new JSON file
    json_output_object = json.dumps(dict_to_json, indent=4)
    with open(output_file, "w") as json_output_file:
        json_output_file.write(json_output_object)


def analyze_unrecognized_requests(data_file: str, output_file: str, num_rep: str, min_size: str) -> None:
    """
    :param data_file: Our dataset
    :param output_file: Our relative full-qualified name, of our upcoming JSON output file.
    :param num_rep: A string of the number of representatives in each cluster.
    :param min_size: A string of the minimal size that each clustered-requests should have.
    :return: None
    """
    X: np.ndarray = extract_data_set(data_file)                                                             # Step 1
    clusters, unclustered = find_clusters(X, min_size)                                                      # Step 2
    clusters_representatives = get_all_clusters_representatives(X, clusters, num_rep)                       # Step 3
    clusters_names_lst = get_clusters_names(X, clusters)                                                    # Step 4
    create_final_json(X, clusters, unclustered, clusters_representatives, clusters_names_lst, output_file)  # Step 5
    print("---")


def my_clusters_representatives_and_labels_accuracy_rate(my_solution: json, output_solution: json) -> None:
    """
    A metric for evaluating parts 2 and 3. We know it is not the most accurate way, but it is good for intuition.
    :param my_solution: Our JSON file we are using to compute the accuracy of our representatives and labels.
    For each representative, we check if it appears in the solution's representatives set (which we create).
    For each label, we check if it appears in the solution's labels set (which we create).
    :param output_solution: The ground truth JSON file, we are using for this type of metric.
    :return: Prints both our representatives and labels accuracies (if they appear in the ground truth).
    """
    representatives_success: int = 0
    representatives_total: int = 0
    labels_success: int = 0
    labels_total: int = 0
    # We create a dictionary of our JSON file
    with open(my_solution) as my_file:
        my_data = json.load(my_file)
        with open(output_solution) as solution_file:
            solution_file = json.load(solution_file)
    # We create a dictionary of the ground truth JSON file
    all_k_representatives_set = set()
    all_labels_set = set()
    # We count 'true' representatives and labels
    for cluster in solution_file['cluster_list']:
        all_k_representatives_set.update(set(cluster['representative_sentences']))
        all_labels_set.add(cluster['cluster_name'])

    for cluster in my_data['cluster_list']:
        for representative in cluster['representative_sentences']:
            if representative in all_k_representatives_set:
                representatives_success = representatives_success + 1
            representatives_total = representatives_total + 1
        if cluster['cluster_name'] in all_labels_set:
            labels_success = labels_success + 1
        labels_total = labels_total + 1
    # We print the percentage
    print(f"Representatives success rate: {str(round((representatives_success / representatives_total * 100), 3))}%")
    print(f"Labels success rate: {str(round((labels_success / labels_total * 100), 3))}%")


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # evaluate_clustering(config['example_solution_file'], config['output_file'])
