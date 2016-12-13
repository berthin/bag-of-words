import bag_of_words

# Let's suppose that you have N images, each image contains 1..M data-points, each data-point has a length of K features
# data_train will contain all this information, so that, it will have a shape of NxMxK (however, the second dimension is not fixed since it can vary, but just suppose that it is constant)

# To build the codebook, in theory, we must use all data-points, so that:
words, codebook = bag_of_words.build_codebook(
  data_train = data_train.reshape(-1, K),
  number_of_words = n_words, #n_words = 300
  clustering_method = 'random',
)

# Once the codebook is built, the bag-of-words histograms can be calculated through:
all_features = np.empty([N, n_words])

for data_image, idx in zip(data_train, xrange(N)):
  all_features[idx, ...] = bag_of_words.coding_pooling_per_video(
    codebook_predictor = codebook,
    number_of_words = n_words,
    data_per_video = data_image.reshape(-1, K),
    type_coding = type_coding, #type_coding = hard
    type_pooling = type_pooling, #type_pooling = sum
  )

# If you want to use multi-processing
from sklearn.externals.joblib import Parallel, delayed

all_features = Parallel(n_jobs = -1) (delayed(bag_of_words.coding_pooling_per_video)(codebook_predictor = codebook, number_of_words = n_words, data_per_video = data_image.reshape(-1, K), type_coding = type_coding, type_pooling = type_pooling,) for data_image in data_training)

all_features = np.array(all_features).reshape(-1, n_words)
