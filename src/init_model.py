import numpy as np
text_features16 = np.load("ai-memer_embeddings16.npy")
print(text_features16.shape)
import pickle
annotations = pickle.load(open("ai-memer_annotations.pkl", "rb"))
print(annotations[520000])