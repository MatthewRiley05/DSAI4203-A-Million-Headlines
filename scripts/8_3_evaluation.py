import pickle

with open("outputs/models/semisupervised/self_train_y_lab.pkl", "rb") as f:
    y_all = pickle.load(f)

num_fake = (y_all == 0).sum()
num_real = (y_all == 1).sum()

print("Fake:", num_fake)
print("Real:", num_real)
print("Total:", len(y_all))
