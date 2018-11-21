# http://surprise.readthedocs.io/en/stable/getting_started.html

ratings = pd.read_csv('ratings_small.csv') # reading data in pandas df

from surprise import Reader, Dataset

# to load dataset from pandas df, we need `load_fromm_df` method in surprise lib

ratings_dict = {'itemID': list(ratings.movieId),
                'userID': list(ratings.userId),
                'rating': list(ratings.rating)}
df = pd.DataFrame(ratings_dict)

# A reader is still needed but only the rating_scale param is required.
# The Reader class is used to parse a file containing ratings.
reader = Reader(rating_scale=(0.5, 5.0))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
# Split data into 5 folds

data.split(n_folds=5)

from surprise import SVD, evaluate
from surprise import NMF

# svd
algo = SVD()
evaluate(algo, data, measures=['RMSE'])

# nmf
algo = NMF()
evaluate(algo, data, measures=['RMSE'])
ratings = pd.read_csv('ratings_small.csv') # loading data from csv
"""
ratings_small.csv has 4 columns - userId, movieId, ratings, and timestammp
it is most generic data format for CF related data
"""

val_indx = get_cv_idxs(len(ratings))  # index for validation set
wd = 2e-4 # weight decay
n_factors = 50 # n_factors - dimension of embedding matrix (D)

# data loader
cf = CollabFilterDataset.from_csv(path, 'ratings_small.csv', 'userId', 'movieId', 'rating')

# learner initializes model object
learn = cf.get_learner(n_factors, val_indx, bs=64, opt_fn=optim.Adam)

# fitting model with 1e-2 learning rate, 2 epochs, 
# (1 cycle length and 2 cycle multiple for learning rate scheduling)
learn.fit(1e-2,2, wds = wd, cycle_len=1, cycle_mult=2)
x = ratings.drop([‘rating’],axis=1)
y = ratings[‘rating’].astype(np.float32)
data = ColumnarModelData.from_data_frame(path, val_indx, x, y, [‘userId’, ‘movieId’], 64)

# nh = dimension of hidden linear layer
# p1 = dropout1
# p2 = dropout2

class EmbeddingNet(nn.Module):
    def __init__(self, n_users, _n_movies, nh = 10, p1 = 0.05, p2= 0.5):
        super().__init__()
        (self.u, self.m, self.ub, self.mb) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_movies, n_factors),
            (n_users,1), (n_movies,1)
        ]]
        
        self.lin1 = nn.Linear(n_factors*2, nh)  # bias is True by default
        self.lin2 = nn.Linear(nh, 1)
        self.drop1 = nn.Dropout(p = p1)
        self.drop2 = nn.Dropout(p = p2)
    
    def forward(self, cats, conts): # forward pass i.e.  dot product of vector from movie embedding matrixx
                                    # and vector from user embeddings matrix
        
        # torch.cat : concatenates both embedding matrix to make more columns, same rows i.e. n_factors*2, n : rows
        # u(users) is doing lookup for indexed mentioned in users
        # users has indexes to lookup in embedding matrix. 
        
        users,movies = cats[:,0],cats[:,1]
        u2,m2 = self.u(users) , self.m(movies)
       
        x = self.drop1(torch.cat([u2,m2], 1)) # drop initialized weights
        x = self.drop2(F.relu(self.lin1(x))) # drop 1st linear + nonlinear wt
        r = F.sigmoid(self.lin2(x)) * (max_rating - min_rating) + min_rating               
        return r
        
# n_users: count unique users (671), n_movies: count unique movies (9066)
model = EmbeddingNet(n_users, n_movies)

# model.parameters() for back-propagation of weights 
# lr = 1e-3, weight decay = 1e-5 and using adam optimizer 
opt = optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

# fitting model,
fit(model, data, 3, opt, F.mse_loss)

# learning rate annealing (for more, check links at the end)
set_lrs(opt, 1e-3)
fit(model, data, 3, opt, F.mse_loss)
