import recsys.algorithm
recsys.algorithm.VERBOSE = True

from recsys.algorithm.factorize import SVD
svd = SVD()
filename = './data4'
filename = './data3.csv'
#filename = './data2.csv'
svd.load_data(filename=filename,
        sep=',',
        format={'col':0, 'row':1, 'value':2, 'ids': int})
# col -> user, row -> item, value -> label, ids -> timestamp

k = 100
r = svd.compute(k=k,
            min_values=2,
            pre_normalize=None,
            mean_center=False,
            post_normalize=True,
            savefile='/tmp/movielens')

#ITEMID1 = 109    # Toy Story (1995)
#ITEMID2 = 106 # A bug's life (1998)

#print(svd.similarity(ITEMID1, ITEMID2))
# 0.67706936677315799


item_set = set()
import csv
with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        item_set.add(int(row[1]))

similar_items = {}
for item in item_set:
    # print("similar for item: " + str(item))
    try:
        similar_items[item] = svd.similar(item)
    except Exception as e:
        print(e)

for k, v in similar_items.items():
    if v[1] > 0.1:
        print(k, ["%d: %0.3f" % (id, weight) for (id, weight) in v])

# import pdb;pdb.set_trace()

print(svd.similar(ITEMID1))

# Returns: <ITEMID, Cosine Similarity Value>

MIN_RATING = 0.0
MAX_RATING = 1.0
ITEMID = 109
USERID = 3837663637323963363639393565373833613237396534393132376338386362

print('testing..')
print(svd.predict(ITEMID, USERID, MIN_RATING, MAX_RATING))
# Predicted value 5.0

print(svd.get_matrix().value(ITEMID, USERID))

# Real value 5.0

# Recommend (non-rated) movies to a user:
print('recommend to user')
print(svd.recommend(USERID, is_row=False)) #cols are users and rows are items, thus we set is_row=False

print(svd.recommend(ITEMID))

import pdb;pdb.set_trace()
