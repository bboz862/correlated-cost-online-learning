I use two different ways to generate the data and save the in two independent folders.
One way is to generate it based on logistic regression, which means I first generate y=ax+b and then transform it to[0,1] based on logistic function , and generate a number uniformly between [0,1] to decide the label

The other way is just find a separating hyperplane, which is ax+b>0, then y=1, otherwise y=0

I use N=10000, dimension=5 and generate data and cost in two files respectively.

In the folder Generate_data_cost_logisticregression
X_lr: is the file of X
y_lr: is the file of y
h_lr: is the file of real hypothesis slope
cost_sg: is the cost correlated with sample group
cost_fts: is the cost correlated with features

The corresponding same files are in the other folder.