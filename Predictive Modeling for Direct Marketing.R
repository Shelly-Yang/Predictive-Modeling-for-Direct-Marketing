# September 2019
# title: "Predictive Modeling for Direct Marketing - Bank Case"
# author: "Tong Niu, Zhaohui Li, Yunqing Yu, Qiqi Liu, Shelly Yang"
# instructor: |
#   | Rajiv Dewan
# | University of Rochester


# predicted profit = 130 * (1 - FNR) * 40 - 870 * FPR * 10 = 5200 - 5200 * FNR(beta) - 8700 * FPR(alpha)

library(readr)
bankData <- read_csv("~/Desktop/BA/r homework/bank.csv")
bankData <- data.frame(bankData)  # convert to a dataframe 
bankData$job <- as.factor(bankData$job) 
bankData$marital <- as.factor(bankData$marital) 
bankData$education <- as.factor(bankData$education)
bankData$default <- as.factor(bankData$default)
bankData$housing <- as.factor(bankData$housing)
bankData$loan <- as.factor(bankData$loan) 
bankData$contact <- as.factor(bankData$contact)
#bankData$month <- as.factor(bankData$month) 
bankData$campaign <- as.factor(bankData$campaign)
bankData$poutcome <- as.factor(bankData$poutcome)
bankData$y <- as.factor(bankData$y)

summary(bankData)

set.seed(644)

train = sample(1:nrow(bankData),nrow(bankData)*0.666667)

b.train = bankData[train,]   # 6,666 rows
b.test = bankData[-train,]   # the other 3,334 rows

save(bankData, b.train, b.test, file='~/Desktop/BA/r homework/bankData.Rda')
write.csv(b.train, file='~/Desktop/BA/r homework/b.train.csv') 

prop.table(table(b.train$y))

library(rpart)

# grow the biggest tree
# xval = 10 : cross-sample with 10 folds to determine error rate at each node
# minsplit = 10  : min number of observations to attempt split
# cp = 0  : minimum improvement in complexity parameter for splitting
fit = rpart(y ~ ., # formula
            data=b.train, # dataframe used
            control=rpart.control(xval=10, minsplit=10, cp = 0.0))
nrow(fit$frame)
# the biggest tree have 193 nodes

plotcp(fit, # tree for which to plot
       upper="size")

# find the CP which provides the lowest error
bestcp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
bestcp
# It appears that lowest error occurs at CP = 0.01186944

# POST-PRUNING
fit.post = prune.rpart(fit, cp=bestcp)
nrow(fit.post$frame)
# the pruned tree have 17 nodes

plot(fit.post, uniform=T, branch=0.5, compress=T,
     main="Tree with Post-Pruning with best cp (17 Nodes)", margin=0.05)
text(fit.post,  splits=T, all=F, use.n=T, 
     pretty=T, fancy=F, cex=1.2) 

# compute the confusion matrices
# resubstitution
confusionMatrix(table(predict(fit.post, b.train, type="class"),
                      b.train$y), positive='yes')
# test set
cm = confusionMatrix(table(predict(fit.post, b.test, type="class"),
                           b.test$y), positive='yes') 
cm 
# let us look up values in the cm object
# alpha = 0.02721088
(1-cm$byClass["Specificity"][[1]])
# beta = 0.6467391
(1-cm$byClass["Sensitivity"][[1]])

# profit = 1600.222
5200 - 5200 * (1-cm$byClass["Sensitivity"][[1]]) - 
  8700 * (1-cm$byClass["Specificity"][[1]])

########################################################################################################################
######################################### CREATE A BALANCED DATASET ####################################################
########################################################################################################################

# splitting the training dataset, into two new data frames: b.train.yes and b.train.no
b.train.yes = subset(b.train, y == 'yes')
b.train.no = subset(b.train, y == 'no')

# take a sub-sample from b.train.no with the same number of observations as b.train.yes. 
set.seed(234)

no = sample(1:nrow(b.train.no),nrow(b.train.yes))
b.train.no = b.train.no[no,]

# Combine b.train.yes and the sub sample of b.train.no to make a new balanced training data frame called b.bal. 
b.bal =  rbind(b.train.yes, b.train.no)

########################################################################################################################
##################################### NOW USE BALANCED DATA TO BUILD TREE ###########################################
########################################################################################################################

# split the balanced data into training and test data sets

save(b.bal, file='~/Desktop/BA/r homework/b.bal.Rda') 
write.csv(b.bal, file='~/Desktop/BA/r homework/b.bal.csv') 

prop.table(table(b.bal$y))

library(rpart)

# grow the biggest tree for the balanced training data
# xval = 10 : cross-sample with 10 folds to determine error rate at each node
# minsplit = 10  : min number of observations to attempt split
# cp = 0  : minimum improvement in complexity parameter for splitting
bal.fit = rpart(y ~ ., # formula
            data=b.bal, # dataframe used
            control=rpart.control(xval=10, minsplit=10, cp = 0.0))
nrow(bal.fit$frame)
# the biggest tree have 95 nodes

plotcp(bal.fit, # tree for which to plot
       upper="size")

#find the CP which provides the lowest error
bal.bestcp = bal.fit$cptable[which.min(bal.fit$cptable[,"xerror"]),"CP"]
bal.bestcp
# It appears that lowest error occurs at CP = 0.007418398

# POST-PRUNING
bal.fit.post = prune.rpart(bal.fit, cp=bal.bestcp)
nrow(bal.fit.post$frame)
# the pruned tree have 19 nodes

plot(bal.fit.post, uniform=T, branch=0.5, compress=T,
     main="Tree with Post-Pruning with best cp (19 Nodes)", margin=0.05)
text(bal.fit.post,  splits=T, all=F, use.n=T, 
     pretty=T, fancy=F, cex=1.2)

# compute the confusion matrices
# resubstitution
confusionMatrix(table(predict(bal.fit.post, b.bal, type="class"),
                      b.bal$y), positive='yes')
# test set
bal.cm = confusionMatrix(table(predict(bal.fit.post, b.test, type="class"),
                           b.test$y), positive='yes')
bal.cm
# let us look up values in the bal.cm object
# alpha = 0.25548
(1-bal.cm$byClass["Specificity"][[1]])
# beta = 0.1521739
(1-bal.cm$byClass["Sensitivity"][[1]])

# profit = 2186.02
5200 - 5200 * (1-bal.cm$byClass["Sensitivity"][[1]]) - 
  8700 * (1-bal.cm$byClass["Specificity"][[1]])


########################################################################################################################
##################################### NOW WE DETERMINE OPTIMAL CUTOFF ##################################################
########################################################################################################################

library(ROCR)
##  Start by predicting prob instead of class
y.pred = as.data.frame(predict(bal.fit.post, b.bal, type="prob"))
head(y.pred)
# the pos column of churn.pred has the score = prob of pos
# now we will use the ROCR package
# first step is compute the score object
y.pred.score = # first of two steps - compute the score
  prediction(y.pred[,2],  # the predicted P[Yes]
             b.bal$y) # the actual class

# # next step is to compute the performance object for the curve we want
# y.pred.perf = performance(y.pred.score, "tpr", "fpr")
# 
# plot(y.pred.perf, 
#      colorize=T, # colorize to show cutoff values
#      lwd=4) # make the line 4 times thicker than default
# abline(0,1)  # draw a line with intercept=0, slope = 1
# abline(h=1) #draw a horizontal line at y = 1
# abline(v=0) #draw a vertical line at x = 0
# # more plot options: 
# # http://www.statmethods.net/advgraphs/parameters.html
# 
# # General evaluation of ROC of classifiers
# # area under the curve (AUC)
# # note that the performance object is an S4 object
# # it has slots not columns.  Pick slots using the @ operator rather than $
# performance(y.pred.score, "auc")@y.values  # 0.8617625
# 
# # now for cost tradeoff
# # let us first determine the profit using the test data
# # first let us look at the profit from default cutoff of 50%
# cm = confusionMatrix(table(pred=predict(bal.fit.post, b.test, type="class"),
#                            actual = b.test$y), positive='yes')
# cm
# # let us look up values in the cm object
# #alpha =  0.25548
# (1-cm$byClass["Specificity"][[1]])
# #beta = 0.1521739
# (1-cm$byClass["Sensitivity"][[1]])
# 
# # profit = 2186.02
# 5200 - 5200 * (1-cm$byClass["Sensitivity"][[1]]) - 
#   8700 * (1-cm$byClass["Specificity"][[1]])

# now we will determine the optimal cutoff
# rather than use the ROCR curve directly, we will use the cost built into ROCR
# the expected profit function from clas: 5200 - 5200 * beta - 8700 * alpha
# let us start with the default prediction (50% cutoff)
# NOTE: We are still using the trainig data
#   as picking the cutoff is part of model building
y.cost = performance(y.pred.score, measure="cost", 
                         cost.fn=5200, cost.fp=8700) 
plot(y.cost)
#seems to be minimized around 0.7
# we can find this more precisely
cutoff.best = y.cost@x.values[[1]][which.min(y.cost@y.values[[1]])]
cutoff.best  # 0.7142857

# meaning if Prob[yes] <= custoff.best assign to NO else YES
# Let us make predictions using this cutoff rate for the test set
# create a vector of predictions based on optimal cutoff value
y.pred.test = predict(bal.fit.post, b.test, type="prob")
head(y.pred.test)

y.pred.test.cutoff = 
  ifelse(y.pred.test[,2] > cutoff.best,'yes','no') 

# now let us find the profit for the test data set using the optimal cutoff
#make the confusion matrix using table()
cm = confusionMatrix(table(pred=y.pred.test.cutoff,
                           actual = b.test$y), positive='yes')
cm

#alpha = 0.2403628
(1-cm$byClass["Specificity"][[1]])
#beta = 0.1630435
(1-cm$byClass["Sensitivity"][[1]])

# profit =  2261.017
5200 - 5200 * (1-cm$byClass["Sensitivity"][[1]]) - 
  8700 * (1-cm$byClass["Specificity"][[1]])

# How do the models built using b.test, b.bal with default cutoff, and b.bal with cost minimizing cutoff
# compare? 
# with cost minimizing cutoff, we lower the FPR and increase the FNR, cost of FPR is much higher than that of FNR,
# predicted profit changed from 2186.02 to 2261.017.

