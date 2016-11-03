
####################################################################
# Install R library for decision tree
####################################################################
#library(ISLR)
#library(tree)
#attach(Carseats)
library(rpart)
#library(rpart.plot)
#data = read.csv("C:/Users/Merlin Mpoudeu/SkyDrive/Documents/CSCE_ML/dataset/wine.csv")

#data = read.csv("C:/Users/Merlin Mpoudeu/SkyDrive/Documents/CSCE_ML/dataset/MONKS1_TRAIN.csv")
#data = read.csv("C:/Users/Merlin Mpoudeu/SkyDrive/Documents/CSCE_ML/dataset/VoteData.csv")
data = read.csv("C:/Users/Merlin Mpoudeu/SkyDrive/Documents/CSCE_ML/dataset/Car_data.csv")
#data = read.csv(file = "/work/statsgeneral/vcdim/Code/Car_data.csv", header = T)

data = data[sample(1:nrow(data), size = nrow(data), replace = FALSE),]

for (i in names(data)[1:(ncol(data)-1)]) {
  data[,i] = as.factor(data[,i])
}
set.seed(124)
#data = iris
# Randomization and divide the the data into training and testing set
Index = sample(1:nrow(data), size = nrow(data)/2, replace = FALSE)
test_data = data[Index,-ncol(data)]
y_test = data[Index,ncol(data)]
y_train = data[-Index,ncol(data)]
train_data = data[-Index,-ncol(data)]
Maxcounter = 500
#######################################################################
# Tree data
######################################################################
test_data_tree = data[Index,]
train_data_tree = data[-Index,]


# The activation function
sigmoid = function(z){
  1/(1+exp(-z))
}
# The derivative of the activation function
derivsigmoid = function(z){
  sigmoid(z)*(1-sigmoid(z))
}
# The foraward part of ANN
Forward = function(W,B,V,x){
  TT = W%*%t(x)
  Z = sigmoid(TT)
  ZZ = rbind(1,Z)
  G = B%*%ZZ
  L = sigmoid(G)
  LL = rbind(1,L)
  FF = V%*%LL
  Y = sigmoid(FF)
  return(list(Y = Y,FK = FF,L = LL,Z = ZZ, G =G, TT = TT))
}
output = function(Y){
  out = numeric(nrow(Y))
  out[which.max(Y)] = 1
  return(out)
}
coltype = apply(train_data, 2, FUN = is.numeric)

XMat = function(train_data, rows, coltype){
  x = c(1)
  for (dd in 1:ncol(train_data)) {
    if(coltype[dd]== "FALSE"){
      aa = grep(train_data[rows,dd], levels(train_data[,dd]))
      vect = numeric(nlevels(train_data[,dd]))
      vect[aa] = 1
      x = c(x, vect)
    }else{
      x = c(x,train_data[rows,dd])
    }
  }
  
  x = matrix(x, nrow = 1, byrow = TRUE)
  
}
# create a vector of target attribute
target = function(y_train, rows){
  if(length(unique(y_train)) == 2){
    r = y_train[rows]
  }else{
    r = numeric(nlevels(as.factor(y_train)))
    Index = grep(y_train[rows], levels(y_train))
    r[Index] = 1
  }
  
}
#####################################################################
# Accuracy function
#####################################################################
Accuracy = function(W,B,V, test_data, y_test, cutoff=0.5)
{
  coltype = apply(test_data, 2, FUN = is.numeric)
  Prediction = numeric(nrow(test_data))
  for (rows in 1:nrow(test_data)) {
    x = XMat(test_data, rows, coltype)
    Prod = Forward(W,B,V,x)
    out = Prod$Y
    if(length(unique(y_test))>2){
      Prediction[rows] = levels(as.factor(y_test))[which.max(out)]
    }else{
      if(out>cutoff){
        Prediction[rows] = 1
      }else{
        Prediction[rows] = 0
      }
    }
  }

  confusionMatrix = table(y_test, Prediction)
  accuracy = mean(Prediction == y_test)
  return(list(accuracy=accuracy,confusionMatrix=confusionMatrix, Prediction=Prediction))
}
# first derivative
M = 10 # number of hidden unit in the first layer
H = 10 # number of hidden unit in the second layer
# Back propagation
TwoLayers_ANN = function(train_data, y_train, test_data, y_test,Maxcounter, cutacc = 0.8,eta=0.03, M=20, H=20)
{
  coltype = apply(train_data, MARGIN = 2, FUN = is.numeric)
  d = 0 # is the dimension of our dataset
  for (p in 1:ncol(train_data)) {
    if(coltype[p] == FALSE){
      d = d + nlevels(train_data[,p])
    }else{
      d = d + 1
    }
  }
  if(length(unique(y_train)) == 2){
    k = 1  # the number of unit in the output layer
  }else{
    k = length(unique(y_train))
  }
  # Generate Initial
  set.seed(M*(d+1+120156))
  W = matrix(runif(M*(d+1), min = -0.05, max = 0.05), nrow = M, ncol = (d+1))
  B = matrix(runif(H*(M+1), min = -0.05, max = 0.05), nrow = H, ncol = (M+1))
  V = matrix(runif(k*(H+1), min = -0.05, max = 0.05), nrow = k, ncol = (H+1))
  #####################################################################
  coltype = apply(train_data, 2, FUN = is.numeric)
  counter = 0
  Error_mat = numeric(Maxcounter)
  Acc_Inedx = sample(nrow(train_data), size = 20, replace = FALSE)
  train_acc = train_data[Acc_Inedx,]
  y_acc = y_train[Acc_Inedx]
  acc = 0
  while (counter <= Maxcounter && acc < cutacc) {
    Error_row = 0
    for (rows in 1:nrow(train_data)) {
      x = XMat(train_data, rows, coltype)
      Prod = Forward(W, B, V, x)
      FK = Prod$FK
      output = Prod$Y
      L = Prod$L
      error = target(y_train, rows) - output
      #Error_row = sum(target(y_train, rows)*log(output)) + Error_row
      Error_row = 0.5*(sum(error^2)) + Error_row
      deltaK = error*derivsigmoid(FK) # derivative at the output node
      DeltaVki = matrix(NA, nrow = k, ncol = (H+1))
      for (kk in 1:k) {
        for (ii in 1:(H+1)) {
          DeltaVki[kk,ii] = eta*deltaK[kk,1]*L[ii,]
        }
      }
      #####################################################################
      # Derivative at the second hidden Layer
      ####################################################################
      G = Prod$G
      deltaH = matrix(NA, nrow = H, ncol = 1)
      VV = matrix(V[,2:ncol(V)], ncol = (ncol(V)-1), nrow = nrow(V))
      for (hh in 1:H) {
        sum = 0
        for (kk in 1:k) {
          sum = sum + deltaK[kk,]*VV[kk,hh]
        }
        deltaH[hh,] = sum*derivsigmoid(G[hh,])
      }
      DeltaBhi = matrix(NA, nrow = H, ncol = (M+1))
      Z = Prod$Z
      for (HH in 1:H) {
        for (II in 1:(M+1)) {
          DeltaBhi[HH,II] = eta*deltaH[HH,]*Z[II,]
        }
      }
      #####################################################################
      # Derivative of the first Hidden Layer
      ####################################################################
      deltaM = matrix(NA, nrow = M, ncol = 1)
      TT = Prod$TT
      BB = B[,2:ncol(B)]
      for (mm in 1:M) {
        sum = 0
        for (nn in 1:H) {
          sum = sum + deltaH[nn,]*BB[nn,mm]
        }
        deltaM[mm,] = sum*derivsigmoid(TT[mm,])
      }
      DeltaWMI = matrix(NA, nrow = M, ncol = (d+1))
      for (tt in 1:M) {
        for (gg in 1:(d+1)) {
          DeltaWMI[tt,gg] = eta*deltaM[tt,]*x[1,gg]
        }
      }
      ####################################################################
      # update W
      ###################################################################
      for (w in 1:nrow(W)) {
        for (oo in 1:ncol(W)) {
          W[w,oo] = W[w,oo] + DeltaWMI[w,oo]
        }
      }
      #cat("The Updated w is :")
      #show(W)
      ####################################################################
      # update B
      ##################################################################
      for (br in 1:nrow(B)) {
        for (bc in 1:ncol(B)) {
          B[br,bc] = B[br,bc] + DeltaBhi[br,bc]
        }
      }
      #cat("The Updated B is :")
      #show(B)
      #####################################################################
      # Update V
      ####################################################################
      for (vr in 1:nrow(V)) {
        for (vc in 1:ncol(V)) {
          V[vr,vc] = V[vr,vc] + DeltaVki[vr,vc]
        }
      }
      #cat("The Updated V is :", "\n")
      #show(V)
      
    }
    acc = Accuracy(W, B , V , test_data, y_test,cutoff=0.5)$accuracy
    #Average_Error = -Error_row
    
    Average_Error = Error_row/nrow(train_data)
    cat("The accuracy on the validation set is : ", acc, " at the ", counter, " run", "\n")
    counter = counter + 1
    Error_mat[counter] = Average_Error
  }
  
  return(list(W=W,B=B,V=V, ErrorMat = Error_mat))
  
}
nn = TwoLayers_ANN(train_data, y_train, test_data, y_test,Maxcounter=100, cutacc = 0.8, eta=0.1, M=20, H=20)
test = Accuracy(W = nn$W, B = nn$B, V = nn$V, test_data, y_test,cutoff=0.5)
test$accuracy
test$confusionMatrix
test$Prediction
###################################################################
library(rpart)
#Randomly shuffle the data
data<-data[sample(nrow(data)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(data)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
ErrorMat = matrix(NA, ncol = 2, nrow = 10)
for(i in 1:10){
  
  #Segment your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  Mytestdata = data[testIndexes,-ncol(data)]
  ytest = data[testIndexes,ncol(data)]
  ytrain = data[-testIndexes,ncol(data)]
  data_treeTrain = data[-testIndexes,]
  data_treeTest = data[testIndexes,]
  y_treetest = data[testIndexes,ncol(data)]
  Mytrainingdata = data[-testIndexes,-ncol(data)]
  tree_model = rpart(as.formula(paste(names(data)[ncol(data)],"~.", sep = "")), data = data_treeTrain, method = "class")
  prune_model = prune(tree_model, cp = tree_model$cptable[which.min(tree_model$cptable[,"xerror"]),"CP"])
  p = predict(prune_model, data_treeTest, type = "class")
  error2 = mean(p != y_treetest)
  out1 =  TwoLayers_ANN(Mytrainingdata, ytrain, Mytestdata, ytest,Maxcounter=50, cutacc = 0.8, eta=0.1, M=20, H=20)
  
  test = Accuracy(W = out1$W, B = out1$B, V = out1$V, Mytestdata, ytest,cutoff=0.5)
  test$accuracy
  error1 = 1 - test$accuracy
  error_diff1 = error1 - error2
  error_diff2 = error2 - error1
  ErrorMat[i,1] = error_diff1
  ErrorMat[i,2] = error_diff2
  #Use the test and train data partitions however you desire...
}
pi = apply(ErrorMat, 2, FUN = mean, na.rm = TRUE)
sump1 = 0
sump2 = 0
for (q in 1:nrow(ErrorMat)) {
  sump1 = sump1 + (ErrorMat[q,1] - pi[1])^2
  sump2 = sump2 + (ErrorMat[q,2] - pi[2])^2
}
sp1 = sqrt((1/(10*9))*sump1)
sp2 = sqrt((1/(10*9))*sump2)
upperBound1 = pi[1] + qt((1-0.05), 9)*sp1
upperBound2 = pi[2] + qt((1-0.05), 9)*sp2
message("The CI for p1 - p2 = 0 is ", "(-infinity ","," , upperBound1, "]" )
message("The CI for p2 - p1 = 0 is ", "(-infinity","," ,upperBound2, "]" )
plot(ErrorMat[,1], type = "l", ylab="Error Difference", xlab = "Number of Cross Validation", col = 1, lty = 1, main= "ANN vs ID3")
points(ErrorMat[,2], type = "l", ylab="Error Difference", xlab = "Number of Cross Validation", col = 2, lty = 2)
aa = c("ANN_VS_ID3", "ID3_VS_ANN")
cyl.f <- factor(as.factor(folds), levels= c(1,0),
                labels = c("1 ANN_VS_ID3", "0 ID3_VS_ANN")) 

colfill<-c(2:(2+length(levels(cyl.f)))) 

legend(locator(1), levels(cyl.f), fill=colfill)
#####################################################################

#####################################################################
# ROC CURVE
####################################################################
coltype = apply(train_data, 2, FUN = is.numeric)
W = nn$W
B = nn$B
V = nn$V
y_hat = numeric(nrow(test_data))
for (rows in 1:nrow(test_data)) {
  x = XMat(test_data, rows, coltype)
  Prod = Forward(W,B,V,x)
  out = Prod$Y
  y_hat[rows] = out
 
}
dat = data.frame(y_test, y_hat)
Prediction = numeric(nrow(test_data))
dat = dat[order(y_hat, decreasing = TRUE),]
library(sm)
cyl.f <- factor(as.factor(y_test), levels= c(1,0),
                labels = c("1 Vote", "0 Vote")) 

sm.density.compare(y_hat,as.factor(y_test), xlab="Probability")
colfill<-c(2:(2+length(levels(cyl.f)))) 

legend(locator(1), levels(cyl.f), fill=colfill)

ROCMAT = matrix(NA, nrow = length(y_hat), ncol = 2)

colnames(ROCMAT) = c("sensitivity", "specificity")

for (f in 2:(length(y_hat)-1)) {
  cutoff = dat[f,"y_hat"]
  for (j in 1:length(y_hat)) {
    if(dat[j,"y_hat"]> cutoff){
      Prediction[j] = 1
    }else{
      Prediction[j] = 0
    }
  }
  ConfMat = table( y_test =dat[,"y_test"],Prediction)
  tp = ConfMat[2,2]
  fn = ConfMat[2,1]
  tn = ConfMat[1,1]
  fp = ConfMat[1,2]
  sensitivity = tp/(tp+fn)
  specificity= tn/(tn+fp)
  ROCMAT[f,"sensitivity"] = sensitivity
  ROCMAT[f,"specificity"] = 1 - specificity
}
ROCMAT = ROCMAT[-c(1,length(y_hat)),]
ANN.scores <- prediction(dat$y_hat, dat$y_test)
ANN.perf <- performance(ANN.scores, "tpr", "fpr")
#plot(ANN.perf, col = "green", lwd = 1.5, lty = 2, pch = 3, main = "")
# AUC for the ANN
ANN.auc <- performance(ANN.scores, "auc")
ANN.auc


#############################################################################
# ROC CURVE for ID3
########################################################################
library(rpart)
tree_model = rpart(as.formula(paste(names(train_data_tree)[ncol(train_data_tree)],"~.", sep = "")), data = train_data_tree, method = "class")
#plot(tree_model)
#text(tree_model, pretty = 0)


#We can find this value as 
#printcp(tree_model)
# Here we see that the value is 0.013514.

#We can write a small script to find this value
tree_model.cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
tree_model.prune <- prune(tree_model,tree_model.cp)

#We need to score the pruned tree model the same way we did for the Logistic model.
test_data_tree$t1.yhat <- predict(tree_model.prune, test_data_tree, type = "prob")

#We will also plot the ROC curve for this tree.
library(ROCR)
t1.scores <- prediction(test_data_tree$t1.yhat[,2], test_data_tree[,ncol(data)])
t1.perf <- performance(t1.scores, "tpr", "fpr")
plot(t1.perf, col = "green", lwd = 1.5, lty = 2, pch = 3, main = "")
# AUC for the decision tree
t1.auc <- performance(t1.scores, "auc")
t1.auc
points(ROCMAT[,"sensitivity"] ~ ROCMAT[,"specificity"], 
       type = "l", col = "red", lty = 1, pch = 4, lwd = 2,
       xlab = "1 - specificity", ylab = "sensitivity")
legend(0.6, 0.8, c("ID3", "ANN"), col = c("green", "red"),
       text.col = "green4", lty = c(2,  1), pch = c(3, 4),lwd = c(1.5,2),
       merge = TRUE, bg = "gray90")
title(main = paste("ROC Curve Of ID3 VS ANN for Vote'S data", "\n"," AUC_ID3 = " ,0.989 , "\n"," AUC_ANN = ",0.988 ))
