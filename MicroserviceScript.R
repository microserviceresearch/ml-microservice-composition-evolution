##Load the three packages required to train the classifiers

library(caret)
library(e1071)
library(randomForest)

##Load the microservice Train dataset and the Test dataset

microserviciosTrain <- read.csv("TrainDatasetC50Rpart.csv")

microserviciosTest <- read.csv("TestDatasetC50Rpart.csv")

##Load the microservice Train dataset and the Test dataset for the Random Forest technique

microserviciosTrain2 <- read.csv("TrainDatasetRandomForest.csv")

microserviciosTest2 <- read.csv("TestDatasetRandomForest.csv")

##Configure the parameters for the cross-validation technique

fitControl <- trainControl(method = "cv", number = 10)
set.seed(1001)

##Train phase

##Decision tree C5.0

DTFit <- train(Adaptation.rule ~., data = microserviciosTrain, method = "C5.0", trControl = fitControl, winnow = TRUE)
DTFit

##Decision tree RPART

DTFit2 <- train(Adaptation.rule ~., data = microserviciosTrain, method = "rpart", trControl = fitControl)
DTFit2

##Decision tree Random Forest (ntree = 5, 10, 15)

DTFit3 <- randomForest(x = microserviciosTrain2[,-11], y=microserviciosTrain2$Adaptation.rule, ntree=5)
DTFit3

DTFit4 <- randomForest(x = microserviciosTrain2[,-11], y=microserviciosTrain2$Adaptation.rule, ntree=10)
DTFit4

DTFit5 <- randomForest(x = microserviciosTrain2[,-11], y=microserviciosTrain2$Adaptation.rule, ntree=15)
DTFit5

##Test phase

##Test the Decision Tree C5.0

microserviciosPredictionC50 <- predict(DTFit, microserviciosTest)
microserviciosPredictionC50 <- as.character(microserviciosPredictionC50)

u <- union(microserviciosPredictionC50, microserviciosTest$Adaptation.rule)
t <- table(factor(microserviciosPredictionC50, u), factor(microserviciosTest$Adaptation.rule, u))
confusionMatrix(t)

##Test the Decision Tree RPART

microserviciosPredictionRPART <- predict(DTFit2, microserviciosTest)
microserviciosPredictionRPART <- as.character(microserviciosPredictionRPART)

u <- union(microserviciosPredictionRPART, microserviciosTest$Adaptation.rule)
t <- table(factor(microserviciosPredictionRPART, u), factor(microserviciosTest$Adaptation.rule, u))
confusionMatrix(t)

##Test the Decision Tree Random Forest

microserviciosPredictionRF5 <- predict(DTFit3, microserviciosTest)
microserviciosPredictionRF5 <- round(microserviciosPredictionRF5)

u <- union(microserviciosPredictionRF5, microserviciosTest2$Adaptation.rule)
t <- table(factor(microserviciosPredictionRF5, u), factor(microserviciosTest2$Adaptation.rule, u))
confusionMatrix(t)
