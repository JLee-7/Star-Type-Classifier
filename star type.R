star_data <- read.csv("star data.csv")
head(star_data)

lapply(star_data, class)
star_data$Star.type <- as.factor(star_data$Star.type)

library(caret)

#Split data
set.seed(42)
splitIndex <- createDataPartition(star_data$Star.type, p = .8, list = FALSE)
train_data <- star_data[splitIndex,]
test_data <- star_data[-splitIndex,]

#install.packages("nnet")
library(nnet)

#multinominal logistic regression model
#lapply(star_data2, class)
#star_data$Star.type <- as.factor(star_data$Star.type)

multinom_fit <- multinom(Star.type ~ Temperature..K. + Luminosity.L.Lo. + Radius.R.Ro. + Absolute.magnitude.Mv., data = train_data)

#SVM model
library(e1071)
svm_fit <- svm(Star.type ~ Temperature..K. + Luminosity.L.Lo. + Radius.R.Ro. + Absolute.magnitude.Mv., data = train_data)
print(svm_fit)

#Ten-fold Cross-validation
train_control <- trainControl(method="cv", number=10)

#Random Forest Model
forest <- train(Star.type ~ Temperature..K. + Luminosity.L.Lo. + Radius.R.Ro. + Absolute.magnitude.Mv.,
                data = train_data,
                method = "rf",
                trControl = train_control,
                metric = "Accuracy")
forest$finalModel

#for SVM
cv_svm <- train(Star.type ~ Temperature..K. + Luminosity.L.Lo. + Radius.R.Ro. + Absolute.magnitude.Mv., data=train_data, method="svmRadial", trControl=train_control)
print(cv_svm)
#for multinominal
set.seed(42)

folds <- createFolds(train_data$Star.type, k = 10)
accuracy <- numeric(length = 10)
fold_counter <- 1

for(fold_index in folds){
  cv_train <- train_data[-fold_index, ]
  cv_validation <- train_data[fold_index, ]

  cv_fit <- multinom(Star.type ~ Temperature..K. + Luminosity.L.Lo. + Radius.R.Ro. + Absolute.magnitude.Mv., data = cv_train)

  predictions <- predict(cv_fit, newdata=cv_validation)

  correct_predictions <- sum(predictions == cv_validation$Star.type)
  accuracy[fold_counter] <- correct_predictions / length(cv_validation$Star.type)

  fold_counter <- fold_counter + 1
}

mean_accuracy <- mean(accuracy)
print(paste("Mean CV Accuracy:", mean_accuracy))



#Test set predictions:
#multinom
multinom_predictions <- predict(multinom_fit, newdata = test_data)
multinom_confMatrix <- table(Predicted = multinom_predictions, Actual = test_data$Star.type)
multinom_accuracy <- sum(diag(multinom_confMatrix)) / sum(multinom_confMatrix)
print(paste("Multinomial Logistic Regression Test Accuracy: ", multinom_accuracy*100, "%", sep = ""))

#svm
svm_predictions <- predict(svm_fit, newdata = test_data)
svm_confMatrix <- table(Predicted = svm_predictions, Actual = test_data$Star.type)
svm_accuracy <- sum(diag(svm_confMatrix)) / sum(svm_confMatrix)
print(paste("SVM Test Accuracy: ", svm_accuracy*100, "%", sep = ""))

#Random Forest
forest_test <- predict(object = forest,
                       newdata = test_data[,-5])
accuracy_forest <- mean(forest_test == test_data$Star.type)*100
print(paste("Random Forest Accuracy: ", accuracy_forest, "%", sep = ""))


##Correlation heatmap -- to reveal if any two features are highly correlated
#this may affect the performance of models
#install.packages("reshape2")
library(reshape2)
# Select numeric columns only for the correlation matrix
numeric_data <- star_data[, sapply(star_data, is.numeric)]
cor_data <- cor(numeric_data)
# Melting the data for ggplot
melted_cor_data <- melt(cor_data)
ggplot(melted_cor_data, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", midpoint = 0) +
  theme_minimal() +
  labs(title = "Correlation Heatmap", x = "", y = "")
#ggsave("heatmap_correlation.png")

library(ggplot2)
library(randomForest)

#Visualization for SVM accuracy
svm_confMatrix <- confusionMatrix(as.factor(svm_predictions), as.factor(test_data$Star.type))
conf_table <- as.data.frame(svm_confMatrix$table)
# Plotting the confusion matrix
ggplot(conf_table, aes(Reference, Prediction, fill = Freq)) +
  geom_tile(color = "white") + 
  geom_text(aes(label = Freq), vjust = 1.5, color = "black") +
  scale_fill_gradient(low = "white", high = "green") +
  labs(title = "Confusion Matrix for SVM Model", x = "Actual Class", y = "Predicted Class") +
  theme_minimal()
#ggsave("confusion_matrix_svm.png", width = 10, height = 8)

#Visualization for multinom accuracy
multinom_confMatrix <- confusionMatrix(as.factor(multinom_predictions), as.factor(test_data$Star.type))
conf_table <- as.data.frame(multinom_confMatrix$table)
# Plotting the confusion matrix
ggplot(conf_table, aes(Reference, Prediction, fill = Freq)) +
  geom_tile(color = "white") + 
  geom_text(aes(label = Freq), vjust = 1.5, color = "black") +
  scale_fill_gradient(low = "white", high = "purple") +
  labs(title = "Confusion Matrix for Multinomial Logistic Regression", x = "Actual Class", y = "Predicted Class") +
  theme_minimal()
#ggsave("confusion_matrix_multinom.png", width = 10, height = 8)

#Random Forest Visualization
library(rpart)
library(rpart.plot)
tr <- rpart(Star.type ~ Temperature..K. + Luminosity.L.Lo. + Radius.R.Ro. + Absolute.magnitude.Mv.,
            data = train_data, cp = 0.00001, minbucket = 20, maxdepth = 7)
options(scipen = 10)
printcp(tr)
pfit <- prune(tr,cp = tr$cptable[which.min(tr$cptable[,"xerror"]),"CP"])
prp(tr)
