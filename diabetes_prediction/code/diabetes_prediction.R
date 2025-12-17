#diabitic predictation
library(ggplot2)
library(reshape2)
library(dplyr)
library(e1071)

#Load the data set
diabetes <- read.csv("C:/Users/HP/Downloads/diabetes.csv")
head(diabetes)
#Here we can looked some values missing in some of col not in outcome so we want replace that zero values 
#Because we cant remove data that can be effect in our output



#########IDENTIFY MISSING VALUES


colSums(is.na(diabetes))# missing not but that representing 0 so
zero_counts <- sapply(diabetes[, c("Glucose","BloodPressure","SkinThickness","Insulin","BMI")],
                      function(x) sum(x == 0))
zero_counts

# Replace impossible zeros with NA
replace_zero_na <- c("Glucose", "BloodPressure", "SkinThickness", 
                     "Insulin", "BMI")

diabetes[replace_zero_na] <- lapply(diabetes[replace_zero_na], function(x) {
  x <- ifelse(x == 0, NA, x)
})

# Check missing values
colSums(is.na(diabetes))

# Impute missing values using median imputation method
for(col in replace_zero_na){
  diabetes[[col]][is.na(diabetes[[col]])] <- median(diabetes[[col]], na.rm = TRUE)
}
# rechecking if any missing values available
colSums(is.na(diabetes))
#now no missig va;ues no check outliers



#######outliers handling


detect_outliers <- function(x){
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  x < (Q1 - 1.5*IQR) | x > (Q3 + 1.5*IQR)
}

sapply(diabetes[, -ncol(diabetes)], function(x) sum(detect_outliers(x)))
#Graphically identified distributions

dia_con<-melt(diabetes[,1:8])
ggplot(dia_con,aes(x=value))+geom_histogram(aes(y=after_stat(density)),fill="skyblue",color="red")+geom_density(color="black",linewidth=1)  +facet_wrap(~variable,scales="free",ncol=3)+theme_minimal()+labs(title="histograms of diabetes data")

# transformation
#log
dia_con_tran<-dia_con %>% mutate(across(where(is.numeric),~log(.+1)))

ggplot(dia_con_tran,aes(x=value))+geom_histogram(aes(y=after_stat(density)),fill="skyblue",color="red")+geom_density(color="black",linewidth=1)  +facet_wrap(~variable,scales="free",ncol=3)+theme_minimal()+labs(title="histograms of diabetes dataafter trasformation")
#comparison before and after
sapply(dia_con,skewness,na.rm=TRUE)
sapply(dia_con_tran,skewness,na.rm=TRUE)
#square root
dia_con_squre<-dia_con%>%mutate(across(where(is.numeric),~sqrt(.+1)))
ggplot(dia_con_squre,aes(x=value))+geom_histogram(aes(y=after_stat(density)),fill="skyblue",color="yellow")+geom_density(color="black",linewidth=1)  +facet_wrap(~variable,scales="free",ncol=3)+theme_minimal()+labs(title="histograms of diabetes dataafter trasformation")
#comparison before and after
sapply(dia_con_squre,skewness,na.rm=TRUE)

#comparison of both method go with square root transformation


dia <- diabetes %>%
  mutate(across(
    .cols = c(Glucose, BloodPressure, SkinThickness, Insulin, BMI,
              Pregnancies, DiabetesPedigreeFunction, Age),
    ~ sqrt(. + 1)
  ))

sum(duplicated(dia))#identified duplicates
str(dia)
summary(dia)
#status(diabetes)#summary with prob

#identifying our independent values relationship for each of them

library(corrplot)

num_data <- dia %>% select(-Outcome)
cor_matrix <- cor(num_data)

corrplot(cor_matrix, method = "color", type = "upper")
cor(dia %>% select(-Outcome))#less 0.7


############### Standardization ###################
#Train–Test Split the data set before standardization
set.seed(123)
train_index <- sample(1:nrow(dia), 0.8 * nrow(dia))

train <- dia[train_index, ]
test  <- dia[-train_index, ]

#Standardization

# Predictor variables (exclude Outcome)
predictors <- setdiff(names(dia), "Outcome")

center_vals <- sapply(train[predictors], mean)
scale_vals  <- sapply(train[predictors], sd)
# Create copies
train_scaled <- train
test_scaled  <- test
# Standardize using TRAIN data
train_scaled[predictors] <- scale(train[predictors],
                                  center = center_vals,
                                  scale  = scale_vals)
# Apply SAME scaling to test data (avoid data leakage)
test_scaled[predictors] <- scale(test[predictors],
                                 center = center_vals,
                                 scale  = scale_vals)
#check its standardization it's works or not
round(colMeans(train_scaled[predictors]), 3)
round(apply(train_scaled[predictors], 2, sd), 3)

######################step 03 ##############
library(caret)
library(randomForest)
library(e1071)
library(pROC)
#logistic reg model
log_model <- glm(Outcome ~ ., 
                 data = train_scaled, 
                 family = binomial)

log_pred_prob <- predict(log_model, test_scaled, type = "response")
log_pred <- ifelse(log_pred_prob > 0.5, 1, 0)
log_pred <- factor(log_pred, levels = c(0,1))
##use Random Forest
#(using non scaled data)
#converting outcome as a factor
# Convert Outcome to factor (for classification)
train$Outcome <- as.factor(train$Outcome)
test$Outcome  <- as.factor(test$Outcome)

set.seed(123)
rf_model <- randomForest(
  Outcome ~ ., 
  data = train,
  ntree = 200,
  importance = TRUE
)

rf_pred <- predict(rf_model, test)
rf_pred_prob <- predict(rf_model, test, type = "prob")[,2]
##Support vector machine-SVM=using standardized data
# Convert Outcome to factor for SVM
train_scaled$Outcome <- factor(train_scaled$Outcome, levels = c(0,1))
test_scaled$Outcome  <- factor(test_scaled$Outcome,  levels = c(0,1))

svm_model <- svm(
  Outcome ~ ., 
  data = train_scaled,
  kernel = "radial",
  probability = TRUE
)

svm_pred <- predict(svm_model, test_scaled)
svm_pred_prob <- attr(
  predict(svm_model, test_scaled, probability = TRUE),
  "probabilities"
)[,1]
head(svm_pred_prob)
range(svm_pred_prob)


###############Model Evaluation
#evaluation function

evaluate_model <- function(true, pred, prob, model_name){
  cat("\n------------------------\n")
  cat(model_name, "\n")
  cat("------------------------\n")
  
  cm <- confusionMatrix(
    as.factor(pred),
    as.factor(true),
    positive = "1"
  )
  print(cm)
  
  roc_obj <- roc(true, prob)
  cat("ROC-AUC:", auc(roc_obj), "\n")
}


############## Step 04 ############
#evaluate all models
evaluate_model(test$Outcome, log_pred, log_pred_prob, "Logistic Regression")
evaluate_model(test$Outcome, rf_pred, rf_pred_prob, "Random Forest")
evaluate_model(test$Outcome, svm_pred, svm_pred_prob, "SVM")

# Ensure factors (VERY IMPORTANT)
log_pred <- factor(log_pred, levels = c(0,1))
rf_pred  <- factor(rf_pred,  levels = c(0,1))
svm_pred <- factor(svm_pred, levels = c(0,1))
test$Outcome <- factor(test$Outcome, levels = c(0,1))

library(caret)
library(pROC)

# -----------------------------
# Confusion Matrices
# -----------------------------
confusionMatrix(log_pred, test$Outcome)
confusionMatrix(rf_pred,  test$Outcome)
confusionMatrix(svm_pred, test$Outcome)

# -----------------------------
# ROC - AUC
# -----------------------------
roc_log <- roc(test$Outcome, log_pred_prob)
auc(roc_log)

roc_rf <- roc(test$Outcome, rf_pred_prob)
auc(roc_rf)

roc_svm <- roc(
  as.numeric(as.character(test_scaled$Outcome)),
  svm_pred_prob
)
auc(roc_svm)


######Step 05 #######

############## Step 05: Interpretation ##############

# Feature Importance - Random Forest
varImpPlot(rf_model, 
           main = "Feature Importance - Random Forest")


# Optional: Save importance values
importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  Importance = importance(rf_model)[, "MeanDecreaseGini"]
)

importance_df <- importance_df[order(-importance_df$Importance), ]
print(importance_df)

summary(log_model)
odds_ratio<-exp(coef(log_model))
odds_ratio
