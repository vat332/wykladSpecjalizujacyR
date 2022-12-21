# Wczytanie bibliotek
library(caret)

# Wczytanie danych
data <- read.csv("letter-recognition.data", header = FALSE)

# Pierwsze 15 linii
head(data, 15)

# Wymiary i struktura tablicy danych
dim(data)
str(data)

# Część zestawu danych stanowią dane każdej klasy
table(data$V1)

# Standardowe odchylenie kwadratowe, skośność i kurtoza dla każdej cechy liczbowej
sapply(data[, -1], function(x) c(mean = mean(x), sd = sd(x), skewness = skewness(x), kurtosis = kurtosis(x)))

# Cecha decyzyjna i cechy objaśniające
decision_variable <- data$V1
explanatory_variables <- data[, -1]

# Wizualizacja danych

# Histogramy dla cech typu numerycznego
sapply(explanatory_variables, function(x) hist(x))

# Wykresy słupkowe dla typów jakościowych
sapply(explanatory_variables, function(x) barplot(table(x)))

# Diagram analizy korelacji
corrplot(cor(explanatory_variables))

# Wykresy pudełkowe dla atrybutów typu liczbowego
sapply(explanatory_variables, function(x) boxplot(x))

# Wykresy rozrzutu dla cech typu numerycznego w zależności od klas
sapply(explanatory_variables, function(x) {
  plot(x, decision_variable)
})

# Wykresy rozkładu empirycznego (gęstości) dla cech typów liczbowych w zależności od klas
sapply(explanatory_variables, function(x) {
  for (class in unique(decision_variable)) {
    densityplot(x[decision_variable == class])
  }
})

# Podział na zbiór uczący i testowy
set.seed(123)
split <- createDataPartition(decision_variable, p = 0.75, list = FALSE)
training_data <- data[split, ]
test_data <- data[-split, ]

# Przygotowanie danych do modelowania
preprocessing <- preProcess(training_data[, -1])
training_data_preprocessed <- predict(preprocessing, training_data[, -1])
test_data_preprocessed <- predict(preprocessing, test_data[, -1])

# Klasyfikacja za pomocą algorytmów

# Klasyfikator naiwny Bayesa
model_naive_bayes <- train(V1 ~ ., data = training_data_preprocessed, method = "naive_bayes")
predictions_naive_bayes <- predict(model_naive_bayes, newdata = test_data[, -1])
confusion_matrix_naive_bayes <- confusionMatrix(predictions_naive_bayes, test_data$V1)

# K najbliższych sąsiadów
model_knn <- train(V1 ~ ., data = training_data_preprocessed, method = "knn")
predictions_knn <- predict(model_knn, newdata = test_data[, -1])
confusion_matrix_knn <- confusionMatrix(predictions_knn, test_data$V1)

# Maszyny wektorów nośnych
model_svm <- train(V1 ~ ., data = training_data_preprocessed, method = "svmLinear")
predictions_svm <- predict(model_svm, newdata = test_data[, -1])
confusion_matrix_svm <- confusionMatrix(predictions_svm, test_data$V1)

# Drzewa klasyfikacyjne
model_tree <- train(V1 ~ ., data = training_data_preprocessed, method = "rpart")
predictions_tree <- predict(model_tree, newdata = test_data[, -1])
confusion_matrix_tree <- confusionMatrix(predictions_tree, test_data$V1)

# Lasy losowe
model_random_forest <- train(V1 ~ ., data = training_data_preprocessed, method = "rf")
predictions_random_forest <- predict(model_random_forest, newdata = test_data[, -1])
confusion_matrix_random_forest <- confusionMatrix(predictions_random_forest, test_data$V1)

# Oszacowanie i porównanie skuteczności zastosowanych algorytmów na podstawie miar oceny klasyfikacji
accuracy_naive_bayes <- confusion_matrix_naive_bayes$overall[1]
kappa_cohen_naive_bayes <- confusion_matrix_naive_bayes$kappa
accuracy_knn <- confusion_matrix_knn$overall[1]
kappa_cohen_knn <- confusion_matrix_knn$kappa
accuracy_svm <- confusion_matrix_svm$overall[1]
kappa_cohen_svm <- confusion_matrix_svm$kappa
accuracy_tree <- confusion_matrix_tree$overall[1]
kappa_cohen_tree <- confusion_matrix_tree$kappa
accuracy_random_forest <- confusion_matrix_random_forest$overall[1]
kappa_cohen_random_forest <- confusion_matrix_random_forest$kappa

# Porównanie skuteczności algorytmów
#cat("Accuracy (naive bayes):", accuracy_naive_bayes, "\n")
#cat("Kappa Cohena (naive bayes):", kappa_cohen_naive_bayes, "\n")
#cat("Accuracy (KNN):", accuracy_knn, "\n")
#cat("Kappa Cohena (KNN):", kappa_cohen_knn, "\n")
#cat("Accuracy (SVM):", accuracy_svm, "\n")
#cat("Kappa Cohena (SVM):", kappa_cohen_svm, "\n")
#cat("Accuracy (drzewa klasyfikacyjne):", accuracy_tree, "\n")
#cat("Kappa Cohena (drzewa klasyfikacyjne):", kappa_cohen_tree, "\n")
#cat("Accuracy (lasy losowe):", accuracy_random_forest, "\n")
#cat("Kappa Cohena (lasy losowe):", kappa_cohen_random_forest, "\n")



# Wybór najskuteczniejszego algorytmu i zastosowanie go do klasyfikacji zbioru testowego
# (tutaj przyjmujemy, że najskuteczniejszym algorytmem jest algorytm lasu losowego)
final_model <- model_random_forest
final_predictions <- predictions_random_forest
final_confusion_matrix <- confusion_matrix_random_forest

# Obliczenie macierzy błędów oraz podstawowych miar jakości
print(final_confusion_matrix$table)
print(final_confusion_matrix$overall)
