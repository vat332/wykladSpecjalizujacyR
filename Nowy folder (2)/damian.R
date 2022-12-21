# wczytaj bibliotekę caret i dane wine.data
library(caret)
data("wine")

# wyświetl pierwszych 15 linii
head(wine, 15)

# wyznacz wymiary i strukturę tablicy danych
dim(wine)
str(wine)

# wyznacz jaką część zestawu danych stanowią dane każdej klasy
table(wine$class)

# wyznacz standardowe odchylenie kwadratowe, skośność i kurtozę dla każdej cechy liczbowej
sapply(wine[, 1:13], sd)
sapply(wine[, 1:13], skewness)
sapply(wine[, 1:13], kurtosis)

# wyznacz cechę decyzyjną i cechy objaśniające
class = wine$class
explanatory = wine[, 1:13]

# stwórz histogramy dla cech typu numerycznego
par(mfrow=c(3,5))
for (i in 1:13) {
  hist(explanatory[,i])
}

# stwórz wykresy słupkowe dla typów jakościowych
barplot(table(wine$class))

# zbuduj diagram analizy korelacji
cor(explanatory)

# skonstruuj wykresy pudełkowe dla atrybutów typu liczbowego
boxplot(explanatory)

# skonstruuj wykresy rozrzutu dla cech typu numerycznego w zależności od klas
par(mfrow=c(3,5))
for (i in 1:13) {
  plot(explanatory[,i] ~ class, data=wine)
}

# sporządź wykresy rozkładu empirycznego (gęstości) dla cech typów liczbowych w zależności od klas
par(mfrow=c(3,5))
for (i in 1:13) {
  plot(density(explanatory[class=="1",i]), col="red", lwd=2)
  lines(density(explanatory[class=="2",i]), col="blue", lwd=2)
  lines(density(explanatory[class=="3",i]), col="green", lwd=2)
}

# podziel zbiór danych na uczący i testowy (75% w zbiorze uczącym)
set.seed(123)

#podziel zbiór danych na uczący i testowy (75% w zbiorze uczącym)
set.seed(123)
trainIndex <- createDataPartition(class, p = 0.75, list = FALSE)
trainData <- wine[trainIndex, ]
testData <- wine[-trainIndex, ]

#stwórz obiekt szablonu z zastosowaniem algorytmu naiwnego Bayesa
naiveBayesTemplate <- train(class ~ ., data = trainData, method = "nb")

#stwórz obiekt szablonu z zastosowaniem algorytmu K najbliższych sąsiadów
knnTemplate <- train(class ~ ., data = trainData, method = "knn")

#stwórz obiekt szablonu z zastosowaniem algorytmu maszyny wektorów nośnych
svmTemplate <- train(class ~ ., data = trainData, method = "svmLinear")

#stwórz obiekt szablonu z zastosowaniem algorytmu drzewa klasyfikacyjnego
decisionTreeTemplate <- train(class ~ ., data = trainData, method = "rpart")

#stwórz obiekt szablonu z zastosowaniem algorytmu lasu losowego
randomForestTemplate <- train(class ~ ., data = trainData, method = "rf")

#oszacuj i porównaj skuteczność zastosowanych algorytmów na podstawie miar oceny klasyfikacji
predNaiveBayes <- predict(naiveBayesTemplate, newdata = testData)
confusionMatrix(predNaiveBayes, testData$class)
postResample(predNaiveBayes, testData$class)

predKnn <- predict(knnTemplate, newdata = testData)
confusionMatrix(predKnn, testData$class)
postResample(predKnn, testData$class)

predSvm <- predict(svmTemplate, newdata = testData)
confusionMatrix(predSvm, testData$class)
postResample(predSvm, testData$class)

predDecisionTree <- predict(decisionTreeTemplate, newdata = testData)
confusionMatrix(predDecisionTree, testData$class)
postResample(predDecisionTree, testData$class)

predRandomForest <- predict(randomForestTemplate, newdata = testData)
confusionMatrix(predRandomForest, testData$class)
postResample(predRandomForest, testData$class)

#wybierz najskuteczniejszy algorytm i zastosuj go do klasyfikacji zbioru testowego
bestModel <- randomForestTemplate
predBestModel <- predict(bestModel, newdata = testData)


#oblicz macierz błędów oraz podstawowe miary jakości
confusionMatrix(predBestModel, testData$class)
postResample(predBestModel, testData$class)