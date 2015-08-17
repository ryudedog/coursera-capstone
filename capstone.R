# capstone project

library(tm)
library(ngram)
library(wordcloud)
library(slam)
library(SnowballC)

#
# Functions
#

# Clean the input string
cleanStr <- function(s) {
  o <- gsub("[[:punct:]]|[[:cntrl:]]|[^\x20-\x7E]+", "  ", s)
  o <- gsub("[[:blank:]]+", " ", o)
  o <- gsub("^[[:blank:]]+|[[:blank:]]+$", "", o)
  return(o)
}

getStopwords <- function() {
  inFile <- "c:/home/local/workspace/github/coursera-capstone/Coursera-SwiftKey/final/en_US/stopwords.txt"  
  sf <- read.csv(inFile, header=FALSE)
  return(sf)
}

toSpace <- content_transformer(function (x, pattern) gsub(pattern, " ", x))
toBlank <- content_transformer(function (x, pattern) gsub(pattern, "", x))
toNewline <- content_transformer(function (x, pattern) gsub(pattern, " endofline", x))


createCleanCorpus <- function(inDir, outDir, corpusFilename) {
  rawCorpus <- Corpus(DirSource(inDir))
  rawCorpus <- tm_map(rawCorpus, toNewline, "\\.")
  rawCorpus <- tm_map(rawCorpus, content_transformer(tolower))
  rawCorpus <- tm_map(rawCorpus, removeNumbers)
  rawCorpus <- tm_map(rawCorpus, removePunctuation)
  rawCorpus <- tm_map(rawCorpus, toSpace, "[[:punct:]]|[[:cntrl:]]|[^\x20-\x7E]+")
  rawCorpus <- tm_map(rawCorpus, toSpace, "[[:blank:]]+")
  rawCorpus <- tm_map(rawCorpus, removeWords, stopwords("english"))
  rawCorpus <- tm_map(rawCorpus, removeWords, getStopwords())
  rawCorpus <- tm_map(rawCorpus, stripWhitespace)
  rawCorpus <- tm_map(rawCorpus, toBlank, "^[[:blank:]]+|[[:blank:]]+$")
  #rawCorpus <- tm_map(rawCorpus, stemDocument)
  
  # output corpus to file
  writeCorpus(rawCorpus, path=outDir, filenames=corpusFilename)
  
  return(rawCorpus)
}  


#
# variables
#
baseDir <- "c:/home/local/workspace/github/coursera-capstone/Coursera-SwiftKey/final/en_US/corpus_test"
corpusInput <- paste0(baseDir,"/input")
corpusOutput <- paste0(baseDir,"/output")

corpusCleanName <- "corpus_clean.txt"
corpusCleanFullPath <- paste0(corpusOutput, "/", corpusCleanName)

ngram2File <- paste0(corpusOutput, "/ngram2.txt")
ngram3File <- paste0(corpusOutput, "/ngram3.txt")
ngram4File <- paste0(corpusOutput, "/ngram4.txt")

cleanCorpus <- createCleanCorpus(corpusInput, corpusOutput, corpusCleanName)

readCon <- file(corpusCleanFullPath, "r")
writenGram2Con <- file(ngram2File, "w")
writenGram3Con <- file(ngram3File, "w")
writenGram4Con <- file(ngram4File, "w")

rowLen=1
while(rowLen>0) {
  row <- readLines(readCon, 1)
  rowLen=length(row)

  if (rowLen>0) {
    wordCount <- length(unlist(strsplit(row," ")))
    
    if (wordCount>=4) {
      vgrams <- get.ngrams(ngram(row, n=4))
      write(vgrams, writenGram4Con, append=TRUE)
    }
    
    if (wordCount>=3) {
      vgrams <- get.ngrams(ngram(row, n=3))
      write(vgrams, writenGram3Con, append=TRUE)
    }
    
    if (wordCount>=2) {
      vgrams <- get.ngrams(ngram(row, n=2))
      write(vgrams, writenGram2Con, append=TRUE)
    }
  }
}
  
  
close(writenGram2Con)
close(writenGram3Con)
close(writenGram4Con)
close(readCon)




dtm <- DocumentTermMatrix(cleanCorpus)
tdm <- TermDocumentMatrix(cleanCorpus)
tdr <- rollup(tdm, 2, na.rm=TRUE, FUN=sum)
tddf <- as.data.frame(as.matrix(tdr))
word_freqs <- sort(rowSums(tddf), decreasing=TRUE)
dm=data.frame(word=names(word_freqs), freq=word_freqs)

# word cloud
set.seed(100)
wordcloud(head(dm$word,100), head(dm$freq,100), random.order=FALSE,
          colors=brewer.pal(8,"Dark2"))

# histogram
barplot(head(dm$freq,5), names.arg=head(dm$word,5), cex.names=.75, cex.axis=.75,
        main="Top 5 Words by Frequency", xlab="Word", ylab="Frequency Count")

# frequent terms

# remove sparse terms
dtms <- removeSparseTerms(dtm, .1)
dim(dtms)
findFreqTerms(dtm, lowfreq=200)


library(SparseM)
library(Matrix)
library(e1071)
classvec <- vector()
#specify the features, vector to be predicted, and kernel method in the svm model
svm_model <- svm(tdm, tddf, kernel="linear")
summary(svm_model)
#inspect results
pred <- predict(svm_model, dtm)
table(pred,classvec)







