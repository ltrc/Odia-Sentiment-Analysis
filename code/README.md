# This details different parameters required for the programs
## 3 programs
1. python train_models_and_perform_k_fold_cross_validation.py --input input_file_path --analyzer analyzer --ngram min_n max_n --classifier classifier_name
   - Unigram  <br>
   python train_models_and_perform_k_fold_cross_validation.py --input input_file_path --analyzer word --ngram 1 1 --classifier svm
   - Unigram + Bigram<br>
   python train_models_and_perform_k_fold_cross_validation.py --input input_file_path --analyzer word --ngram 1 2 --classifier svm
   - Unigram + Bigram + Trigram<br>
   python train_models_and_perform_k_fold_cross_validation.py --input input_file_path --analyzer word --ngram 1 3 --classifier svm
   - Char 2-6 grams  <br>
   python train_models_and_perform_k_fold_cross_validation.py --input input_file_path --analyzer char --ngram 2 6 --classifier svm
   - Char 3-6 grams  <br>
   python train_models_and_perform_k_fold_cross_validation.py --input input_file_path --analyzer word --ngram 3 6 --classifier svm
2. python train_models_and_perform_k_fold_cross_validation_using_senti_wordnet.py --input input_file_path --analyzer word --ngram 1 1 --classifier svm
   - all the other options are similar to the above program
3. python train_models_and_perform_k_fold_cross_validation_combining_word_char_ngrams_and_senti_wordnet.py --input input_file_path --classifier svm
4. 4 classifiers are used in the programs, so pass any of the below classifier names in the <em>classifier</em> parameter
   - **svm** for Support Vector Machines
   - **rf** for Random Forests
   - **logit** for Logistic Regression
   - **bnb** for Bernoulli Naive Bayes
