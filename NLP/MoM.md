

Week 1:
	How to: from plain texst to their classification

		Example: sentiment analysis

			*Input: text
			*Output: class of sentiment: positive/negative
			


		Text pre-processing
			Text is:
				* Charcters
				* Words
				* Phrases and named entitles
				* Paragraphs

			Words:
				Word is:
					Meanining sequence of characters

					In English, it's eay to find word

					Input: Friends, Romains, Countrymen, lend me your ears;
					Output: Friends Romans, Countrymen, lend, me, your, ears

			Token:
				meaningful unit furtheror further processing (e.g. word, phrase, sentence etC)	

				Example tokenizers: 
					*whitespace: ntlk.tokenize.WhitespaceTokenizer
						Problem: it, it? different tokens

					*punctuation: ntlk.tokenze.WordPunctTokenizer
						Problem: s, isn, t are not very meaningful

					*set of rules: ntlk.tokenze.TreebankTokenizer
						Example: 'This' 'is' 'Andrew' "'s" 'text' ','' 'is' "n't" 'it' '?'


			Token normalization:
				Problem statement:
					*wolf, wolves -> wolf
					*talk, talks -> talk

				Solution:
					Stemming - process of removing and replacing suffixes to get root form of the word which is called stem;
					nltk.stem.PorterStemming
						SSES -> SS
						IES -> I	
						wolves -> wolv
						feet -> feet
						talked -> talked *

					Lemmatization: same as stemming but doing things properly with the use of vocabularry and morphological analysis, return dictionary form of word - lemma
					nltk.stem.WordNetLemmatizer
						feet -> foot
						wolves -> wolf

						Problem: not all forms are reduces (e.g. talked -> talked)

			Further normalization
				 Normalizing capital letters
				 	US, us -> us
				 	us, US
				 	heuristic (lowercasing the beigning of the sentence, lowercasing words in titles etc)	

		Feature extracting from text (tokens 2 features)

			Bag of words (BOW)
				Marker words in the sentence which (non)presentce makes decision

				Count token occurences:
					Motivation: we're looking for marker words like "excelent" or "disappointed"
					For each token we will have a feature column, this is called - text vectorization

				  
										good movie not a did like
				  good movie 			  1	  1     0  0  0    0
				  not a good movie		  1	  1     1  1  0    0
				  did not like 			  0	  0     1  0  1    1

				  Problems:
				  	 - loose word order, hence the name "bag of words"
				  	 - counters are not normalized

			 Preserve some ordering
			 	n-gram:
			 		1-grams for token
			 		2-grams for token pairs


			 							good movie not a did like (good movie) ...
				  good movie 			  1	  1     0  0  0    0	   1
				  not a good movie		  1	  1     1  1  0    0	   0
				  did not like 			  0	  0     1  0  1    1		0	 

				  Problem:
				     - too many features

			Remove some n-grams based on frequency
				High frequency n-grams (articles, preposition =  stop-words);
				Low frequency n-grams (typos, rare n-grams);

				Medium frquency- remained (what's needed)


			There're lot of medium frequency n-grams
				the n-gram with small freq. can be more discriminating because it can capture the specific issues


			TF-IDF
				Idea: find high term frequency (in the given document) and a low document freq of the term in the while collection (specific issues)

				term freq (TF)
					- tf(t,d) - freq. for term (or n-gram) t in document d

				Freq variants:
					1. binary:  0/1
					2. count:   f[t,d]
					3. term freq: f[t,d] / sum(f[t,d])
					4. log norm: 1 + log (f[t,d])
				
				Inverse doc freq
					N = | D | - total number of documents
					|{d e D: t e d}| - number of document where the term t appears

					idf(t, D) = log( N / |{d e D: t e d}|)			

					TF-IDF:
						tfidf(t,d,D) = tf(t,d) * idf(t,d)

		Linear models for sentiment analysis	
			IMDB data:
				* 25000/25000 positive/negative
				* 30 review per movice
				* at least 7 stars -> positive (1)
				* at most 4 stars -> negative (0)
				* 50/50 train/test split
				* Evaluation accuracy


			TF-IDF
				* 2500



		Hashing trick in spam filtering

			

