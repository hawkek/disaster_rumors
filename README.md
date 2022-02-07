# Detecting Rumours in Disaster Related Tweets

### Contributors:
* Claire Cateland, Kevin Luu, Kirsty Hawke, Richard Stassen, Saud Nasri, Sean Atkinson

## Problem Statement

While social media increasingly becomes the first source of information about disasters, this information is not always correct. Posts can be misleading or mistaken. On rarer occasions, users intentionally spread misinformation. In short, can we produce a machine learning model to identify whether tweets are related to disaster events and to assess their credibility?

## Description

### Datasets
Relevancy:
| Name                                                                                        | Description                                                                                    | Use               |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------- |
| [Kaggle - Disaster Relevance](https://www.kaggle.com/jannesklaas/disasters-on-social-media) | Collection of tweets, hand classified as relevant or irrelevant to disasters of numerous types | Model Training    |
| [Google News Word2Vec](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)             | Pre-trained Google News corpus (3 billion running words) word vector model                     | Embedding Weights |
| [Kaggle - NLP Competition](https://www.kaggle.com/c/nlp-getting-started)                    | Collection of tweets, hand classified as relevant or irrelevant to disasters of numerous types | Model Testing     |

Credibility:
| Name     | Description                                                                                                                                                   | Use            |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| [CREDBANK](https://github.com/compsocial/CREDBANK-data#readme) | Streaming tweets collected between mid Oct 2014 and end of Feb 2015: topics are classified as events or non events, events annotated with credibility ratings | Model Training |

## Files
* Executive Summary:
* Report: 
* Web Scraping code:
* Disaster Modeling code:
* Rumor Modeling:
* Web app:


## Results
### Relevance Model Results
| Model                        | Training Accuracy | Validation Accuracy |
| ---------------------------- | ----------------- | ------------------- |
| Logistic Regression          | 0.883             | 0.795               |
| Naive Bayes                  | 0.901             | 0.801               |
| Decision Tree                | 0.981             | 0.728               |
| Random Forest                | 0.988             | 0.797               |
| Convolutional Neural Network | 0.835             | 0.831               |

### Credibility Model Results
| Model                        | Training Accuracy | Validation Accuracy |
| ---------------------------- | ----------------- | ------------------- |
| MultiNB                      | 0.954             | 0.938               |
| Random Forest                | 0.996             | 0.978               |
| Bagging                      | 0.995             | 0.973               |
| ADABoost                     | 0.954             | 0.952               |
| KNN                          | 0.881             | 0.862               |
| Decision tree                | 0.997             | 0.969               |
| SVC                          | 0.992             | 0.983               |
| Convolutional Neural Network | 0.983             | 0.982               |

## Conclusion
The results of our analysis are largely consistent with Hunt, Agarwal & Puneet (2020). Each of our ML models achieves a high degree of accuracy when classifying disaster-related tweets as rumors or non-rumours, but only when evaluating tweets relating to events that are represented in training data. When we attempt to generalize our model. applying it to tweets relating to an unseen event, it shows poor performance with an accuracy similar to our baseline. There are several reasons why this may be the case, not the least of which is data quality. Even simple image labeling can be expensive and time-consuming, but credibility data needs to be labeled as true or false, which can require fact-checking and sometimes domain knowledge. Humans are also subject to biases and interpretation, which can result in incorrect or imprecise data labels.

That said, the issues with the CREDBANK leave the question open, as it is possible that a similar dataset that is rigorously cleaned and vetted could perform well. Moreover, a robust dataset that is specifically about disaster tweets could also have potential as a training set for ML algorithms. For future research into verification of disaster tweets we’d recommend one of the following approaches:
* Use a purpose-built dataset comprised of disaster-related tweets that are manually labelled as rumour or not-rumour.
* Use a broader feature set that includes textual and non-textual features in evaluating the veracity of disaster-tweets.
* Select a web app host that can support a neural net.
* Explore if transformers could perform better.

## References

C. Buntain and J. Golbeck, "Automatically Identifying Fake News in Popular Twitter Threads," 2017 IEEE International Conference on Smart Cloud (SmartCloud), 2017, pp. 208-215, doi: 10.1109/SmartCloud.2017.40.

Han S., Gao, J., Ciravegna, F. (2019). "Neural Language Model Based Training Data Augmentation for Weakly Supervised Early Rumor Detection", The 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM 2019), Vancouver, Canada, 27-30 August, 2019

Mitra, T., & Gilbert, E. (2021). CREDBANK: A Large-Scale Social Media Corpus With Associated Credibility Annotations. Proceedings of the International AAAI Conference on Web and Social Media, 9(1), 258-267. Retrieved from https://ojs.aaai.org/index.php/ICWSM/article/view/14625

Murayama, T., Wakamiya, S., Aramaki, E., & Kobayashi, R. (2021). Modeling the spread of fake news on Twitter. PLOS ONE, 16(4). https://doi.org/10.1371/journal.pone.0250419
Sharma, K., Qian, F., Jiang, H., Ruchansky, N., Zhang, M., & Liu, Y. (2019). Combating fake news. ACM Transactions on Intelligent Systems and Technology, 10(3), 1–42. https://doi.org/10.1145/3305260

Shu K, Mahudeswaran D, Wang S, Lee D, Liu H (2020) FakeNewsNet: a data repository with news content, social context, and spatiotemporal information for studying fake news on social media. Big Data 8:3, 171–188, DOI: 10.1089/big.2020.0062. 
