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
| CREDBANK | Streaming tweets collected between mid Oct 2014 and end of Feb 2015: topics are classified as events or non events, events annotated with credibility ratings | Model Training |


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

## Future Improvements
