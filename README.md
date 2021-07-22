![NFT Sale Prediction](images/NFT_Banner.png)
# Predicting whether an NFT will sell
**Author:** Shawn Samuel

## Overview

Using the OpenSea API to retrieve NFT data to train a model that can predict whether a given NFT will sell of not. Features were convertec to binary columns that returned a binary prediction with 91.26% precision. After exploring feature importance, we found that which particular marketplace the NFT was sold on greatly impacted the outcome. Several columns were acting as proxies for this information. We would recommmend that if an NFT is being created with the primary purpose of trying to sell it, or purchased for the purposes of being resold, the marketplace that the NFT is on is an important factor. 

## Business Problem

For this project, I am answering the question "What are the 5 best zip codes to invest in?" for a fictional real estate agency: ACME Realtors. To further clarify, I will be designing a tool for first-time home buyers (FTHB) to help them think like investors. Generally, first-time home buyers are overwhelmed by many factors when considering what home to buy. The investment potential of particualr areas may not be top of mind. Also, FTHB may not have access to high-performance data analytics and machine learning tools that are usually geared towards real estate investors. The use case for this tool will be for FTHB who have an idea (either due to preference or restriction) of a group of zipcodes they are interested in. Once this machine learning tool is built, it can be used as an added-value product ACME Realtors can provide their FTHB clients. For the FTHB, it will be a resource to enable them to think about their home purchase as an investment and maximize returns.

## Data Understanding

The data was gathered in sets of 10,000 rows according to the OpenSea API limit. Various filters were used to try and obtain as many unique NFTs as possible. One of the major challenges were that despite running 10 sets of 10,000 pulls, there were still many duplicates in the various queries. Another major hurdle is the fact that, since OpenSea acts as a marketplace for other marketplaces, the columns of data contain many null values based on which columns are used by which marketplace. Even the target had to be inferred from three different columns. We decided to create binary columns for each feature. Significant decison making and engineering went into the creation of binary columns with different approaches for boolean, integer, object columns w/ nulls and object columns w/o nulls.


In addition, the process was repeated to create a [test set](notebooks/new_test_data) of an additional 26,241 NFTs.

## Methods

Our methodoolgy had four main components. 

1. [Data Gathering](notebooks/1_data_gather.ipynb) - the OpenSea API allows for 10,000 NFTs to be retrieved at a time. This was done several times and condensed to 44,752 unique NFTs
2. [Data Cleaning](notebooks/2_cleaning.ipynb) & [Preprocessing](notebooks/3_preprocessing.ipynb)- the unique NFTs were then cleaned to dilineate the target columns and remove nulls. The target columns were converted to binary and combined; the boolean, integer and object feature columns were also converted to binary based on relevant parameters
3. [Model](project-notebook) & [Analyze Features](notebooks/5_visualizations) - we trained several different models and evaluated against a baseline. The decision tree returned best results and features were analyzed using feature importance plot, Skope Rules and Shap. The important features were further visualized to draw conclusions.
   
## Results

Of the models we built, the decision tree with shuffling that was fit on the entire dataset, crossvalidated for an average precision of **91.26%** with Std Dev of .27%. Our final model dt5 outperforms simply selecting the major class which would would result in 60% precision.

## Conclusion

Through various methods of evaluating feature importance, *'asset_contract.total_supply', 'token_metadata', and 'collection.display_data.card_display_style'* surfaced as important features. However after [visualization](notebooks/5_visualizations.ipynb) we found that some of these were proxies to separating popular marketplaces with high sell rates. At a minimum, it is clear that where you mint your NFT is a strong indicator of whether it will sell or not. This is a simple binary indicator and we believe more work can be done to create further utility from such a prediction. To guide further work the question is WHY? NFTs, much like the underlying cryptocurrency framework, have been extremely volatile and have seen waves of popularity. Providing tools to the creators, such as information about what would make an NFT more valuable to potential buyers, would help further this discovery phase.

## Future Work

Future work suggestions include:

1. Gathering data with a wider variety to train the model
2. Moving beyond binary features and build out richer features based on importance 
3. Build a regression model with an actual price prediction
4. Building a deployed user interface where an OpenSea NFT ID could be entered and prediction returned

## For More Information

Please review the full analysis in the [Jupyter Notebook](project-notebook.ipynb) or the [presentation](project-presentation.pdf).

For additional info, contact Shawn Samuel at [shawnjohnsamuel@gmail.com](mailto:shawnjohnsamuel@gmail.com)

## Repository Structure

```
├── data
├── images
├── models
├── notebooks
├── README.md
├── project-notebook.ipynb
└── project-presentation.pdf
```
