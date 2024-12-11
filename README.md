# LLM as Presidential Candidate: Analyzing Shifts in Political Rhetoric Over Time
## 6.8610 Project Code & Documentation

## Data

Data used for this project can be found in the following Google Drive folder: https://drive.google.com/drive/folders/1N9eaog0ySz0lvh6EruWv-EyUq83HPHCb?usp=sharing

- original_data
    - This folder contains the data that we preprocessed from The American Presidency Project.
        - presidential_clean.csv is the preprocessed data, containing all presidential & vice-presidential debate rhetoric from 1960-2024
        - 2016_democrat_data.csv, 1976_democrat_data.csv, 2016_republican_data.csv, 1976_republican_data.csv are all subsets of presidential_clean.csv, subsetted by party and by time period (1976-1984 and 2016-2024, respectively)
- sentiment_data
    - rep_2016_sentiments.csv, rep_1976_sentiments.csv, dem_2016_sentiments.csv, dem_1976_sentiments.csv all contain the sentiment scores calculated by VADER, RoBERTa, and GPT-4o mini for each model's generated responses
- generated_responses
    - dem_1976_results.csv, rep_1976_results.csv, dem_2016_results.csv, dem_2016_results.csv all contain the respective model's responses to the different questions
- finetuning_data
    - This folder contains the data used to fine-tune GPT-4o mini on the OpenAI website. 
        - dem_1976_finetuning_data.csv, rep_1976_finetuning_data.csv, dem_2016_finetuning_data.csv, rep_2016_finetuning_data.csv are the csv files used to create the .jsonl files in the processed_train_val and processed_jsonl folders

## Experiment 1 - Sentiment Analysis

- This folder contains the code (sentiment_analysis_exp1.ipynb) used to perform sentiment analysis on the real presidential and vice-presidential debates from 1960-2024.

## Experiment 2 - Fine-tuning LLMs to Simulate Presidential Candidates
- This folder contains the code used to fine-tune GPT-4o mini to simulate presidential candidates, generate debate-like speech from these models, and perform analysis on the generated responses.
    - cosine_similarity_exp_2.ipynb
        - This notebook calculates the cosine similarities between the generated responses from the fine-tuned LLMs
    - generate_questions_democratic_models_exp_2.ipynb
        - This notebook generates moderator questions (using GPT 4-mini) based on the debate dialogues for Democrats
    - generate_questions_republican_models_exp_2.ipynb
        - This notebook generates moderator questions (using GPT 4-mini) based on the debate dialogues for Republicans
    - LDA_analysis_exp_2.ipynb
        - This notebook performs Latent Dirichlet Analysis (LDA) on the generated responses from the fine-tuned LLMs.
    - processing_data_finetuning_exp_2.ipynb
        - This notebook converts data into the proper format to fine-tune GPT-4o mini on OpenAI's API
    - query_finetuned_models_exp_2.ipynb
        - This notebook contains the code for making API Calls to our fine-tuned GPT-4o mini models in order to simulate debate speech
    - sentiment_analysis_exp_2.ipynb
        - This notebook contains the code used to perform sentiment analysis on the generated responses form the fine-tuned LLMs
    - tf_idf_by_prompt_exp_2.ipynb
        - This notebook contains the code for performing TF-IDF analysis by prompt/question for the generated responses from the fine-tuned LLMs
    - tf_idf_exp_2.ipynb
        - This notebook contains the code for performing TF-IDF analysis by model for the generated responses from the fine-tuned LLMs
    