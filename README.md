# LLM as Presidential Candidate: Analyzing Shifts in Political Rhetoric Over Time

## Goal
This project aims to analyze the shifts in political rhetoric and sentiment in U.S. presidential and vice-presidential debates over time. Specifically, it focuses on:
- Investigating the evolution of political sentiment and polarization in rhetoric.
- Fine-tuning a language model to simulate political candidates from different parties and time periods.
- Evaluating sentiment, bias, and polarization in the generated responses.
- Analyzing how rhetoric varies based on party affiliation, time period, and other factors.

## Methodology/Models

### Data Preprocessing
- **Data Source**: Debate transcripts from The American Presidency Project covering U.S. presidential and vice-presidential debates from 1960 to 2024.
- **Data Cleaning**: 
    - Removed irrelevant metadata.
    - Tokenized text for NLP models.
    - Filtered data by party affiliation and time period (1976-1984, 2016-2024).
    - Sentiment analysis scores were computed for each transcript using VADER, RoBERTa, and GPT-4o mini models.

### Sentiment Analysis
- **VADER Sentiment Analysis**: Applied VADER for calculating sentiment scores across debate speeches to gauge sentiment polarity (positive, negative, neutral).
- **RoBERTa and GPT-4o mini Sentiment Analysis**: Fine-tuned and used these models for deeper sentiment analysis to understand nuances in debate rhetoric.

### Fine-tuning LLMs for Candidate Simulation
- **Fine-tuning GPT-4o mini**: Fine-tuned GPT-4o mini to simulate presidential candidates from different parties and time periods. Training data was structured around questions and debates specific to each time period.
    - **Data**: Subsets of debate data were processed and formatted for fine-tuning.
    - **Objective**: Train the models to generate realistic responses that simulate how presidential candidates might respond to debate questions.
  
### Question Generation and Analysis
- **Question Generation**: Used GPT-4 mini to generate debate-like questions based on real presidential debate topics.
- **Analysis**: Performed Latent Dirichlet Analysis (LDA) and TF-IDF to identify themes and topics within the generated responses.

### Evaluation
- **Cosine Similarity**: Used cosine similarity metrics to assess the similarity between generated responses from fine-tuned models and original debate transcripts.
- **Sentiment Evaluation**: Sentiment shifts were analyzed across different models and time periods.

## Results

### Sentiment Analysis Results
- **VADER**: Sentiment scores demonstrated consistent political leanings across party affiliation, with higher sentiment variance in presidential debates.
- **RoBERTa & GPT-4o mini**: The sentiment scores from these models showed finer nuances in emotional tone, such as sarcasm, empathy, or anger, based on party and time period.

### Fine-tuning Results
- The fine-tuned GPT-4o mini models successfully simulated realistic presidential candidate rhetoric. 
    - **Democratic Models (2016, 1976)**: Generated responses with stronger policy emphasis and diplomatic rhetoric.
    - **Republican Models (2016, 1976)**: Exhibited more combative rhetoric with a focus on national security and economic issues.
  
### Cosine Similarity and TF-IDF Results
- **Cosine Similarity**: The generated responses exhibited moderate similarity with real debate responses, indicating that the fine-tuned models captured debate-style language well.
- **TF-IDF**: The analysis revealed varying emphasis on issues over time, with economic concerns increasing in prominence in recent years.

### LDA Analysis
- The LDA analysis on debate speech revealed several distinct themes, including:
    - **1976 Debates**: Focus on foreign policy and economic recovery.
    - **2016 Debates**: Strong focus on immigration and healthcare, with clear polarization between the two parties.

## Data

Data used for this project can be found in the following Google Drive folder: [Google Drive Folder](https://drive.google.com/drive/folders/1N9eaog0ySz0lvh6EruWv-EyUq83HPHCb?usp=sharing)

- **original_data**
    - This folder contains the data that we preprocessed from The American Presidency Project.
        - `presidential_clean.csv` is the preprocessed data, containing all presidential & vice-presidential debate rhetoric from 1960-2024
        - `2016_democrat_data.csv`, `1976_democrat_data.csv`, `2016_republican_data.csv`, `1976_republican_data.csv` are all subsets of `presidential_clean.csv`, subsetted by party and by time period (1976-1984 and 2016-2024, respectively)
- **sentiment_data**
    - `rep_2016_sentiments.csv`, `rep_1976_sentiments.csv`, `dem_2016_sentiments.csv`, `dem_1976_sentiments.csv` all contain the sentiment scores calculated by VADER, RoBERTa, and GPT-4o mini for each model's generated responses
- **generated_responses**
    - `dem_1976_results.csv`, `rep_1976_results.csv`, `dem_2016_results.csv`, `rep_2016_results.csv` all contain the respective model's responses to the different questions
- **finetuning_data**
    - This folder contains the data used to fine-tune GPT-4o mini on the OpenAI website.
        - `dem_1976_finetuning_data.csv`, `rep_1976_finetuning_data.csv`, `dem_2016_finetuning_data.csv`, `rep_2016_finetuning_data.csv` are the CSV files used to create the `.jsonl` files in the `processed_train_val` and `processed_jsonl` folders

## Experiment 1 - Sentiment Analysis
- This folder contains the code (`sentiment_analysis_exp1.ipynb`) used to perform sentiment analysis on the real presidential and vice-presidential debates from 1960-2024.

## Experiment 2 - Fine-tuning LLMs to Simulate Presidential Candidates
- This folder contains the code used to fine-tune GPT-4o mini to simulate presidential candidates, generate debate-like speech from these models, and perform analysis on the generated responses.
    - `cosine_similarity_exp_2.ipynb`
        - This notebook calculates the cosine similarities between the generated responses from the fine-tuned LLMs
    - `generate_questions_democratic_models_exp_2.ipynb`
        - This notebook generates moderator questions (using GPT 4-mini) based on the debate dialogues for Democrats
    - `generate_questions_republican_models_exp_2.ipynb`
        - This notebook generates moderator questions (using GPT 4-mini) based on the debate dialogues for Republicans
    - `LDA_analysis_exp_2.ipynb`
        - This notebook performs Latent Dirichlet Analysis (LDA) on the generated responses from the fine-tuned LLMs.
    - `processing_data_finetuning_exp_2.ipynb`
        - This notebook converts data into the proper format to fine-tune GPT-4o mini on OpenAI's API
    - `query_finetuned_models_exp_2.ipynb`
        - This notebook contains the code for making API calls to our fine-tuned GPT-4o mini models in order to simulate debate speech
    - `sentiment_analysis_exp_2.ipynb`
        - This notebook contains the code used to perform sentiment analysis on the generated responses from the fine-tuned LLMs
    - `tf_idf_by_prompt_exp_2.ipynb`
        - This notebook contains the code for performing TF-IDF analysis by prompt/question for the generated responses from the fine-tuned LLMs
    - `tf_idf_exp_2.ipynb`
        - This notebook contains the code for performing TF-IDF analysis by model for the generated responses from the fine-tuned LLMs
