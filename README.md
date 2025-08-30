# Sift 🧹

LLM-powered spam review filter

Sift is a project that filters out spam, inappropriate, and irrelevant reviews from authentic ones. By combining open-source LLMs, regex-based fine-tuning, and data preprocessing techniques, Sift aims to make online reviews more reliable and trustworthy.

## ✨ Features

Review Scraping – Uses Apify to scrape reviews from Google Maps.

Data Cleaning & Preprocessing – Handles missing values, formatting, and text normalization with pandas.

Topic Classification – Identifies review categories such as:

- ✅ Authentic (genuine customer feedback)

- 🚫 Inappropriate (offensive language, harassment)

- 🔒 Personal Information (sensitive details shared in reviews)

- 📢 Advertisement/Spam (promotional or irrelevant content)

Robustness Testing – Includes synthetic edge-case reviews (e.g., misspellings, disguised spam) to test resilience.

Hybrid Approach – Combines TF-IDF topic exploration, regex-based signals, and LLM classification for best results.

## 🛠️ Tech Stack

Apify – for web scraping (Google Maps reviews)

pandas – for data cleaning and handling missing values

scikit-learn (TF-IDF) – for topic modeling and feature extraction

Open-source LLMs – for flexible, semantic classification

Regular Expressions (Regex) – for rule-based fine-tuning

## 📊 Demo

We prepared a 3-minute video demo showcasing:

The scraping and preprocessing pipeline

Our classification logic (TF-IDF + regex + LLM)

Real and synthetic reviews being filtered into categories

## ⚖️ Limitations & Future Work

Current classification relies on few-shot prompting + regex.

Could definitely be improved! Perhaps by training a custom fine-tuned model.

Expansion to more intricate and abstract categories (e.g., sarcasm, fake positivity) seems like a fun challenge that we would be interested in looking into!

Possible deployment as a REST API or browser extension.

## 👨‍💻 Team

Developed as part of team APXGP

Contributors:
- Yap Jia Wei [https://github.com/heeelol]
- Chin Li-Loong [https://github.com/chinll1]
