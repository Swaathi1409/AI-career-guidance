# -*- coding: utf-8 -*-
"""CVtoResume.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1r_X26mqX6cueiEBC4W7J-q3sgTqBiQ2p
"""

import spacy
import docx
import PyPDF2
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import defaultdict

"""EXTRACTING TEXT FROM PDF"""

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

"""EXTRACTING TEXT FROM DOC

"""

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    doc = docx.Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text



"""EXTRACTING REQIREMENTS"""



def main():
    # Load spaCy model
    nlp = spacy.load('en_core_web_lg')


    # File paths
    cv_path = "C:\\Users\\vaish\\Downloads\\sample-cv.pdf"  
    job_desc_path = "C:\\Users\\vaish\\Downloads\\entry-job-description.pdf" 
    output_path = "C:\\Users\\vaish\\Downloads\\targeted_resume.docx"

    # Extract text from files
    cv_text = (extract_text_from_pdf(cv_path) if cv_path.endswith('.pdf')
               else extract_text_from_docx(cv_path))

    job_text = (extract_text_from_pdf(job_desc_path) if job_desc_path.endswith('.pdf')
                else extract_text_from_docx(job_desc_path))


if __name__ == "__main__":
    main()

import google.generativeai as genai
import os
import spacy
from spacy.matcher import PhraseMatcher

# Configure your Gemini API key
genai.configure(api_key="AIzaSyDTIynYPyoY_vjVXUqarwX-2BDg1Efp-aE")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_skills_gemini(job_description):
    """
    Extracts skills from a job description using the Gemini API, handling multi-word skills.
    """
    model = genai.GenerativeModel('gemini-pro')
    doc = nlp(job_description)

    # 1. Phrase Matching for Known Skills
    phrase_matcher = PhraseMatcher(nlp.vocab)
    skill_patterns = [nlp(skill) for skill in ["machine learning", "deep learning", "data analysis"]] #Add to this list.
    phrase_matcher.add("Skills", skill_patterns)
    matched_skills = [doc[start:end].text for match_id, start, end in phrase_matcher(doc)]

    # 2. Extract Noun Chunks
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # Combine and Remove duplicates.
    potential_skills = list(set(matched_skills + noun_chunks))

    # 3. Filter Noun Chunks (Optional, but recommended)
    pronouns = ["we", "you", "he", "she", "it", "they", "i", "me", "him", "her", "us", "them"]
    common_nouns = ["experience", "knowledge", "ability", "skills", "understanding", "background", "role", "responsibilities", "requirements", "duties", "qualifications"]

    filtered_skills = [
        skill for skill in potential_skills
        if skill.lower() not in pronouns and skill.lower() not in common_nouns
    ]

    skills = []
    for skill in filtered_skills:
        prompt = f"Is '{skill}' a specific professional skill? Answer with 'yes' or 'no' only."
        try:
            response = model.generate_content(prompt)
            answer = response.text.strip().lower()

            if "yes" in answer:
                skills.append(skill)

        except Exception as e:
            print(f"Error processing '{skill}': {e}")

    return skills

# Example usage:
job_description = "We are seeking a skilled Data Scientist with expertise in Python, machine learning, and deep learning. Experience with cloud platforms like AWS and Azure is essential. Strong knowledge of SQL and data analysis is required."

skills = extract_skills_gemini(job_description)
print("Skills (Gemini):", skills)

