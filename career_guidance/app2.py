import streamlit as st
import spacy
import fitz
import docx
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import Dict, List, Tuple, Any
import re
from spacy.matcher import Matcher

# Load environment variables and configure Gemini AI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except:
    st.error("Please install spaCy model: python -m spacy download en_core_web_lg")

class EnhancedResumeParser:
    def __init__(self):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all matching patterns for different resume sections"""
        # Education patterns
        self.education_patterns = [
            [{"LOWER": {"IN": ["bachelor", "master", "b.tech", "m.tech", "b.e.", "m.e."]}},
             {"LOWER": {"IN": ["of", "in"]}, "OP": "?"},
             {"POS": "PROPN", "OP": "+"}],
            [{"LOWER": {"IN": ["bachelor", "master", "b.tech", "m.tech"]}},
             {"LOWER": {"IN": ["of", "in"]}, "OP": "?"},
             {"LOWER": "computer"},
             {"LOWER": "science"}],
            [{"LOWER": {"IN": ["b.tech", "m.tech", "phd", "mba", "b.sc", "m.sc"]}},
             {"POS": "PROPN", "OP": "*"}]
        ]

        # Technical skills patterns
        self.technical_patterns = [
            [{"LOWER": {"IN": [
                "python", "java", "javascript", "typescript", "c++", "c#",
                "react", "angular", "vue", "node.js", "express", "django",
                "flask", "sql", "mysql", "postgresql", "mongodb", "aws",
                "azure", "docker", "kubernetes", "git", "rest api"
            ]}}]
        ]

        # Soft skills patterns
        self.soft_skills_patterns = [
            [{"LOWER": {"IN": [
                "leadership", "communication", "teamwork", "problem solving",
                "analytical", "project management", "time management",
                "critical thinking", "collaboration", "adaptability"
            ]}}]
        ]

        # Tools patterns
        self.tools_patterns = [
            [{"LOWER": {"IN": [
                "jira", "confluence", "github", "gitlab", "bitbucket",
                "jenkins", "terraform", "ansible", "visual studio code",
                "intellij", "postman", "figma", "slack"
            ]}}]
        ]

        # Add patterns to matcher
        self.matcher.add("EDUCATION", self.education_patterns)
        self.matcher.add("TECHNICAL", self.technical_patterns)
        self.matcher.add("SOFT_SKILLS", self.soft_skills_patterns)
        self.matcher.add("TOOLS", self.tools_patterns)

    def _extract_cgpa(self, text: str) -> Dict[str, str]:
        """Extract CGPA and academic scores from text"""
        scores = {}
        
        # CGPA patterns
        cgpa_patterns = [
            r'CGPA[\s:]+([0-9.]+)(?:/\d+)?',
            r'GPA[\s:]+([0-9.]+)(?:/\d+)?'
        ]
        
        # School score patterns
        school_patterns = {
            '10th': [r'(?:10th|X|Tenth|SSC)[\s:]+([0-9.]+\s*%?)'],
            '12th': [r'(?:12th|XII|Twelfth|HSC)[\s:]+([0-9.]+\s*%?)']
        }

        # Extract CGPA
        for pattern in cgpa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scores['cgpa'] = match.group(1)
                break

        # Extract school scores
        for grade, patterns in school_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    scores[f'{grade}_score'] = match.group(1)
                    break

        return scores

    def parse_resume(self, text: str) -> Dict[str, Any]:
        """Parse resume text and extract structured information"""
        doc = self.nlp(text.lower())
        
        extracted_info = {
            "education": [],
            "skills": {
                "technical": [],
                "soft_skills": [],
                "tools": []
            }
        }

        # Find matches using matcher
        matches = self.matcher(doc)
        
        # Process matches
        for match_id, start, end in matches:
            span = doc[start:end]
            category = self.nlp.vocab.strings[match_id]
            
            if category == "EDUCATION":
                extracted_info["education"].append(span.text.title())
            elif category == "TECHNICAL":
                extracted_info["skills"]["technical"].append(span.text.lower())
            elif category == "SOFT_SKILLS":
                extracted_info["skills"]["soft_skills"].append(span.text.lower())
            elif category == "TOOLS":
                extracted_info["skills"]["tools"].append(span.text.lower())

        # Remove duplicates
        extracted_info["education"] = list(set(extracted_info["education"]))
        for category in extracted_info["skills"]:
            extracted_info["skills"][category] = list(set(extracted_info["skills"][category]))

        return extracted_info

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = " ".join(text.split())
        # Convert to lowercase
        text = text.lower()
        return text

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyMuPDF"""
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf_document:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts"""
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1] * 100

def show_resume_analyzer():
    st.title("Resume Analyzer & Parser")
    
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    
    if uploaded_file:
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_docx(uploaded_file)
        
        # Initialize parser and parse resume
        parser = EnhancedResumeParser()
        parsed_data = parser.parse_resume(text)
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Skills Analysis")
            for category, skills in parsed_data["skills"].items():
                with st.expander(f"{category.title()} Skills"):
                    for skill in skills:
                        st.write(f"- {skill}")
            
            # Calculate skill coverage
            total_skills = sum(len(skills) for skills in parsed_data["skills"].values())
            skill_score = min(100, (total_skills / 15) * 100)  # Assuming 15 skills is a good baseline
            st.metric("Skills Coverage Score", f"{skill_score:.1f}%")
        
        with col2:
            st.subheader("Education")
            for edu in parsed_data["education"]:
                st.write(f"- {edu}")
            
            st.subheader("Recommendations")
            with st.expander("Skills to Add"):
                st.write("- Consider adding cloud computing skills")
                st.write("- Include relevant certifications")
                st.write("- Add specific technical tools used")
            
            with st.expander("Format Improvements"):
                st.write("- Use more action verbs")
                st.write("- Add quantifiable achievements")
                st.write("- Include project outcomes")

def show_job_matcher():
    st.title("Job Matching & Skill Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Resume")
        resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
        
        if resume_file:
            if resume_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(resume_file)
            else:
                resume_text = extract_text_from_docx(resume_file)
            
            parser = EnhancedResumeParser()
            resume_data = parser.parse_resume(resume_text)
    
    with col2:
        st.subheader("Job Description")
        job_description = st.text_area("Paste job description here", height=200)
        
        if job_description and resume_file:
            # Calculate match score
            match_score = calculate_similarity_score(resume_text, job_description)
            
            # Parse job description for skills
            job_data = parser.parse_resume(job_description)
            
            # Display match analysis
            st.subheader("Match Analysis")
            st.metric("Overall Match Score", f"{match_score:.1f}%")
            
            # Compare skills
            with st.expander("Skills Comparison"):
                for category in resume_data["skills"]:
                    resume_skills = set(resume_data["skills"][category])
                    job_skills = set(job_data["skills"][category])
                    
                    st.write(f"\n{category.title()} Skills:")
                    st.write("Matching Skills:", ", ".join(resume_skills & job_skills))
                    st.write("Missing Skills:", ", ".join(job_skills - resume_skills))

def show_resume_enhancer():
    st.title("AI Resume Enhancement")
    
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_docx(uploaded_file)
        
        parser = EnhancedResumeParser()
        parsed_data = parser.parse_resume(text)
        
        st.subheader("Enhancement Suggestions")
        
        with st.expander("Content Analysis"):
            st.write("### Action Verbs")
            st.write("- Replace passive phrases with active verbs")
            st.write("- Use impactful action words")
            
            st.write("### Achievements")
            st.write("- Add specific metrics to accomplishments")
            st.write("- Include project outcomes")
        
        with st.expander("Keyword Optimization"):
            st.write("Recommended additional keywords:")
            st.write("- Industry-specific: 'Digital Transformation', 'Agile Development'")
            st.write("- Technical: 'Cloud Architecture', 'CI/CD'")
            st.write("- Soft Skills: 'Cross-functional Leadership'")

def show_interview_analysis():
    st.title("AI-Powered Interview Analysis")
    
    uploaded_video = st.file_uploader("Upload interview recording", type=["mp4", "mov"])
    
    if uploaded_video:
        st.video(uploaded_video)
        
        # Placeholder for video analysis results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Speech Analysis")
            st.metric("Speaking Pace", "Good")
            st.metric("Clarity", "85%")
            st.metric("Filler Words", "12")
        
        with col2:
            st.subheader("Visual Analysis")
            st.metric("Eye Contact", "90%")
            st.metric("Confidence", "85%")
            st.metric("Engagement", "88%")

def main():
    st.set_page_config(page_title="AI Career Assistant", page_icon="💼", layout="wide")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Resume Analyzer", "Job Matcher", "Resume Enhancer", "Career Chat", "Interview Analysis"]
    )

    if page == "Resume Analyzer":
        show_resume_analyzer()
    elif page == "Job Matcher":
        show_job_matcher()
    elif page == "Resume Enhancer":
        show_resume_enhancer()
    elif page == "Career Chat":
        show_enhanced_career_chat()
    else:
        show_interview_analysis()

def show_enhanced_career_chat():
    st.title("AI Career Assistant")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask me anything about your career...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        context = "\n".join([f"{msg['role']}: {msg['content']}" 
                           for msg in st.session_state.chat_history[-5:]])
        
        response = get_gemini_response(user_input, context)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.rerun()

if __name__ == "__main__":
    main()
