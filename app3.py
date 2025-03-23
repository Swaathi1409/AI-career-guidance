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

        # Extract CGPA/scores
        extracted_info["academic_scores"] = self._extract_cgpa(text)

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

def get_gemini_response(user_input, context=""):
    """Generates a response from Gemini with career context."""
    system_prompt = """You are a helpful career advisor. Answer questions related to career guidance, 
    job searching, skill development, resumes, interviews, and professional growth. 
    If a question is unrelated to careers and entrepreneurship (e.g., food, entertainment, sports), 
    firmly respond: 'I'm here to assist with career-related topics only.'"""
    
    full_prompt = f"{system_prompt}\n\nContext: {context}\n\nUser: {user_input}"
    response = model.generate_content(full_prompt)
    return response.text

def generate_cover_letter(resume_data, job_description):
    """Generate a cover letter using the parsed resume data and job description."""
    
    # Extract skills as comma-separated string
    all_skills = []
    for skill_category in resume_data["skills"]:
        all_skills.extend(resume_data["skills"][skill_category])
    skills_str = ", ".join(all_skills)
    
    # Education as string
    education_str = ", ".join(resume_data["education"])
    
    prompt = f"""
    Create a professional cover letter based on the following:

    Resume Details:
    - Skills: {skills_str}
    - Education: {education_str}
    
    Job Description:
    {job_description}

    Instructions:
    1. Tailor the cover letter to the job description, highlighting relevant skills.
    2. Use a professional and enthusiastic tone.
    3. Keep the letter concise (3-4 paragraphs).
    4. Include a generic salutation like "Dear Hiring Team".
    5. Format it properly with date, address blocks, and signature.
    """
    
    response = model.generate_content(prompt)
    return response.text

def generate_interview_questions(resume_data):
    """Generate technical interview questions based on the resume."""
    
    # Extract skills as comma-separated string
    all_skills = []
    for skill_category in resume_data["skills"]:
        all_skills.extend(resume_data["skills"][skill_category])
    skills_str = ", ".join(all_skills)
    
    # Education as string
    education_str = ", ".join(resume_data["education"])
    
    prompt = f"""
    Based on the candidate's profile:

    Skills: {skills_str}
    Education: {education_str}
    
    Generate 5 technical interview questions that:
    1. Focus on the candidate's core skills
    2. Include practical problem-solving scenarios
    3. Assess both theoretical knowledge and practical application
    4. Progress from basic concepts to more complex scenarios
    5. Cover different aspects of their technical expertise

    Format each question clearly and include any necessary context.
    """
    
    response = model.generate_content(prompt)
    questions = [q.strip() for q in response.text.split("\n") if q.strip() and not q.strip().startswith(("1.", "2.", "3.", "4.", "5."))]
    return questions

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
            
            # Show academic scores if available
            if parsed_data.get("academic_scores"):
                st.subheader("Academic Scores")
                for score_type, value in parsed_data["academic_scores"].items():
                    st.write(f"- {score_type.replace('_', ' ').title()}: {value}")
            
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
    
    # Store parsed resume in session state
    if "parsed_resume_data" not in st.session_state:
        st.session_state.parsed_resume_data = None
    
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
            st.session_state.parsed_resume_data = resume_data
            st.session_state.resume_text = resume_text
    
    with col2:
        st.subheader("Job Description")
        job_description = st.text_area("Paste job description here", height=200)
        
        if job_description and resume_file:
            # Calculate match score
            match_score = calculate_similarity_score(st.session_state.resume_text, job_description)
            
            # Parse job description for skills
            parser = EnhancedResumeParser()
            job_data = parser.parse_resume(job_description)
            
            # Display match analysis
            st.subheader("Match Analysis")
            st.metric("Overall Match Score", f"{match_score:.1f}%")
            
            # Compare skills
            with st.expander("Skills Comparison"):
                for category in st.session_state.parsed_resume_data["skills"]:
                    resume_skills = set(st.session_state.parsed_resume_data["skills"][category])
                    job_skills = set(job_data["skills"][category])
                    
                    st.write(f"\n{category.title()} Skills:")
                    st.write("Matching Skills:", ", ".join(resume_skills & job_skills) if resume_skills & job_skills else "None")
                    st.write("Missing Skills:", ", ".join(job_skills - resume_skills) if job_skills - resume_skills else "None")
    
    # Cover letter generator
    if st.session_state.parsed_resume_data is not None and job_description:
        st.subheader("Generate Cover Letter")
        if st.button("Generate Cover Letter"):
            with st.spinner("Generating your cover letter..."):
                cover_letter = generate_cover_letter(st.session_state.parsed_resume_data, job_description)
                st.markdown("### Your Cover Letter")
                st.markdown(cover_letter)
                st.download_button(
                    label="Download Cover Letter",
                    data=cover_letter,
                    file_name="cover_letter.txt",
                    mime="text/plain"
                )

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
        
        # Get AI suggestions for resume improvement
        prompt = f"""
        Analyze this resume information and provide specific suggestions for improvement:
        
        Skills: {[skill for category in parsed_data['skills'] for skill in parsed_data['skills'][category]]}
        Education: {parsed_data['education']}
        
        Provide specific suggestions for:
        1. Skills to add or elaborate on
        2. Resume formatting improvements
        3. Content enhancements (action verbs, metrics, achievements)
        4. ATS optimization tips
        """
        
        with st.spinner("Analyzing your resume..."):
            suggestions = model.generate_content(prompt).text
        
        st.subheader("Enhancement Suggestions")
        st.markdown(suggestions)
        
        with st.expander("AI Rewrite Examples"):
            st.write("### Weak bullet point:")
            st.write("Responsible for managing a team.")
            
            st.write("### AI improved version:")
            st.write("Led cross-functional team of 8 members, increasing department productivity by 27% and reducing costs by $45K annually.")

def generate_interview_questions(resume_data):
    """Generate technical interview questions based on the resume."""
    
    # Extract skills as comma-separated string
    all_skills = []
    for skill_category in resume_data["skills"]:
        all_skills.extend(resume_data["skills"][skill_category])
    skills_str = ", ".join(all_skills)
    
    # Education as string
    education_str = ", ".join(resume_data["education"])
    
    prompt = f"""
    Based on the candidate's profile:

    Skills: {skills_str}
    Education: {education_str}
    
    Generate exactly 5 technical interview questions that:
    1. Focus on the candidate's core skills
    2. Include practical problem-solving scenarios
    3. Assess both theoretical knowledge and practical application
    4. Progress from basic concepts to more complex scenarios
    5. Cover different aspects of their technical expertise

    IMPORTANT FORMATTING INSTRUCTIONS:
    - Generate ONLY the questions themselves
    - Each question MUST end with a question mark
    - Do NOT include any numbering or labels (like "Q1:", "Question 1:", etc.)
    - Do NOT include any headers, categories, or explanations
    - Do NOT include any notes or instructions
    - Output EXACTLY 5 questions total, each on its own line
    - Make each question directly actionable for an interview
    """
    
    response = model.generate_content(prompt)
    
    # Process the response to ensure we only get questions
    questions = []
    for line in response.text.split("\n"):
        line = line.strip()
        # Skip empty lines, headings, and non-question content
        if not line or not line.endswith("?"):
            continue
        
        # Remove any numbering prefixes if they exist
        line = re.sub(r"^(\d+\.|\(?\d+\)?|Q\d+:|Question \d+:)\s*", "", line)
        questions.append(line)
    
    # If we still don't have exactly 5 questions, look for non-obvious questions
    if len(questions) < 5:
        for line in response.text.split("\n"):
            line = line.strip()
            # Skip empty lines and lines already added
            if not line or line in questions:
                continue
            
            # Remove any numbering prefixes
            clean_line = re.sub(r"^(\d+\.|\(?\d+\)?|Q\d+:|Question \d+:)\s*", "", line)
            
            # If it has sentence structure but doesn't end with ?, add it with a ?
            if len(clean_line) > 20 and not clean_line.endswith("?") and clean_line not in questions:
                clean_line = clean_line + "?"
                questions.append(clean_line)
                if len(questions) >= 5:
                    break
    
    # Limit to 5 questions
    return questions[:5]

def show_interview_preparation():
    st.title("AI Powered Interview Preparation")
    
    # Allow user to upload resume and job description
    resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    job_description = st.text_area("Paste job description (optional)", height=150)
    
    # Additional options
    industry = st.selectbox("Select your industry", 
                           ["Software Development", "Data Science", "Marketing", 
                            "Sales", "Finance", "Healthcare", "Other"])
    
    role_level = st.selectbox("Experience level", 
                             ["Entry Level", "Mid-level", "Senior", "Management"])
    
    if st.button("Generate Interview Questions"):
        if resume_file:
            # Extract text from resume
            if resume_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(resume_file)
            else:
                resume_text = extract_text_from_docx(resume_file)
                
            # Parse resume for skills
            parser = EnhancedResumeParser()
            resume_data = parser.parse_resume(resume_text)
            
            # Generate technical questions
            with st.spinner("Generating technical questions..."):
                technical_questions = generate_interview_questions(resume_data)
            
            # Generate behavioral questions
            with st.spinner("Generating behavioral questions..."):
                prompt = f"""
                Generate exactly 5 behavioral interview questions for a {role_level} position in {industry}.
                
                Resume skills: {', '.join(resume_data['skills']['technical'] + resume_data['skills']['soft_skills'])}
                Job description: {job_description}
                
                IMPORTANT FORMATTING INSTRUCTIONS:
                - Generate ONLY the interview questions, NOT the topics
                - Each question MUST end with a question mark (?)
                - Do NOT include any numbering, labels, or explanations
                - Do NOT include headers, categories, or topics—only direct, full questions
                - Output EXACTLY 5 questions
                - Ensure each question is fully formed and clear
                """
                
                behavioral_response = model.generate_content(prompt)
                
                # Extract only actual questions
                # Extract only actual questions
                behavioral_questions = []
                for line in behavioral_response.text.split("\n"):
                    line = line.strip()
                    if not line or "**" in line:  # Ignore bold text (headers)
                        continue

                    # Remove numbering prefixes (e.g., "1.", "Q1:", "(1)", etc.)
                    line = re.sub(r"^(\d+\.|\(?\d+\)?|Q\d+:|Question \d+:)\s*", "", line)
                    
                    # Ensure the line is a valid question (ends with '?')
                    if line.endswith("?") and len(line.split()) > 3:  # Ensuring it's not just a single-word question
                        behavioral_questions.append(line)

                # Keep exactly 5 valid questions
                behavioral_questions = behavioral_questions[:5]

            
            # Display technical questions
            st.subheader("Technical Questions")
            for i, question in enumerate(technical_questions, 1):
                with st.expander(f"Question {i}:"):
                    st.write(question)
                    st.text_area(f"Practice your answer to question {i}", key=f"tech_q_{i}", height=150)
            
            # Display behavioral questions
            st.subheader("Behavioral Questions")
            for i, question in enumerate(behavioral_questions, len(technical_questions) + 1):
                with st.expander(f"Question {i}:"):
                    st.write(question)
                    st.text_area(f"Practice your answer to question {i}", key=f"behav_q_{i}", height=150)
            
            # Tips section
            with st.expander("Interview Success Tips"):
                st.markdown("""
                ### STAR Method for Behavioral Questions
                - **Situation**: Describe the context
                - **Task**: Explain your responsibility
                - **Action**: Detail the steps you took
                - **Result**: Share the outcome and any lessons learned
                
                ### Technical Question Tips
                - Start with the fundamentals before diving into complex solutions
                - Think out loud to show your problem-solving approach
                - If you don't know something, explain how you would find the answer
                """)
        else:
            st.warning("Please upload your resume to generate tailored questions")
            
def show_enhanced_career_chat():
    st.title("AI Career Assistant")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    user_input = st.chat_input("Ask me anything about your career...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Format context from previous messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" 
                           for msg in st.session_state.chat_history[-5:]])
        
        # Get response from Gemini
        with st.spinner("Thinking..."):
            response = get_gemini_response(user_input, context)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Force UI update
        st.rerun()

def main():
    st.set_page_config(page_title="AI Career Assistant", page_icon="💼", layout="wide")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "parsed_resume_data" not in st.session_state:
        st.session_state.parsed_resume_data = None

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Resume Analyzer", "Job Matcher", "Resume Enhancer", "Career Chat", "Interview Preparation"]
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
        # Fix: Call the correct function name
        show_interview_preparation()

if __name__ == "__main__":
    main()
