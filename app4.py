import streamlit as st
st.set_page_config(page_title="AI Career Assistant", page_icon="💼")

import spacy
import fitz
import docx
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import base64
from typing import Dict, List, Tuple, Any
import re
from spacy.matcher import Matcher

# Load spaCy model
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_lg")
    except:
        # Show a progress message
        with st.spinner("Downloading spaCy model (this may take a few minutes)..."):
            # Run the download command
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
        # Try loading again after download
        return spacy.load("en_core_web_lg")
nlp = load_nlp_model()

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

# Rule-based text generation functions to replace Gemini
def generate_cover_letter(resume_data, job_description):
    """Generate a cover letter using the parsed resume data and job description without using Gemini."""
    # Extract skills as comma-separated string
    all_skills = []
    for skill_category in resume_data["skills"]:
        all_skills.extend(resume_data["skills"][skill_category])
    
    # Get top 5 skills (or fewer if not enough skills)
    top_skills = all_skills[:min(5, len(all_skills))]
    
    # Education as string
    education = resume_data["education"][0] if resume_data["education"] else "relevant degree"
    
    # Extract job title from job description (simplified approach)
    job_title_match = re.search(r'(developer|engineer|analyst|manager|designer|specialist)', 
                               job_description.lower())
    job_title = job_title_match.group(0).title() if job_title_match else "Position"
    
    # Current date
    from datetime import date
    today = date.today().strftime("%B %d, %Y")
    
    # Generate cover letter
    cover_letter = f"""
{today}

Dear Hiring Team,

I am writing to express my strong interest in the {job_title} position as advertised. With my background in {education} and expertise in {', '.join(top_skills)}, I am confident in my ability to make significant contributions to your team.

Based on the job description, I understand you're looking for someone with strong technical skills and problem-solving abilities. Throughout my career, I have consistently demonstrated these qualities while working on various projects. My experience with {top_skills[0] if top_skills else 'relevant technologies'} has prepared me well for the challenges this role presents.

I am particularly drawn to this opportunity because it aligns with my professional goals and passion for technology. I am excited about the possibility of bringing my skills in {', '.join(top_skills[1:3] if len(top_skills) > 1 else top_skills)} to your team and contributing to your company's success.

I would welcome the opportunity to discuss how my background, skills, and experiences would be an ideal fit for this position. Thank you for considering my application, and I look forward to the possibility of working with your team.

Sincerely,
[Your Name]
"""
    return cover_letter

def generate_technical_questions(resume_data):
    """Generate technical interview questions based on the resume without using Gemini."""
    # Get technical skills
    technical_skills = resume_data["skills"]["technical"]
    
    # Basic questions for common skills
    question_templates = {
        "python": [
            "Can you explain how exception handling works in Python?",
            "What are Python decorators and how would you use them?",
            "Describe the difference between lists and tuples in Python."
        ],
        "java": [
            "What is the difference between an interface and an abstract class in Java?",
            "Explain Java's garbage collection process.",
            "How does multithreading work in Java?"
        ],
        "javascript": [
            "Explain the concept of closures in JavaScript.",
            "What's the difference between '==' and '===' in JavaScript?",
            "How does asynchronous programming work in JavaScript?"
        ],
        "sql": [
            "What is the difference between INNER JOIN and LEFT JOIN?",
            "How would you optimize a slow SQL query?",
            "Explain the concept of database normalization."
        ],
        "react": [
            "What are React hooks and how do they work?",
            "Explain the Virtual DOM concept in React.",
            "How would you manage state in a large React application?"
        ]
    }
    
    # Generic technical questions for any role
    generic_questions = [
        "Describe a challenging technical problem you faced and how you solved it.",
        "How do you approach debugging a complex issue in your code?",
        "What's your process for learning new technologies?",
        "How do you ensure code quality in your projects?",
        "Describe your experience with agile development methodologies."
    ]
    
    # Build list of questions based on skills
    questions = []
    for skill in technical_skills:
        if skill in question_templates and len(questions) < 3:
            # Add 1 random question for this skill
            skill_questions = question_templates[skill]
            import random
            questions.append(random.choice(skill_questions))
    
    # Fill remaining slots with generic questions
    while len(questions) < 5:
        remaining = 5 - len(questions)
        questions.extend(generic_questions[:remaining])
        
    return questions

def generate_behavioral_questions(industry, role_level):
    """Generate behavioral interview questions based on industry and role level."""
    # Basic templates based on role level
    entry_level_questions = [
        "Tell me about a time when you had to learn a new skill quickly. How did you approach it?",
        "Describe a situation where you had to work with a difficult team member. How did you handle it?",
        "Give me an example of a time when you showed initiative.",
        "Tell me about a project you're particularly proud of. What was your role?",
        "How do you prioritize tasks when you have multiple deadlines?"
    ]
    
    mid_level_questions = [
        "Describe a time when you had to lead a project. What challenges did you face?",
        "Tell me about a time when you had to resolve a conflict within your team.",
        "Give me an example of a situation where you had to make a difficult decision.",
        "How have you handled criticism of your work?",
        "Describe a situation where you failed. What did you learn from it?"
    ]
    
    senior_questions = [
        "Tell me about a time when you had to make a strategic decision that affected the entire team.",
        "Describe your approach to mentoring junior team members.",
        "Give an example of how you've improved a process or system in your previous role.",
        "How do you stay current with industry trends and advancements?",
        "Describe a situation where you had to influence stakeholders to adopt your vision."
    ]
    
    # Select question set based on role level
    if "entry" in role_level.lower():
        return entry_level_questions
    elif "senior" in role_level.lower() or "management" in role_level.lower():
        return senior_questions
    else:
        return mid_level_questions

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
        
        # Provide suggestions without using Gemini
        st.subheader("Enhancement Suggestions")
        
        with st.expander("Skills Enhancement"):
            st.markdown("""
            ### Skills Recommendations
            - **Quantify your skills** with specific metrics and achievements
            - **Add skill levels** (e.g., Proficient, Advanced) to highlight strengths
            - **Group related skills** for better readability
            - Consider adding these in-demand skills relevant to your field:
              - Cloud platforms (AWS, Azure, GCP)
              - CI/CD tools (Jenkins, GitHub Actions)
              - Data analysis tools
              - Project management methodologies
            """)
            
        with st.expander("Format Improvements"):
            st.markdown("""
            ### Format Recommendations
            - Use a **clean, ATS-friendly layout** with clear section headings
            - Ensure **consistent formatting** of dates, job titles, and company names
            - Add a **professional summary** at the top highlighting your value proposition
            - Use **bullet points** instead of paragraphs for better readability
            - Include a dedicated **skills section** that's easy to scan
            - Limit resume to **1-2 pages** maximum
            """)
            
        with st.expander("Content Enhancements"):
            st.markdown("""
            ### Content Recommendations
            - **Begin bullet points with strong action verbs** (Developed, Implemented, Led)
            - **Add measurable achievements** with numbers and percentages
            - **Tailor your resume** to each job application
            - Include **relevant projects** with clear outcomes
            - Add **certifications and continuous learning** activities
            - Remove outdated or irrelevant experience
            """)
        
        with st.expander("ATS Optimization Tips"):
            st.markdown("""
            ### ATS Optimization
            - **Use keywords** from the job description naturally throughout your resume
            - Avoid **complex formatting** like tables, headers/footers, or text boxes
            - Use **standard section headings** (Experience, Education, Skills)
            - Avoid **graphics and images** that ATS can't read
            - Use **standard file formats** (.docx or .pdf)
            - **Spell check thoroughly** - ATS doesn't recognize misspelled keywords
            """)
        
        with st.expander("AI Rewrite Examples"):
            st.write("### Weak bullet point:")
            st.write("Responsible for managing a team.")
            
            st.write("### Improved version:")
            st.write("Led cross-functional team of 8 members, increasing department productivity by 27% and reducing costs by $45K annually.")
            
            st.write("### Weak bullet point:")
            st.write("Worked on software development.")
            
            st.write("### Improved version:")
            st.write("Developed and deployed 3 customer-facing web applications using React and Node.js, reducing page load time by 40% and increasing user engagement by 25%.")

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
                technical_questions = generate_technical_questions(resume_data)
            
            # Generate behavioral questions
            with st.spinner("Generating behavioral questions..."):
                behavioral_questions = generate_behavioral_questions(industry, role_level)
            
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
    st.title("Career Assistant")
    
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
        
        # Simple rule-based responses for career questions
        career_responses = {
            "resume": "To improve your resume, focus on quantifying achievements, using action verbs, and tailoring it to each job application. Keep it concise and highlight relevant skills.",
            "interview": "Prepare for interviews by researching the company, practicing common questions, preparing stories using the STAR method, and having thoughtful questions ready for the interviewer.",
            "skills": "To develop in-demand skills, consider taking online courses, working on personal projects, contributing to open source, and seeking mentorship in your field of interest.",
            "job search": "Effective job search strategies include networking, optimizing your LinkedIn profile, setting up job alerts, and customizing your applications for each position.",
            "salary": "When negotiating salary, research industry standards, highlight your unique value, practice your negotiation, and consider the entire compensation package including benefits.",
            "career change": "For a successful career change, identify transferable skills, gain relevant qualifications, network in your target industry, and be prepared to start at a different level.",
            "networking": "Build your professional network by attending industry events, joining professional associations, engaging on LinkedIn, and reaching out for informational interviews.",
            "linkedin": "Optimize your LinkedIn profile with a professional photo, compelling headline, detailed experience, and regular engagement with industry content.",
            "cover letter": "Write effective cover letters by addressing the hiring manager by name, highlighting relevant accomplishments, showing knowledge of the company, and explaining why you're the perfect fit.",
            "remote work": "Succeed in remote work by establishing a dedicated workspace, maintaining regular communication, setting clear boundaries, and focusing on results rather than hours worked."
        }
        
        # Default response if no keywords match
        response = "For specific career advice, please ask about topics like resume improvement, interview preparation, skill development, job searching strategies, or networking tips."
        
        # Check for keyword matches
        for keyword, resp in career_responses.items():
            if keyword in user_input.lower():
                response = resp
                break
                
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Force UI update
        st.rerun()

def main():

    # Add API Key Configuration to sidebar
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Google API Key (Optional)", type="password", 
                                    help="Enter your Google API Key for enhanced AI features. Leave blank to use basic features.")
    
    if api_key:
        st.sidebar.success("API Key configured! Enhanced AI features are available.")
    else:
        st.sidebar.info("Running in basic mode. Add API Key for enhanced features.")

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
        show_interview_preparation()

if __name__ == "__main__":
    main()
