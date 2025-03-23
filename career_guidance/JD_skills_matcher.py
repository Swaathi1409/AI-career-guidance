import spacy
import re
from spacy.matcher import Matcher, PhraseMatcher
from typing import Dict, List, Tuple, Any
import PyPDF2
import docx
import os
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeJobMatcher:
    def __init__(self, gemini_api_key: str):
        """Initialize with required models and APIs"""
        # Load spaCy models
        self.nlp = spacy.load("en_core_web_lg")
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Initialize patterns
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all matching patterns"""
        # Your existing patterns from ResumeParser
        self.skill_patterns = [
            [{"LOWER": {"IN": [
                # Programming Languages
                "python", "java", "javascript", "typescript", "c", "c++", "c#", "ruby",
                # Web Development
                "html", "css", "react", "angular", "vue", "node.js",
                # Databases
                "sql", "mysql", "mongodb", "postgresql",
                # Cloud & DevOps
                "aws", "azure", "docker", "kubernetes",
                # Data Science & AI/ML
                "machine learning", "deep learning", "tensorflow", "pytorch",
                # Additional skills can be added here
            ]}}]
        ]
        
        # Add patterns to matcher
        self.matcher.add("SKILLS", self.skill_patterns)

    def extract_skills_from_job(self, job_description: str) -> List[str]:
        """Extract required skills from job description using Gemini API and NLP"""
        try:
            # Use Gemini to extract skills
            prompt = f"""
            Extract a list of technical skills and competencies from this job description.
            Only return the skills as a comma-separated list:
            
            {job_description}
            """
            response = self.gemini_model.generate_content(prompt)
            gemini_skills = [skill.strip() for skill in response.text.split(',')]
            
            # Use spaCy for additional skill extraction
            doc = self.nlp(job_description)
            spacy_skills = [doc[start:end].text.lower() for match_id, start, end in self.matcher(doc)]
            
            # Combine and deduplicate skills
            all_skills = list(set(gemini_skills + spacy_skills))
            return all_skills
            
        except Exception as e:
            print(f"Error extracting job skills: {str(e)}")
            return []

    def calculate_match_score(self, resume_skills: List[str], job_skills: List[str]) -> Tuple[float, Dict]:
        """Calculate match score between resume skills and job requirements"""
        # Convert skills to lowercase for better matching
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        job_skills_lower = [skill.lower() for skill in job_skills]
        
        # Calculate direct matches
        matched_skills = set(resume_skills_lower) & set(job_skills_lower)
        
        # Calculate semantic similarity for non-exact matches
        non_matched_resume_skills = set(resume_skills_lower) - matched_skills
        non_matched_job_skills = set(job_skills_lower) - matched_skills
        
        # Use spaCy for semantic matching
        semantic_matches = {}
        for resume_skill in non_matched_resume_skills:
            for job_skill in non_matched_job_skills:
                similarity = self.nlp(resume_skill).similarity(self.nlp(job_skill))
                if similarity > 0.8:  # Threshold for semantic similarity
                    semantic_matches[resume_skill] = job_skill
        
        # Calculate final score
        total_required_skills = len(job_skills_lower)
        exact_matches = len(matched_skills)
        semantic_match_count = len(semantic_matches)
        
        if total_required_skills == 0:
            match_score = 0
        else:
            match_score = ((exact_matches + semantic_match_count) / total_required_skills) * 100
        
        # Prepare detailed matching report
        matching_details = {
            "exact_matches": list(matched_skills),
            "semantic_matches": semantic_matches,
            "missing_skills": list(set(job_skills_lower) - matched_skills - set(semantic_matches.values())),
            "match_score": round(match_score, 2)
        }
        
        return match_score, matching_details

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Parse resume and extract information"""
        try:
            # Read file
            text = self._read_file(file_path)
            
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract skills using matcher
            skills = []
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                if self.nlp.vocab.strings[match_id] == "SKILLS":
                    skills.append(doc[start:end].text)
            
            # Deduplicate skills
            skills = list(set(skills))
            
            return {
                "skills": skills,
                "full_text": text
            }
        except Exception as e:
            raise Exception(f"Error parsing resume: {str(e)}")

    def _read_file(self, file_path: str) -> str:
        """Read text from PDF or DOCX files"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return ' '.join(page.extract_text() for page in pdf_reader.pages)
                    
            elif file_extension == '.docx':
                doc = docx.Document(file_path)
                return ' '.join(paragraph.text for paragraph in doc.paragraphs)
                
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    def analyze_match(self, resume_path: str, job_description: str) -> Dict[str, Any]:
        """Main method to analyze resume-job match"""
        try:
            # Parse resume
            resume_data = self.parse_resume(resume_path)
            
            # Extract job skills
            job_skills = self.extract_skills_from_job(job_description)
            
            # Calculate match score
            match_score, matching_details = self.calculate_match_score(
                resume_data["skills"],
                job_skills
            )
            
            # Prepare analysis result
            analysis = {
                "resume_skills": resume_data["skills"],
                "required_skills": job_skills,
                "matching_details": matching_details,
                "match_score": match_score
            }
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Error analyzing match: {str(e)}")

if __name__ == "__main__":
    try:
        # Initialize matcher with your Gemini API key
        matcher = ResumeJobMatcher("AIzaSyDXhFXhfkzi_gkS0N2dZgLGtGZiO1FTGrM")
        
        # Use actual path to your resume file
        resume_path = "C:\\Users\\vaish\\Downloads\\pradeepaa_resume.pdf"  # Replace with your actual resume path
        
        # Example job description
        job_description = """
        We are seeking a Senior Python Developer with expertise in:
        - Python programming
        - Machine learning and AI
        - Web development with React
        - Database management (SQL, MongoDB)
        - Cloud platforms (AWS or Azure)
        """
        
        print("\nAnalyzing resume match...")
        print(f"Resume path: {resume_path}")
        print("\nJob Description:")
        print(job_description)
        
        # Analyze match
        analysis = matcher.analyze_match(resume_path, job_description)
        
        # Print detailed results
        print("\n=== MATCH ANALYSIS RESULTS ===")
        print(f"\nOverall Match Score: {analysis['match_score']}%")
        
        print("\nSkills Found in Resume:")
        for skill in analysis['resume_skills']:
            print(f"- {skill}")
            
        print("\nRequired Skills from Job:")
        for skill in analysis['required_skills']:
            print(f"- {skill}")
            
        print("\nExact Skill Matches:")
        for skill in analysis['matching_details']['exact_matches']:
            print(f"- {skill}")
            
        print("\nSemantic Matches (Similar Skills):")
        for resume_skill, job_skill in analysis['matching_details']['semantic_matches'].items():
            print(f"- {resume_skill} → {job_skill}")
            
        print("\nMissing Skills:")
        for skill in analysis['matching_details']['missing_skills']:
            print(f"- {skill}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if the resume file path is correct")
        print("2. Ensure the resume file is accessible")
        print("3. Verify the Gemini API key is valid")
        print("4. Make sure all required packages are installed")