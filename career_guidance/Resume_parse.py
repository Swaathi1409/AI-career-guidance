import spacy
import re
from spacy.matcher import Matcher
from typing import Dict, List, Tuple, Any
import PyPDF2
import docx
import os

class ResumeParser:
    def __init__(self):
        """Initialize the resume parser with necessary NLP models and patterns"""
        self.nlp = spacy.load("en_core_web_lg")
        self.matcher = Matcher(self.nlp.vocab)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all matching patterns for different resume sections"""
        # Education patterns
        self.education_patterns = [
            # Bachelor's/Master's degree patterns
            [{"LOWER": {"IN": ["bachelor", "master", "b.tech", "m.tech", "b.e.", "m.e."]}},
             {"LOWER": {"IN": ["of", "in"]}, "OP": "?"},
             {"POS": "PROPN", "OP": "+"}],
            # Degree with specialization
            [{"LOWER": {"IN": ["bachelor", "master", "b.tech", "m.tech"]}},
             {"LOWER": {"IN": ["of", "in"]}, "OP": "?"},
             {"LOWER": "computer"},
             {"LOWER": "science"}],
            # Simple degree mentions
            [{"LOWER": {"IN": ["b.tech", "m.tech", "phd", "mba", "b.sc", "m.sc"]}},
             {"POS": "PROPN", "OP": "*"}]
        ]

        # Skills patterns
        self.skill_patterns = [
            # Technical skills
            [{"LOWER": {"IN": [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "ruby", "swift", "kotlin", "go", "rust", "dart", "scala",
    
    # Web Development
    "html", "css", "bootstrap", "tailwind", "react", "angular", "vue", "svelte", "next.js", "nuxt.js", "redux",
    
    # Backend Development
    "node.js", "express", "fastapi", "flask", "django", "spring", "spring boot", "hibernate", "laravel", "rails", "asp.net",
    
    # Databases
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra", "dynamodb", "neo4j", "firebase",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "digitalocean", "docker", "kubernetes", "terraform", "ansible", "jenkins", "git", "github", "gitlab",
    
    # Data Science & AI/ML
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "matplotlib", "seaborn", "nlp", "openai", "huggingface",
    "opencv", "computer vision", "deep learning", "machine learning", "data visualization",
    
    # APIs & Microservices
    "rest api", "graphql", "grpc", "soap", "openapi", "swagger", "postman", "api testing",
    
    # Cybersecurity
    "penetration testing", "owasp", "burpsuite", "firewalls", "cryptography", "ethical hacking", "security compliance",
    
    # Mobile Development
    "android", "ios", "flutter", "react native", "swiftui", "jetpack compose",
    
    # Software Testing
    "unit testing", "selenium", "pytest", "junit", "cypress", "jest", "automation testing",
    
    # Soft Skills
    "leadership", "communication", "teamwork", "problem solving", "critical thinking", "time management",
    "adaptability", "creativity", "work ethic", "emotional intelligence", "negotiation", "public speaking"
]}}]
,
            # Compound skills
            [{"LOWER": "machine"}, {"LOWER": "learning"}],
            [{"LOWER": "deep"}, {"LOWER": "learning"}],
            [{"LOWER": "artificial"}, {"LOWER": "intelligence"}],
            [{"LOWER": "natural"}, {"LOWER": "language"}, {"LOWER": "processing"}]
        ]
        self.api_patterns = [
            [{"LOWER": "rest"}, {"LOWER": "api"}],   # Matches "REST API"
            [{"LOWER": "restful"}, {"LOWER": "api"}],  # Matches "RESTful API"
            [{"LOWER": "fastapi"}]  # Matches "FastAPI"
        ]
        
        # Add the patterns to matcher
        self.matcher.add("APIS", self.api_patterns)
        # Add patterns to matcher
        self.matcher.add("EDUCATION", self.education_patterns)
        self.matcher.add("SKILLS", self.skill_patterns)

    def _extract_cgpa(self, text: str) -> Dict[str, str]:
        """Extract CGPA and academic scores from text"""
        scores = {}
        
        # CGPA patterns
        cgpa_patterns = [
            r'CGPA[\s:]+([0-9.]+)(?:/\d+)?',
            r'GPA[\s:]+([0-9.]+)(?:/\d+)?',
            r'Grade Point Average[\s:]+([0-9.]+)',
            r'CGPA\s+of\s+([0-9.]+)',
            r'with\s+CGPA\s+([0-9.]+)'
        ]
        
        # School score patterns
        school_patterns = {
            '10th': [
                r'(?:10th|X|Tenth|SSC)[\s:]+([0-9.]+\s*%?)',
                r'(?:10th|X|Tenth|SSC)\s+Grade[\s:]+([0-9.]+\s*%?)'
            ],
            '12th': [
                r'(?:12th|XII|Twelfth|HSC)[\s:]+([0-9.]+\s*%?)',
                r'(?:12th|XII|Twelfth|HSC)\s+Grade[\s:]+([0-9.]+\s*%?)'
            ]
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

    def _clean_and_deduplicate(self, items: List[str]) -> List[str]:
        """Clean and deduplicate items while preserving the most complete entries"""
        if not items:
            return []
            
        # Convert to set to remove exact duplicates
        items = list(set(items))
        
        # Sort by length (longest first) to keep most complete entries
        items.sort(key=len, reverse=True)
        
        # Remove items that are substrings of other items
        result = []
        for item in items:
            if not any(item != other and item.lower() in other.lower() for other in items):
                result.append(item)
        
        return sorted(result)

    def parse_resume(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Main method to parse resume and extract information
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Tuple containing formatted output string and structured data dictionary
        """
        try:
            # Read file
            text = self._read_file(file_path)
            
            # Process with spaCy
            doc = self.nlp(text)
            
            # Initialize extracted information
            extracted_info = {
                "education": [],
                "skills": [],
                "academic_scores": {},
                "projects": []
            }
            
            # Find matches using spaCy matcher
            matches = self.matcher(doc)
            
            # Process matches
            for match_id, start, end in matches:
                span = doc[start:end]
                category = self.nlp.vocab.strings[match_id]
                if category == "EDUCATION":
                    extracted_info["education"].append(span.text)
                elif category == "SKILLS":
                    extracted_info["skills"].append(span.text)
            
            # Clean and deduplicate lists
            extracted_info["education"] = self._clean_and_deduplicate(extracted_info["education"])
            extracted_info["skills"] = self._clean_and_deduplicate(extracted_info["skills"])
            
            # Extract academic scores
            extracted_info["academic_scores"] = self._extract_cgpa(text)
            
            
            # Generate formatted output
            output = self._format_output(extracted_info)
            
            return output, extracted_info
            
        except Exception as e:
            raise Exception(f"Error parsing resume: {str(e)}")

    def _read_file(self, file_path: str) -> str:
        """Read and extract text from various file formats"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ' '.join(page.extract_text() for page in pdf_reader.pages)
                    
            elif file_extension == '.docx':
                doc = docx.Document(file_path)
                text = ' '.join(paragraph.text for paragraph in doc.paragraphs)
                
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            return text
            
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    def _format_output(self, info: Dict[str, Any]) -> str:
        """Format extracted information into readable string"""
        output = []
        
        # Education with scores
        if info["education"]:
            output.append("\nEDUCATION:")
            for edu in info["education"]:
                output.append(f"- {edu}")
                if "cgpa" in info["academic_scores"]:
                    output.append(f"  CGPA: {info['academic_scores']['cgpa']}")
            
            # Add school scores if available
            if "10th_score" in info["academic_scores"]:
                output.append(f"- 10th Score: {info['academic_scores']['10th_score']}")
            if "12th_score" in info["academic_scores"]:
                output.append(f"- 12th Score: {info['academic_scores']['12th_score']}")
        
        # Skills
        if info["skills"]:
            output.append("\nSKILLS:")
            for skill in info["skills"]:
                output.append(f"- {skill}")
        
        # Projects
        if info["projects"]:
            output.append("\nPROJECTS:")
            for project in info["projects"]:
                output.append(f"- {project['name']}")
                if project.get('technologies'):
                    output.append(f"  Technologies: {', '.join(project['technologies'])}")
                if project.get('description'):
                    output.append(f"  Description: {project['description']}")
        
        return "\n".join(output)

# Example usage
if __name__ == "__main__":
    try:
        # Initialize parser
        parser = ResumeParser()
        
        # Specify resume file path
        resume_path = "E:\\fdrive\\3rd sem\\cv\\pradeepaa_resume.pdf"
        
        # Parse resume
        output, parsed_data = parser.parse_resume(resume_path)
        
        # Print results
        print("Extracted Resume Information:")
        print(output)
        
    except Exception as e:
        print(f"Error: {str(e)}")