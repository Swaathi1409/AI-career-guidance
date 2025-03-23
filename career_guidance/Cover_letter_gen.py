import google.generativeai as genai
from google.generativeai import GenerativeModel
import json
import PyPDF2
import docx
import os
import re

# 1️⃣ Read Resume File (Supports PDF/DOCX)
def read_resume_file(file_path):
    """Reads content from a resume file (PDF or DOCX)."""
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''.join([page.extract_text() for page in pdf_reader.pages])
                return text

        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return text

        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    except Exception as e:
        raise Exception(f"Error reading resume file: {str(e)}")


# 2️⃣ Parse Resume Text (Extracts Name, Skills, Experience, etc.)
def parse_resume_text(resume_text):
    """Parses resume text (improved placeholder - replace with real logic)."""
    parsed_data = {
        "name": "Extracted from resume",  # Default name
        "skills": [],
        "work_experience": [],
        "education": [],
        "achievements": []
    }

    # Name Extraction (Improve this!)
    name_match = re.search(r"(?:[A-Z][a-z]+\s?){1,3}", resume_text[:100])  # Basic name search
    if name_match:
        parsed_data["name"] = name_match.group(0).strip()

    sections = resume_text.split('\n\n')
    for section in sections:
        lines = section.split('\n')
        if "NAME" in section.upper():
            parsed_data["name"] = lines[0].strip()
        elif "SKILLS" in section.upper():
            parsed_data["skills"] = [skill.strip() for skill in section.split(',')]
        elif "EXPERIENCE" in section.upper():
            experience_entries = []
            current_entry = {}
            for line in lines[1:]: # Start from the second line (after "EXPERIENCE")
                if ":" in line: # Check for key-value pairs (e.g., "Title:", "Company:")
                    key, value = line.split(":", 1)
                    current_entry[key.strip().lower()] = value.strip()
                elif line.strip(): # Add description lines
                    if "description" not in current_entry:
                        current_entry["description"] = line.strip()
                    else:
                        current_entry["description"] += "\n" + line.strip()

            if current_entry: # Append the complete entry
                experience_entries.append(current_entry)

            parsed_data["work_experience"] = experience_entries

        elif "EDUCATION" in section.upper():
            education_entries = []
            current_entry = {}
            for line in lines[1:]:  # Start from the second line (after "EDUCATION")
                if ":" in line:  # Check for key-value pairs (e.g., "Degree:", "Institution:")
                    key, value = line.split(":", 1)
                    current_entry[key.strip().lower()] = value.strip()
                elif line.strip():  # Add description lines (if any)
                    if "details" not in current_entry:
                        current_entry["details"] = line.strip()
                    else:
                        current_entry["details"] += "\n" + line.strip()

            if current_entry:  # Append the complete entry
                education_entries.append(current_entry)

            parsed_data["education"] = education_entries 

        elif "ACHIEVEMENT" in section.upper():
            parsed_data["achievements"].append(section.strip())

    # Ensure all keys exist
    for key in ["skills", "work_experience", "education", "achievements"]:
        if key not in parsed_data:
            parsed_data[key] = []

    return parsed_data
# 3️⃣ Extract Job Title & Company Name from Job Description using Gemini API
def extract_job_details_using_gemini(model, job_description):
    """Extracts job title and company name using Gemini API."""
    prompt = f"""
    Extract the job position and company name from the following job description.
    If the company name is not explicitly mentioned, return 'Unknown Company'.

    Job Description:
    {job_description}

    Return the result in JSON format:
    {{
        "job_position": "Job Title Here",
        "company_name": "Company Name Here"
    }}
    """

    try:
        response = model.generate_content(prompt)
        extracted_details = json.loads(response.text)
        return extracted_details.get("job_position", "Unknown Position"), extracted_details.get("company_name", "Unknown Company")
    except Exception as e:
        return "Unknown Position", "Unknown Company"


# 4️⃣ Generate Cover Letter using Extracted Job Details & Resume Highlights
def generate_cover_letter(model, resume_highlights, job_description):
    """Generates a cover letter (with empty data checks)."""

    prompt = f"""
    Create a professional cover letter for {resume_highlights.get('name', 'N/A')} based on the following:

    Resume Highlights:
    - Skills: {', '.join(resume_highlights.get('skills', []))}
    - Experience:
    """

    experience = resume_highlights.get('work_experience', [])  # Get experience or empty list
    if experience:  # Check if experience is not empty
        for exp in experience:
            prompt += f"  - Title: {exp.get('title', 'N/A')}\n"
            prompt += f"    Company: {exp.get('company', 'N/A')}\n"
            prompt += f"    Description: {exp.get('description', 'N/A')}\n\n"
    else:
        prompt += "  No work experience provided.\n\n"  # Handle empty experience

    prompt += f"""
    - Education:
    """

    education = resume_highlights.get('education', [])  # Get education or empty list
    if education:
        for edu in education:
            prompt += f"  - Degree: {edu.get('degree', 'N/A')}\n"
            prompt += f"    Institution: {edu.get('institution', 'N/A')}\n"
            prompt += f"    Details: {edu.get('details', 'N/A')}\n\n"
    else:
        prompt += "  No education information provided.\n\n"

    prompt += f"""
    - Achievements: {json.dumps(resume_highlights.get('achievements', []), indent=2)}

    Job Description:
    {job_description}

    Instructions:
    ... 1. Tailor the cover letter to the job description.  Specifically, mention skills and experience that are directly relevant to the job requirements.
    2. Highlight relevant skills and achievements from the resume.  Quantify achievements whenever possible (e.g., "Increased sales by 15%").
    3. Use a professional and enthusiastic tone.  Express genuine interest in the role and the company.
    4. Keep the cover letter concise and easy to read. Aim for approximately 3-4 paragraphs.  Avoid overly long sentences or jargon.
    5. Proofread carefully before submitting. Ensure there are no grammatical errors or typos.
    6. Address the cover letter to the hiring manager if you know their name. Otherwise, use a general salutation like "Dear Hiring Team."
    7.  Mention the specific job title you are applying for.

    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating cover letter: {str(e)}"

# 5️⃣ Initialize Gemini API
def setup_gemini(api_key):
    """Initializes Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    return GenerativeModel('gemini-pro')


# 6️⃣ Main Function: Generates Cover Letter from Resume File
def generate_cover_letter_from_file(resume_file_path, job_description, api_key):
    """Main function to generate a cover letter from resume file."""
    try:
        # Read resume file
        resume_text = read_resume_file(resume_file_path)

        # Parse resume text
        parsed_resume = parse_resume_text(resume_text)

        # Initialize Gemini
        model = setup_gemini(api_key)

        # Extract highlights
        resume_highlights = {
    "name": parsed_resume.get("name", "Unknown"),
    "skills": parsed_resume.get("skills", []),
    "experience": parsed_resume.get("work_experience", []),  
    "education": parsed_resume.get("education", []),
    "achievements": parsed_resume.get("achievements", [])
}


        # Generate cover letter
        cover_letter = generate_cover_letter(model, resume_highlights, job_description)

        return cover_letter

    except Exception as e:
        return f"Error in cover letter generation process: {str(e)}"


# 7️⃣ Example Usage
if __name__ == "__main__":
    # File path to your resume
    resume_path = "C:\\Users\\vaish\\Downloads\\pradeepaa_resume.pdf"  # or .docx

    # Sample job description (This will come from frontend)
    sample_job = """
    We are seeking a Senior Data Scientist with strong Python skills and 
    machine learning experience. The ideal candidate will have a master's 
    degree in Computer Science or related field. Join us at OpenAI!
    """

    # Your Gemini API key
    API_KEY = "AIzaSyDXhFXhfkzi_gkS0N2dZgLGtGZiO1FTGrM"

    # Generate cover letter
    cover_letter = generate_cover_letter_from_file(resume_path, sample_job, API_KEY)
    print(cover_letter)
