# Career_Guidance
Career Guidance platform with enriched resume booster and interviewer prep by team HackHerWay developed in SNUC hackathon
This project is an AI-powered career guidance system designed to optimize job matching, improve resumes, generate cover letters, and provide real-time feedback for video interviews. It integrates multiple AI techniques, including NLP and sentiment analysis, to assist job seekers in their career journey.

## Features

### 1. **Resume Job Matching Optimizer**
   - Uses **cosine similarity scoring** to match resumes with job descriptions.
   - Helps job seekers tailor their resumes to specific job listings.

### 2. **Resume Processing & Enhancement**
   - **Text to LaTeX to PDF**: Converts resume text into a professionally formatted LaTeX-based PDF.
   - **Cover Letter Generator**: Automatically generates a cover letter based on the resume.
   - **Cover Letter to Resume Generator**: Extracts key details from a cover letter to create a structured resume.

### 3. **Career Guidance & Testing**
   - AI-driven **Career Guidance Test** to assess skills, interests, and career paths.
   - Provides personalized career advice based on test results.


### 4. **Chatbot - Career Assistant**
   - AI-powered chatbot for career advice, job search strategies, and resume optimization.
   - Provides real-time suggestions and resources.

### 5. **Video Interview Sentiment Analysis**
   - **Real-time AI feedback system** for video interviews.
   - Analyzes facial expressions, tone of voice, and speech patterns.
   - Provides **subtle real-time cues** (e.g., color indicators) to improve delivery.
   - Post-interview analytics on **body language, speaking pace, and engagement levels**.
   - Uses **Whisper AI** for speech-to-text and sentiment analysis.

## Technologies Used
- **Natural Language Processing (NLP)**: Resume parsing, similarity scoring, chatbot.
- **Machine Learning & AI**: Sentiment analysis, speech recognition.
- **Whisper AI**: Speech-to-text processing.
- **LaTeX & PDF Generation**: Resume and cover letter formatting.
- **Google Chrome Extension Development**: Resume download automation.
- **Computer Vision & Facial Analysis**: Real-time interview feedback.

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/career-guidance-ai.git
   cd career-guidance-ai
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the chatbot or video analysis tool:
   ```sh
   python main.py
   ```
4. Install and enable the Chrome extension for resume downloads.

