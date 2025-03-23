import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import json
from typing import Dict, List
import os
from dotenv import load_dotenv
from career_guidance.Resume_parse import ResumeParser

class InterviewManager:
    def __init__(self):
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

    def generate_questions(self, resume_data: Dict) -> List[str]:
        """Generate interview questions based on parsed resume content."""
        if not isinstance(resume_data, dict):
            print("Error: Resume data is not a dictionary.")
            return []  # Return an empty list to avoid further errors

        skills_str = ", ".join(resume_data.get("skills", ["Not specified"]))
        education_str = "\n".join(resume_data.get("education", ["Not specified"]))
        academic_scores_str = json.dumps(resume_data.get("academic_scores", {}), indent=2)

        prompt = f"""
        Based on the candidate's profile:

        Skills: {skills_str}
        Education: {education_str}
        Academic Performance: {academic_scores_str}

        Generate 5 technical interview questions that:
        1. Focus on the candidate's core skills
        2. Include practical problem-solving scenarios
        3. Assess both theoretical knowledge and practical application
        4. Progress from basic concepts to more complex scenarios
        5. Cover different aspects of their technical expertise

        Format each question clearly and include any necessary context.
        """

        response = self.model.generate_content(prompt)
        questions = [q.strip() for q in response.text.split("\n") if q.strip()]
        return questions

    def conduct_interview(self, questions: List[str]):
        """Conduct the interview using speech recognition and text-to-speech."""
        responses = []

        print("\nStarting Interview Session...")
        for i, question in enumerate(questions[:5], 1): # Limit to 5 questions
            print(f"\nQuestion {i}:")
            print(question)

            self.engine.say(question)
            self.engine.runAndWait()

            with sr.Microphone() as source:
                print("\nListening for response...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

                try:
                    audio = self.recognizer.listen(source, timeout=30, phrase_time_limit=120)
                    response = self.recognizer.recognize_google(audio)
                    print(f"Candidate's response: {response}")

                    responses.append({
                        "question": question,
                        "response": response
                    })
                except sr.WaitTimeoutError:
                    print("No response detected - moving to next question")
                    responses.append({"question": question, "response": "No response provided"})
                except sr.UnknownValueError:
                    print("Could not understand audio - moving to next question")
                    responses.append({"question": question, "response": "Response unclear"})
                    continue # Skip to next iteration
                except sr.RequestError as e:
                    print(f"Error with speech recognition service: {e}")
                    responses.append({"question": question, "response": f"Technical error during response: {e}"})
                    continue # Skip to next iteration

        return responses

    def analyze_responses(self, responses: List[Dict], resume_data: Dict) -> Dict:
        """Analyze interview responses using Gemini API."""
        if not isinstance(resume_data, dict):
            print("Error: Resume data is not a dictionary.")
            return {"feedback": "Resume data error", "responses": responses}

        skills_str = ", ".join(resume_data.get("skills", ["Not specified"]))

        analysis_prompt = f"""
        Analyze the following interview responses for a candidate with expertise in: {skills_str}

        Interview Transcript:
        {json.dumps(responses, indent=2)}

        Provide a detailed evaluation covering:
        1. Technical Proficiency
            - Accuracy of technical explanations
            - Depth of knowledge in claimed skills
            - Problem-solving approach

        2. Communication Skills
            - Clarity of explanations
            - Structured thinking
            - Technical vocabulary usage

        3. Areas of Strength
            - Notable demonstrations of expertise
            - Particularly good responses

        4. Areas for Improvement
            - Knowledge gaps identified
            - Suggested focus areas

        5. Overall Assessment
            - Interview performance summary
            - Fit for technical roles
            - How well the candidate's answer matches with the skills mentioned in the resume.

        6. Follow up questions.
            - Based on the previous answers, create 2 follow up questions.
        """

        analysis = self.model.generate_content(analysis_prompt)
        return {
            "feedback": analysis.text,
            "responses": responses
        }

if __name__ == "__main__":
    try:
        interview_system = InterviewManager()
        parser = ResumeParser()
        resume_path = "E:\\fdrive\\3rd sem\\cv\\pradeepaa_resume.pdf" # Replace with your resume path

        _, resume_data = parser.parse_resume(resume_path)

        questions = interview_system.generate_questions(resume_data)

        responses = interview_system.conduct_interview(questions)

        feedback = interview_system.analyze_responses(responses, resume_data)

        print("\nInterview Analysis:")
        print(feedback["feedback"])

    except Exception as e:
        print(f"Error during interview: {str(e)}")