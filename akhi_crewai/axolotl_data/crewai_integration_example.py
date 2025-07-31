
# Example: Creating a CrewAI Agent with Fine-tuned Qwen

from crewai import Agent, Task, Crew
from integrate_fine_tuned_model import FineTunedQwenIntegration

class IslamicContentAgent:
    def __init__(self):
        self.model_integration = FineTunedQwenIntegration()
    
    def create_agent(self):
        return Agent(
            role="Islamic Content Specialist",
            goal="Provide authentic Islamic guidance and content",
            backstory="""You are an Islamic scholar with deep knowledge of Quran, 
                        Hadith, and Islamic jurisprudence. You provide accurate, 
                        authentic, and practical Islamic guidance.""",
            verbose=True,
            allow_delegation=False
        )
    
    def generate_content(self, query: str):
        return self.model_integration.islamic_content_agent(query)

# Usage example:
# islamic_agent = IslamicContentAgent()
# agent = islamic_agent.create_agent()
# result = islamic_agent.generate_content("What is the importance of charity in Islam?")
