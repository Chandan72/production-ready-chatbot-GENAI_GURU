from dotenv import load_dotenv
load_dotenv()
import os

google_api_key=os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.") 
