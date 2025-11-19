"""
Stage 2: AI-Powered Resume Parser
This module takes extracted resume text and uses AI to parse it into structured data.
"""

import json
import os
from google.genai import types
from typing import List, Optional
import models
import config
from gemini_client import generate_content_resilient


def parse_resume_with_ai(resume_text):
    """
    Send resume text to an AI model and get structured information back.
    
    Args:
        resume_text (str): The plain text extracted from the resume
        
    Returns:
        dict: Structured resume information
    """
    print("Processing resume with AI model...")

    prompt = f"""Extract and return the structured resume information from the text below. Only use what is explicitly stated in the text and do not infer or invent any details.

    Resume text:
    {resume_text}
    """

    response = generate_content_resilient(
        prompt,
        model=config.GEMINI_MODEL_NAME,
        response_mime_type='application/json',
        response_schema=models.Resume,
        temperature=0.0,
    )
    return response.text
