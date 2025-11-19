import logging
import io
supabase_utils = None
import config
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
import json
import pdf_generator 
import re
import asyncio 
genai = None
types = None
from gemini_client import generate_content_resilient
from models import (
    Education, Experience, Project, Certification, Links, Resume,
    SummaryOutput, SkillsOutput, ExperienceListOutput, SingleExperienceOutput,
    ProjectListOutput, SingleProjectOutput, ValidationResponse
)
import time
import pdfplumber
import sys
import os

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Gemini Client ---
client = None

# --- Helper: Deduplicate skills (case-insensitive) ---
def deduplicate_skills(skills: List[str]) -> List[str]:
    """Remove duplicate skills (case-insensitive), keeping first occurrence."""
    seen = {}
    result = []
    for skill in skills:
        key = skill.lower().strip()
        if key not in seen:
            seen[key] = True
            result.append(skill)
    return result

# --- Helper: Extract technologies from project description ---
def extract_technologies_from_description(description: str, existing_techs: List[str] = None) -> List[str]:
    """
    Extract technology keywords from project description.
    Returns combined list of existing + newly found technologies.
    """
    if not description:
        return existing_techs or []
    
    # Comprehensive technology keyword mapping
    tech_keywords = {
        'python': 'Python',
        'scikit-learn': 'Scikit-Learn',
        'sklearn': 'Scikit-Learn',
        'xgboost': 'XGBoost',
        'random forest': 'Random Forest',
        'svm': 'SVM',
        'support vector machine': 'SVM',
        'pyside6': 'PySide6',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'plotly': 'Plotly',
        'sql': 'SQL',
        'mysql': 'MySQL',
        'postgresql': 'PostgreSQL',
        'sqlite': 'SQLite',
        'power bi': 'Power BI',
        'powerbi': 'Power BI',
        'tableau': 'Tableau',
        'excel': 'Excel',
        'git': 'Git',
        'github': 'GitHub',
        'jupyter': 'Jupyter',
        'notebook': 'Jupyter Notebook',
        'flask': 'Flask',
        'django': 'Django',
        'fastapi': 'FastAPI',
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'pytorch': 'PyTorch',
        'opencv': 'OpenCV',
        'nlp': 'NLP',
        'natural language processing': 'NLP',
        'deep learning': 'Deep Learning',
        'neural network': 'Neural Networks',
        'logistic regression': 'Logistic Regression',
        'linear regression': 'Linear Regression',
        'decision tree': 'Decision Trees',
        'k-means': 'K-Means',
        'clustering': 'Clustering',
        'classification': 'Classification',
        'regression': 'Regression',
        'time series': 'Time Series Analysis',
        'statistics': 'Statistical Analysis',
        'statistical': 'Statistical Analysis',
        'a/b test': 'A/B Testing',
        'hypothesis test': 'Hypothesis Testing',
        'data visualization': 'Data Visualization',
        'etl': 'ETL',
        'data pipeline': 'Data Pipelines',
        'aws': 'AWS',
        'azure': 'Azure',
        'gcp': 'GCP',
        'docker': 'Docker',
        'kubernetes': 'Kubernetes',
        'spark': 'Apache Spark',
        'hadoop': 'Hadoop',
        'airflow': 'Apache Airflow',
    }
    
    found_techs = set(existing_techs or [])
    desc_lower = description.lower()
    
    for keyword, proper_name in tech_keywords.items():
        if keyword in desc_lower:
            found_techs.add(proper_name)
    
    # Remove duplicates (e.g., both "Scikit-Learn" from sklearn and scikit-learn)
    return list(found_techs)

# --- LLM Personalization Function ---
def extract_json_from_text(text: str) -> str:
    """
    Extracts and returns the first valid JSON string found in the text.
    """
    fenced_match = re.search(r"```(?:json)?\s*(\[\s*{.*?}\s*\]|\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        json_candidate = fenced_match.group(1).strip()
    else:
        loose_match = re.search(r"(\[\s*{.*?}\s*\]|\[.*?\]|\{.*?\})", text, re.DOTALL)
        if loose_match:
            json_candidate = loose_match.group(1).strip()
        else:
            json_candidate = text.strip()

    try:
        parsed = json.loads(json_candidate)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to extract valid JSON: {e}\nRaw candidate:\n{json_candidate}")

async def personalize_section_with_llm(
    section_name: str,
    section_content: Any,
    full_resume: Resume,
    job_details: Dict[str, Any]
    ) -> Any:
    """
    Uses Gemini to personalize a specific section of the resume for the given job.
    """
    if not section_content:
        logging.warning(f"Skipping personalization for empty section: {section_name}")
        return section_content

    output_model_map = {
        "summary": (SummaryOutput, "summary"),
        "skills": (SkillsOutput, "skills"),
        "experience": (SingleExperienceOutput, "experience"),
        "projects": (SingleProjectOutput, "project"),
    }

    if section_name not in output_model_map:
        logging.error(f"Unsupported section_name for LLM personalization: {section_name}")
        return section_content

    OutputModel, output_key = output_model_map[section_name]

    resume_context_dict = full_resume.model_dump(exclude={section_name})
    resume_context = json.dumps(resume_context_dict, indent=2)

    if isinstance(section_content, list) and section_content and hasattr(section_content[0], 'model_dump'):
        serializable_section_content = [item.model_dump() for item in section_content]
    else:
        serializable_section_content = section_content

    prompts = []

    prompt_intro = f"""
    **Task:** Enhance the specified resume section for the target job application.

    **Target Job**
    - Title: {job_details['job_title']}
    - Company: {job_details['company']}
    - Seniority Level: {job_details['level']}
    - Job Description: {job_details['description']}

    ---

    **Full Resume Context (excluding the section being edited):**
    {resume_context}

    **Resume Section to Enhance:** {section_name}
    """

    system_prompt = f"""
    You are an expert resume writer and a precise JSON generation assistant.
    Your primary function is to enhance specified sections of a resume to better align with a target job description.

    **CRITICAL OUTPUT REQUIREMENTS:**
    1.  You MUST ALWAYS output a single, valid JSON object.
    2.  Your entire response MUST be *only* the JSON object.
    3.  Do NOT include any introductory text, explanations, apologies, or markdown formatting.

    **CORE RESUME WRITING PRINCIPLES:**
    1.  **Truthfulness**: NEVER invent information, skills, projects, job titles, or responsibilities not in the original resume.
    2.  **Strategic Emphasis**: Reframe and emphasize existing facts to match the job requirements.
    3.  **Relevance**: Highlight aspects of the candidate's actual experience that align with the target job.
    4.  **Authenticity**: All enhancements must be factually grounded in the provided resume materials.
    """

    specific_instructions = ""

    if(section_name == "summary"):
        specific_instructions = f"""
        **Original Content of This Section:**
        {json.dumps(serializable_section_content, indent=2)}

        ---
        **Instructions:**
        - Rewrite the summary to be concise, impactful, and highly relevant to the Target Job.
        - Keep it between **60 and 90 words**; use short, direct sentences.
        - **CRITICAL AUTHENTICITY RULES:**
          * If the candidate is transitioning careers (e.g., from semiconductor engineering to data science), acknowledge this authentically
          * Use bridge framing: "[New target role] with background in [actual experience domain], bringing transferable skills in [relevant skills]"
          * Example: "Data Science graduate transitioning from quality engineering, bringing analytical expertise in statistical process control, Python, and SQL"
          * NEVER claim years of experience in a field the candidate hasn't actually worked in
          * Preserve the candidate's actual career stage (graduate, entry-level, 2 years experience, etc.)
        - Highlight 2-3 key qualifications from the resume that align with the job description
        - Use keywords from the job description where they match actual experience
        - Focus on transferable skills: analytical thinking, statistical methods, programming, problem-solving
        ---
        **Expected JSON Output Structure:** {{"summary": "Data Science graduate with..."}}
        """
        prompt = prompt_intro + specific_instructions
        prompts.append(prompt)

    elif(section_name == "experience"):
        for exp_item_content in serializable_section_content:
            specific_instructions = f"""
             **Original Content of This Specific Experience Item:**
            {json.dumps(exp_item_content, indent=2)}

            ---
            **Instructions for this experience item:**
            - Enhance ONLY the 'description' field; all other fields MUST remain UNCHANGED.
            - **FOR DATA-FOCUSED JOBS: Reframe bullets to emphasize data/analytical aspects:**
              * Lead with data scale: "Analyzed X records/samples/tests..."
              * Specify tools: "...using Python/SQL/Excel/Power BI..."
              * Highlight insights: "...identifying Y patterns that led to Z improvement"
              * Quantify impact: percentages, time saved, efficiency gains
              * Example transformation:
                ❌ "Coordinate QRA testing and manage data collection"
                ✅ "Manage quality data pipeline collecting 500+ test records daily, performing root cause analysis using statistical methods to identify top failure modes and reduce debug time by 20%"
            - Each bullet MUST follow: **Action Verb + Data/Tool + Task + Quantified Outcome**
            - Maintain 100% factual accuracy - only reframe emphasis, don't invent responsibilities
            - If the original role isn't data-heavy, focus on: analysis, problem-solving, process improvement, metrics
            - Format as bullet points separated by newline characters (`\n`)
            - Aim for **4–5 bullets** (each 1–2 lines)
            ---
            **Expected JSON Output Structure:** {{"experience": {{"job_title": "Original Job Title", "company": "Original Company", "dates": "Original Dates", "description": "• Enhanced bullet 1\n• Enhanced bullet 2...", "location": "Original Location"}}}}
            """ 
            prompt = prompt_intro + specific_instructions
            prompts.append(prompt)

    elif(section_name == "projects"):
        for project_item_content in serializable_section_content:
            specific_instructions = f"""
            **Original Content of This Specific Project Item:**
            {json.dumps(project_item_content, indent=2)}

            ---
            **Instructions for this project item:**
            - Enhance ONLY the 'description' field; all other fields MUST remain UNCHANGED.
            - Use this structure:
              • **Problem/Goal**: What was the project trying to solve?
              • **Technical Approach**: Methods, algorithms, techniques used (be specific)
              • **Tools & Technologies**: Explicitly mention libraries/frameworks used (Python, Scikit-Learn, SQL, etc.)
              • **Results**: Metrics, accuracy, performance improvements, outcomes
            - For ML projects: Include model types, evaluation metrics (accuracy, F1, ROC-AUC), dataset size
            - Lead with impact: "Developed [X] achieving [Y metric] by [method]"
            - Format as bullet points separated by newline characters (`\n`)
            - Aim for **3 bullets** (each 1–2 lines) focusing on technical depth and outcomes
            - DO NOT invent technologies not mentioned in the original project
            ---
            **Expected JSON Output Structure:** {{"project": {{"name": "Original Name", "technologies": ["Tech1", "Tech2"], "description": "• Enhanced bullet...", "link": "Original Link"}}}}
            """
            prompt = prompt_intro + specific_instructions 
            prompts.append(prompt)

    elif(section_name == "skills"):
        specific_instructions = f"""
        **Original Content of This Section (Candidate's Skills):**
        {json.dumps(serializable_section_content, indent=2)}

        ---
        **Instructions for Generating the Curated Skills List:**

        **1. Identify Candidate's Actual Skills:**
        - Review the 'Full Resume Context' (summary, experience, projects, certifications)
        - Review the 'Original Content of This Section'
        - Compile ALL skills explicitly mentioned in these materials
        - **CRITICAL: DO NOT infer skills. If "Python" isn't written, don't include it**

        **2. Select and Prioritize for the Target Job:**
        - From the candidate's actual skills, select those most relevant to the job description
        - **Output MUST contain 10-14 skills** (this is the optimal range for ATS and readability)
        - Prioritize skills mentioned in the job description that match the candidate's skills
        - Include a mix of:
          * Technical skills (programming languages, tools)
          * Domain skills (data analysis, machine learning, statistical methods)
          * Soft skills if explicitly mentioned (communication, problem-solving)
        - Use specific tool names when possible (e.g., "Power BI" not just "visualization")
        - Remove generic duplicates (e.g., if "SQL" and "MySQL" both present, keep both if both are actually used)

        **3. Formatting:**
        - Use proper capitalization (Python, not python)
        - Be specific (Scikit-Learn, not sklearn)
        - Group related skills logically

        ---
        **Expected JSON Output Structure:** {{"skills": ["Python", "SQL", "Power BI", "Statistical Analysis", "Machine Learning", "Data Visualization", "Excel", "Git", "Problem Solving", "Process Optimization", "Quality Management", "Root Cause Analysis"]}}
        """
        prompt = prompt_intro + specific_instructions 
        prompts.append(prompt)

    logging.info(f"Number of prompts for {section_name}: {len(prompts)}")

    global genai, types, client
    if genai is None or types is None:
        from google import genai as _genai
        from google.genai import types as _types
        genai, types = _genai, _types
    if client is None:
        client = None

    responses = []
    for prompt in prompts:
        logging.info(f"Sending prompt to Gemini for section: {section_name}")

        attempts = 0
        while attempts < 2:
            attempts += 1
            try:
                response = generate_content_resilient(
                    prompt,
                    model=config.GEMINI_MODEL_NAME,
                    temperature=0.2,
                    system_instruction=system_prompt,
                    response_mime_type='application/json',
                    response_schema=OutputModel,
                )
                llm_output = response.text.strip()
                logging.info(f"Received response from Gemini for section: {section_name}")
                try:
                    parsed_response_model = OutputModel.model_validate_json(llm_output)
                    responses.append(parsed_response_model)
                    break
                except ValidationError as e:
                    logging.error(f"Failed to validate LLM JSON output for {section_name}: {e}")
                    logging.error(f"LLM Raw Output: {llm_output}")
                    return section_content
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse LLM JSON output for {section_name}: {e}")
                    logging.error(f"LLM Raw Output: {llm_output}")
                    return section_content
            except Exception as e:
                logging.error(f"Error calling Gemini for section {section_name}: {e}")
                if attempts < 2:
                    logging.info(f"Retrying section {section_name} after delay...")
                    time.sleep(config.GEMINI_REQUEST_DELAY_SECONDS)
                else:
                    return section_content

    logging.info(f"Received {len(responses)} responses for section: {section_name}")

    if(section_name == "summary"):
        return getattr(responses[0], output_key)
    elif(section_name == "skills"):
        return getattr(responses[0], output_key)
    elif(section_name == "experience"):
        experience_list = []
        for response in responses:
            experience_list.append(getattr(response, output_key))
        return experience_list
    elif(section_name == "projects"):
        project_list = []
        for response in responses:
            project_list.append(getattr(response, output_key))
        return project_list

async def validate_customization(
    section_name: str, 
    original_content: Any, 
    customized_content: Any, 
    full_original_resume: Resume, 
    job_details: Dict[str, Any]
    ) -> (bool, str):

    resume_context_dict = full_original_resume.model_dump(exclude={section_name})
    resume_context = json.dumps(resume_context_dict, indent=2)

    if isinstance(original_content, list) and original_content and hasattr(original_content[0], 'model_dump'):
        serializable_original_content = [item.model_dump() for item in original_content]
    else:
        serializable_original_content = original_content 

    if isinstance(customized_content, list) and customized_content and hasattr(customized_content[0], 'model_dump'):
        serializable_customized_content = [item.model_dump() for item in customized_content]
    else:
        serializable_customized_content = customized_content 

    system_prompt=f"""
    You are a meticulous Resume Fact-Checker ensuring authenticity and truthfulness.

    **CRITICAL OUTPUT REQUIREMENTS:**
    1.  Output a single, valid JSON object only
    2.  No explanatory text, markdown, or formatting outside the JSON
    3.  JSON must contain: "is_valid" (boolean) and "reason" (string)

    **VALIDATION PRINCIPLES:**
    - Verify NO fabricated information (skills, experiences, achievements, metrics)
    - Allow strategic reframing and emphasis of existing facts
    - Allow bridge framing for career transitions if supported by education/experience
    - Reject role title fabrications (e.g., "Quality Technician" → "Data Scientist" with no supporting evidence)
    - Accept emphasis shifts (e.g., "Quality Technician analyzing test data" if analysis was part of role)
    """

    user_prompt = f"""
    **Task:** Validate that customization is authentic and doesn't fabricate information.

    **Target Job:** {job_details['job_title']} at {job_details['company']}

    **Original Full Resume Context:**
    {resume_context}

    **Original Section ("{section_name}"):**
    {json.dumps(serializable_original_content, indent=2)}

    **Customized Section ("{section_name}"):**
    {json.dumps(serializable_customized_content, indent=2)}

    ---
    **Validation Criteria:**
    1.  **No Fabrication**: Are there new skills, tools, achievements, or metrics not in the original?
    2.  **Role Authenticity**: Does customization claim a different primary job title without support?
    3.  **Data Truthfulness**: Are quantified metrics (percentages, numbers) invented or inflated?
    4.  **Skill Claims**: Are new technical skills mentioned that aren't in original resume or projects?

    **Acceptable Changes:**
    - Reframing emphasis (e.g., "managed data collection" → "analyzed 500+ test records")
    - Bridge framing for career transition (e.g., "Data Science graduate with semiconductor background")
    - Using synonyms or stronger verbs for existing accomplishments
    - Highlighting relevant aspects of actual experience

    **Unacceptable Changes:**
    - Inventing new responsibilities not in original role
    - Claiming expertise in tools never mentioned
    - Fabricating metrics or achievements
    - Changing job title to a completely different role

    Provide validation result as JSON.
    ---
    **Expected JSON Output:** {{"is_valid": true/false, "reason": "Brief explanation"}}
    """

    global genai, types, client
    if genai is None or types is None:
        from google import genai as _genai
        from google.genai import types as _types
        genai, types = _genai, _types
    if client is None:
        client = None

    try:
        attempts = 0
        parsed_validation_response_model = None
        while attempts < 2 and parsed_validation_response_model is None:
            attempts += 1
            response = generate_content_resilient(
                user_prompt,
                model=config.GEMINI_MODEL_NAME,
                temperature=0.0,
                system_instruction=system_prompt,
                response_mime_type='application/json',
                response_schema=ValidationResponse,
            )
            llm_output = response.text.strip()
            try:
                parsed_validation_response_model = ValidationResponse.model_validate_json(llm_output)
            except ValidationError as e:
                logging.error(f"Failed to validate LLM JSON output: {e}")
                logging.error(f"LLM Raw Output: {llm_output}")
                if attempts < 2:
                    time.sleep(config.GEMINI_REQUEST_DELAY_SECONDS)
                else:
                    return False, "Validation failed - JSON parsing error"
            except json.JSONDecodeError as e: 
                logging.error(f"Failed to parse LLM JSON output: {e}")
                logging.error(f"LLM Raw Output: {llm_output}")
                if attempts < 2:
                    time.sleep(config.GEMINI_REQUEST_DELAY_SECONDS)
                else:
                    return False, "Validation failed - JSON decoding error"
        
        logging.info(f"Validation response: {parsed_validation_response_model}")
        return parsed_validation_response_model.is_valid, parsed_validation_response_model.reason

    except Exception as e:
        logging.error(f"Error calling Gemini for validation: {e}")
        return False, f"Validation error: {str(e)}"

# --- Helper: Filter irrelevant experience for data roles ---
def filter_experience_for_data_roles(experience: List[Experience], job_details: Dict[str, Any]) -> List[Experience]:
    """
    Remove experience entries that are clearly irrelevant for data/analytics roles.
    E.g., pure IT support internships with no analytical component.
    """
    job_title_lower = job_details.get('job_title', '').lower()
    is_data_role = any(kw in job_title_lower for kw in ['data', 'analyst', 'scientist', 'analytics', 'ml', 'machine learning', 'ai'])
    
    if not is_data_role:
        return experience  # Keep all if not a data role
    
    filtered = []
    for exp in experience:
        # Check if experience has any data/analytical relevance
        exp_text = f"{exp.job_title} {exp.description}".lower()
        
        # Red flags for pure IT support with no data relevance
        it_support_only = all([
            any(kw in exp_text for kw in ['it support', 'technical support', 'helpdesk', 'help desk']),
            not any(kw in exp_text for kw in ['data', 'analysis', 'analytics', 'sql', 'python', 'reporting', 'dashboard', 'metrics', 'kpi', 'statistics'])
        ])
        
        if it_support_only:
            logging.info(f"Filtering out IT support role with no data relevance: {exp.job_title} at {exp.company}")
            continue
        
        filtered.append(exp)
    
    return filtered

# --- Main Processing Logic ---
async def process_job(job_details: Dict[str, Any], base_resume_details: Resume, output_dir: Optional[str] = None, no_upload: bool = False):
    """
    Processes a single job: personalizes resume, generates PDF, uploads, updates status.
    """
    global supabase_utils
    if supabase_utils is None:
        import supabase_utils as _supabase_utils
        supabase_utils = _supabase_utils

    job_id = job_details.get("job_id")
    if not job_id:
        logging.error("Job details missing job_id.")
        return

    logging.info(f"--- Starting processing for job_id: {job_id} ---")

    try:
        # 1. Personalize Resume Sections
        personalized_resume_data = base_resume_details.model_copy(deep=True)
        any_validation_failed = False

        sections_to_personalize = {
            "summary": base_resume_details.summary,
            "experience": base_resume_details.experience,
            "projects": base_resume_details.projects,
            "skills": base_resume_details.skills,
        }

        sleep_time = 6

        for section_name, section_content in sections_to_personalize.items():
            if any_validation_failed:
                logging.warning(f"Skipping further personalization for job_id {job_id} due to prior validation failure.")
                break

            if section_content:
                logging.info(f"Waiting for {sleep_time:.2f} seconds before next request...")
                time.sleep(sleep_time)

                logging.info(f"Personalizing section: {section_name} for job_id: {job_id}")
                personalized_content = await personalize_section_with_llm(
                    section_name,
                    section_content,
                    base_resume_details,
                    job_details
                )

                logging.info(f"Waiting for {sleep_time:.2f} seconds before next request...")
                time.sleep(sleep_time)

                # Validate the customization
                logging.info(f"Validating customization for section: {section_name} for job_id: {job_id}")
                is_valid, reason = await validate_customization(
                    section_name,
                    section_content,
                    personalized_content,
                    base_resume_details,
                    job_details 
                )

                if is_valid:
                    logging.info(f"Customization for section {section_name} is valid.")
                    setattr(personalized_resume_data, section_name, personalized_content)
                    sections_to_personalize[section_name] = personalized_content
                else:
                    logging.warning(f"VALIDATION FAILED for section {section_name} for job_id {job_id}. Reason: {reason}")
                    logging.warning(f"Reverting to original content for {section_name}.")
                    setattr(personalized_resume_data, section_name, section_content)
                    sections_to_personalize[section_name] = section_content
                
                logging.info(f"Finished personalizing section: {section_name} for job_id: {job_id}")
            else:
                 logging.info(f"Skipping empty section: {section_name} for job_id: {job_id}")

        if any_validation_failed:
            logging.info(f"--- Aborting PDF generation for job_id: {job_id} due to validation failure. ---")
            return 

        logging.info(f"Enriching skills and project technologies for job_id: {job_id}")
        enriched_resume = enrich_resume(personalized_resume_data)

        # 2. Compress content for two-page target
        logging.info(f"Applying content compression for job_id: {job_id}")
        compressed_resume = compress_resume_for_job(enriched_resume, job_details)

        # 3. Generate PDF (first pass)
        logging.info(f"Generating PDF for job_id: {job_id}")
        try:
            pdf_bytes = pdf_generator.create_resume_pdf(compressed_resume)
            if not pdf_bytes:
                 raise ValueError("PDF generation returned empty bytes.")
            logging.info(f"PDF generation complete for job_id: {job_id}")
        except Exception as e:
            logging.error(f"Failed to generate PDF for job_id {job_id}: {e}")
            return

        # 4. Enforce max pages via tighten loop
        if config.ENFORCE_TWO_PAGE_RESUME:
            try:
                pages = count_pdf_pages(pdf_bytes)
                logging.info(f"Initial PDF pages for job_id {job_id}: {pages}")
                tighten_round = 0
                while pages > config.RESUME_MAX_PAGES and tighten_round < 5:
                    tighten_round += 1
                    logging.info(f"Tightening content (round {tighten_round}) for job_id {job_id}")
                    compressed_resume = tighten_resume(compressed_resume, round_num=tighten_round)
                    pdf_bytes = pdf_generator.create_resume_pdf(compressed_resume)
                    pages = count_pdf_pages(pdf_bytes)
                    logging.info(f"Pages after tighten round {tighten_round}: {pages}")
            except Exception as e:
                logging.error(f"Error during pagination enforcement: {e}")

        if no_upload:
            try:
                out_dir = output_dir or os.path.join(os.getcwd(), "out", "customized_resumes")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"resume_{job_id}.pdf")
                with open(out_path, "wb") as f:
                    f.write(pdf_bytes)
                logging.info(f"Saved PDF locally: {out_path}")
            except Exception as e:
                logging.error(f"Failed to save PDF locally for job_id {job_id}: {e}")
                return
        else:
            destination_path = f"personalized_resumes/resume_{job_id}.pdf"
            logging.info(f"Uploading PDF to {destination_path} for job_id: {job_id}")
            resume_link = supabase_utils.upload_customized_resume_to_storage(pdf_bytes, destination_path)

            if not resume_link:
                logging.error(f"Failed to upload resume PDF for job_id: {job_id}")
                return

            logging.info(f"Successfully uploaded PDF for job_id: {job_id}. Link: {resume_link}")
            logging.info("Adding customized resume to Supabase")
            customized_resume_id = supabase_utils.save_customized_resume(compressed_resume, resume_link)
            logging.info(f"Updating job record for job_id: {job_id} with resume link.")
            update_success = supabase_utils.update_job_with_resume_link(job_id, customized_resume_id, new_status="resume_generated")
            if update_success:
                logging.info(f"Successfully updated job record for job_id: {job_id}")
            else:
                logging.error(f"Failed to update job record for job_id: {job_id}")

        logging.info(f"--- Finished processing for job_id: {job_id} ---")

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing job_id {job_id}: {e}", exc_info=True)

# --- Helper: Count PDF Pages ---
def count_pdf_pages(pdf_bytes: bytes) -> int:
    with io.BytesIO(pdf_bytes) as pdf_file:
        with pdfplumber.open(pdf_file) as pdf:
            return len(pdf.pages)

# --- Helper: Simple keyword extraction ---
def _extract_keywords(job_description: str) -> List[str]:
    text = (job_description or "").lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]{2,}", text)
    stop = {
        "and","or","the","with","for","to","of","in","on","at","by","an","a","as","be","is","are","this","that","it","you","we","they","our","their","from","about","into","over","under","will","must"
    }
    return [t for t in tokens if t not in stop]

# --- Compressor: Trim content to quotas ---
def compress_resume_for_job(resume: Resume, job_details: Dict[str, Any]) -> Resume:
    quotas = config.RESUME_SECTION_QUOTAS
    jd_keywords = _extract_keywords(job_details.get("description", ""))
    global _cached_jd_keywords
    _cached_jd_keywords = jd_keywords

    out = resume.model_copy(deep=True)

    # Summary: cap words
    if out.summary:
        words = re.findall(r"\S+", out.summary)
        if len(words) > quotas["summary_max_words"]:
            out.summary = " ".join(words[:quotas["summary_max_words"]])

    # Skills: deduplicate and cap
    if out.skills:
        skills = deduplicate_skills(out.skills)
        max_sk = quotas["skills_max"]
        min_sk = quotas["skills_min"]
        if len(skills) > max_sk:
            def score_skill(s: str) -> int:
                s_low = s.lower()
                return sum(1 for k in jd_keywords if k in s_low)
            ranked = sorted(skills, key=lambda s: (score_skill(s), len(s)), reverse=True)
            trimmed = ranked[:max_sk]
            out.skills = trimmed if len(trimmed) >= min_sk else ranked[:min_sk]
        else:
            out.skills = skills

    # Experience: filter irrelevant roles, then keep most relevant
    if out.experience:
        # First filter out completely irrelevant roles (e.g., pure IT support for data jobs)
        filtered_exp = filter_experience_for_data_roles(out.experience, job_details)
        
        def exp_score(exp: Experience) -> int:
            score = 0
            if exp.job_title:
                jt = exp.job_title.lower()
                score += sum(1 for k in jd_keywords if k in jt)
            if exp.description:
                desc = exp.description.lower()
                score += sum(1 for k in jd_keywords if k in desc)
            return score

        ranked_exps = sorted(filtered_exp, key=lambda e: exp_score(e), reverse=True)
        out.experience = ranked_exps[:quotas["experience_roles_max"]]

        # Normalize bullets per role
        for exp in out.experience:
            if exp.description:
                bullets = [_cleanup_bullet(b) for b in _normalize_to_bullets(exp.description)]
                bullets = bullets[:quotas["experience_bullets_max"]]
                exp.description = "\n".join(bullets)

    # Projects: keep top N and limit bullets
    if out.projects:
        def proj_score(p: Project) -> int:
            s = 0
            if p.name:
                s += sum(1 for k in jd_keywords if k in p.name.lower())
            if p.description:
                s += sum(1 for k in jd_keywords if k in p.description.lower())
            if p.technologies:
                techstr = " ".join(p.technologies).lower()
                s += sum(1 for k in jd_keywords if k in techstr)
            return s

        ranked_projs = sorted(out.projects, key=lambda p: proj_score(p), reverse=True)
        out.projects = ranked_projs[:quotas["projects_max"]]
        for p in out.projects:
            if p.description:
                bullets = [_cleanup_bullet(b) for b in _normalize_to_bullets(p.description)]
                bullets = bullets[:quotas["project_bullets_max"]]
                p.description = "\n".join(bullets)

    return out

# --- Tighten Strategy: iterative reductions if still >2 pages ---
def tighten_resume(resume: Resume, round_num: int) -> Resume:
    quotas = config.RESUME_SECTION_QUOTAS
    out = resume.model_copy(deep=True)

    if round_num == 1:
        # Trim experience bullets to 4
        for exp in (out.experience or []):
            if exp.description:
                bullets = [_cleanup_bullet(b) for b in _normalize_to_bullets(exp.description)]
                ranked = _rank_bullets(bullets)
                exp.description = "\n".join(ranked[:min(4, quotas["experience_bullets_max"])])
    elif round_num == 2:
        # Trim experience bullets to 3
        for exp in (out.experience or []):
            if exp.description:
                bullets = [_cleanup_bullet(b) for b in _normalize_to_bullets(exp.description)]
                ranked = _rank_bullets(bullets)
                exp.description = "\n".join(ranked[:min(3, quotas["experience_bullets_max"])])
        # Projects: keep most relevant one with up to 2 bullets
        if out.projects:
            out.projects = out.projects[:1]
            for p in out.projects:
                if p.description:
                    bullets = _normalize_to_bullets(p.description)
                    ranked = _rank_bullets(bullets)
                    p.description = "\n".join(ranked[:2])
    elif round_num == 3:
        # Drop certifications
        out.certifications = []
    elif round_num == 4:
        # Drop languages
        out.languages = []
    elif round_num == 5:
        # Reduce skills to min
        if out.skills:
            out.skills = out.skills[:quotas["skills_min"]]

    return out

# --- Helpers: bullets normalization and ranking ---
def _normalize_to_bullets(text: str) -> List[str]:
    if "\n" in text:
        bullets = [b.strip() for b in text.split("\n") if b.strip()]
    else:
        bullets = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return bullets

def _rank_bullets(bullets: List[str]) -> List[str]:
    def score(b: str) -> int:
        s = 0
        bl = b.lower()
        for k in _cached_jd_keywords:
            if k in bl:
                s += 1
        if re.search(r"\b\d+%?\b", b):
            s += 2
        return s
    ranked = sorted(bullets, key=score, reverse=True)
    return ranked

# cached JD keywords for ranking
_cached_jd_keywords: List[str] = []

def _cleanup_bullet(b: str) -> str:
    t = b.strip()
    # Remove bullet symbols if present
    if t.startswith('•'):
        t = t[1:].strip()
    elif t.startswith('-'):
        t = t[1:].strip()
    
    weak = [
        "communicate results effectively",
        "learned to manage time",
        "demonstrated users in utilization",
    ]
    tl = t.lower()
    for w in weak:
        if w in tl:
            return ""
    return t

def enrich_resume(resume: Resume) -> Resume:
    """
    Enrich the resume by:
    1. Extracting technologies from project descriptions
    2. Deduplicating skills
    3. Adding inferred skills based on experience/project content
    """
    out = resume.model_copy(deep=True)
    
    # Collect all text tokens from experience and projects
    tokens = []
    def add_text(s: str):
        if s:
            tokens.extend(re.findall(r"[A-Za-z][A-Za-z0-9+.#\-]{2,}", s))
    
    for exp in (out.experience or []):
        add_text(exp.description or "")
    
    for p in (out.projects or []):
        add_text(p.description or "")
        # Extract technologies from project descriptions
        if p.description:
            extracted_techs = extract_technologies_from_description(p.description, p.technologies)
            p.technologies = list(set(extracted_techs))  # Remove duplicates
    
    tok_set = set(t.lower() for t in tokens)
    
    # Known skill keywords to look for
    known = {
        "python": "Python",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "scikit-learn": "Scikit-Learn",
        "sklearn": "Scikit-Learn",
        "sql": "SQL",
        "mysql": "MySQL",
        "database": "Database Design",
        "erd": "ERD Modeling",
        "modeling": "Data Modeling",
        "power bi": "Power BI",
        "powerbi": "Power BI",
        "tableau": "Tableau",
        "excel": "Excel",
        "jupyter": "Jupyter",
        "git": "Git",
        "kpi": "KPI Reporting",
        "reporting": "Reporting",
        "data cleaning": "Data Cleaning",
        "statistical": "Statistical Analysis",
        "statistics": "Statistical Analysis",
        "feature engineering": "Feature Engineering",
        "classification": "Classification",
        "model evaluation": "Model Evaluation",
        "machine learning": "Machine Learning",
        "data visualization": "Data Visualization",
        "data analysis": "Data Analysis",
        "root cause analysis": "Root Cause Analysis",
        "process optimization": "Process Optimization",
        "quality management": "Quality Management",
        "spc": "Statistical Process Control",
    }
    
    addable = []
    for keyword, proper_name in known.items():
        if any(keyword in t for t in tok_set):
            addable.append(proper_name)
    
    # Deduplicate and merge with existing skills
    current_skills = set(out.skills or [])
    merged_skills = current_skills.union(set(addable))
    
    # Remove "Big Data" if no specific big data tools mentioned
    if "Big Data" in merged_skills:
        big_data_tools = {"Hadoop", "Spark", "Hive", "Kafka"}
        if not any(tool in current_skills for tool in big_data_tools):
            merged_skills.discard("Big Data")
    
    out.skills = deduplicate_skills(list(merged_skills))
    
    return out

async def run_job_processing_cycle(jobs_limit: Optional[int] = None, output_dir: Optional[str] = None, no_upload: bool = False):
    """
    Fetches top jobs and processes them one by one.
    """
    logging.info("Starting new job processing cycle...")

    # 1. Retrieve Base Resume Details
    user_email = config.LINKEDIN_EMAIL
    if not user_email:
        logging.error("LINKEDIN_EMAIL not set in config. Cannot fetch base resume.")
        return

    logging.info(f"Fetching base resume for user: {user_email}")
    global supabase_utils
    if supabase_utils is None:
        import supabase_utils as _supabase_utils
        supabase_utils = _supabase_utils

    raw_resume_details = supabase_utils.get_resume_custom_fields_by_email(user_email)

    if not raw_resume_details:
        logging.error(f"Could not find base resume for user: {user_email}. Aborting cycle.")
        return

    # Parse raw details into Pydantic model
    try:
        for key in ['skills', 'experience', 'education', 'projects', 'certifications', 'languages']:
             if raw_resume_details.get(key) is None:
                 raw_resume_details[key] = []
        base_resume_details = Resume(**raw_resume_details)
        logging.info("Successfully parsed base resume.")
    except Exception as e:
        logging.error(f"Error parsing base resume details into Pydantic model: {e}")
        logging.error(f"Raw base resume data: {raw_resume_details}")
        return

    # 2. Fetch Top Jobs to Process
    jobs_limit = jobs_limit or 2
    logging.info(f"Fetching top {jobs_limit} scored jobs to apply for...")
    jobs_to_process = supabase_utils.get_top_scored_jobs_for_resume_generation(limit=jobs_limit)

    if not jobs_to_process:
        logging.info("No new jobs found to process in this cycle.")
        return

    logging.info(f"Found {len(jobs_to_process)} jobs to process.")

    # 3. Process Each Job Sequentially
    for job_details in jobs_to_process:
        await process_job(job_details, base_resume_details, output_dir=output_dir, no_upload=no_upload)

    logging.info("Finished job processing cycle.")

# --- Script Entry Point ---
if __name__ == "__main__":
    if "--local-test" in sys.argv:
        job_details = {
            "job_title": "Data Analyst",
            "company": "Acme",
            "level": "Mid",
            "description": "We need a data analyst with Python, SQL, dashboards, ETL, AWS, pandas, statistics, A/B testing, machine learning basics, communication skills."
        }
        exp_desc = (
            "Built dashboards with pandas and SQL to monitor KPIs and reduce reporting time by 30%.\n"
            "Developed ETL pipelines on AWS (Lambda, S3) to ingest data daily.\n"
            "Automated data quality checks and alerting; collaborated with stakeholders to prioritize."
        )
        exp = [
            Experience(job_title="Data Analyst", company="XYZ", location="Singapore", start_date="2021", end_date="2024", description=exp_desc),
            Experience(job_title="IT Support Specialist", company="ABC", location="Singapore", start_date="2018", end_date="2021", description=(
                "Provided support; wrote Python scripts to automate tasks.\n"
                "Managed asset inventory and tickets.\n"
                "Improved resolution time by 20%."
            )),
        ]
        projects = [
            Project(name="Sales Analytics", description=(
                "Sales analytics dashboard with Python and Plotly; improved executive visibility.\n"
                "A/B testing framework with statsmodels and pandas."
            ), technologies=["Python","Plotly","pandas"]),
            Project(name="AB Testing", description="Designed and validated experiments; built reporting tooling.", technologies=["Python","statsmodels","SQL"])
        ]
        resume = Resume(
            name="John Doe",
            email="john@example.com",
            phone="123",
            location="Singapore",
            summary=(
                "Data professional with experience in analytics, ETL, and dashboards, focusing on impact and "
                "stakeholder alignment. Experienced in Python, SQL, AWS, and building reporting solutions."
            ),
            skills=["Python","SQL","pandas","AWS","ETL","Dashboards","Statistics","A/B Testing","Communication","Git","CI/CD","Docker"],
            experience=exp,
            education=[Education(degree="BSc", field_of_study="Information Systems", institution="Uni", start_year="2014", end_year="2018")],
            projects=projects,
            certifications=[Certification(name="AWS CP", issuer="AWS", year="2020")],
            languages=["English","Mandarin"],
            links=Links(linkedin="https://linkedin.com/in/john", github=None, portfolio=None)
        )

        compressed = compress_resume_for_job(resume, job_details)
        pdf_bytes = pdf_generator.create_resume_pdf(compressed)
        pages = count_pdf_pages(pdf_bytes)
        tighten_round = 0
        while config.ENFORCE_TWO_PAGE_RESUME and pages > config.RESUME_MAX_PAGES and tighten_round < 5:
            tighten_round += 1
            compressed = tighten_resume(compressed, tighten_round)
            pdf_bytes = pdf_generator.create_resume_pdf(compressed)
            pages = count_pdf_pages(pdf_bytes)

        out_dir = os.path.join(os.getcwd(), "out")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        out_path = os.path.join(out_dir, "local_test_resume.pdf")
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"Saved PDF: {out_path}")
        print(f"Pages: {pages}")
        print(f"Skills: {len(compressed.skills) if compressed.skills else 0}")
        print(f"Experience roles: {len(compressed.experience) if compressed.experience else 0}")
        print(f"Projects: {len(compressed.projects) if compressed.projects else 0}")
    else:
        jobs_limit_arg = None
        output_dir_arg = None
        no_upload_arg = False
        for i, a in enumerate(sys.argv):
            if a == "--jobs-limit" and i + 1 < len(sys.argv):
                try:
                    jobs_limit_arg = int(sys.argv[i + 1])
                except:
                    jobs_limit_arg = None
            elif a == "--output-dir" and i + 1 < len(sys.argv):
                output_dir_arg = sys.argv[i + 1]
            elif a == "--no-upload":
                no_upload_arg = True
        logging.info("Script started.")
        try:
            asyncio.run(run_job_processing_cycle(jobs_limit=jobs_limit_arg, output_dir=output_dir_arg, no_upload=no_upload_arg))
            logging.info("Resume processing completed successfully.")
        except Exception as e:
            logging.error(f"Error during task execution: {e}", exc_info=True)