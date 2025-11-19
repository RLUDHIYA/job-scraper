import io
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib import colors
from models import Resume 

logging.basicConfig(level=logging.INFO)

def create_resume_pdf(resume_data: Resume) -> bytes:
    """
    Generates an ATS-friendly PDF resume with improved design from the provided Resume data object.
    Returns the PDF content as bytes.
    """
    buffer = io.BytesIO()
    
    # Document setup with slightly wider margins
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        leftMargin=0.6*inch, 
        rightMargin=0.6*inch,
        topMargin=0.6*inch, 
        bottomMargin=0.6*inch
    )
    
    # Create custom styles
    styles = getSampleStyleSheet()
    
    # Modern color palette
    primary_color = colors.HexColor('#1976D2')
    secondary_color = colors.HexColor('#455A64')
    accent_color = colors.HexColor('#03A9F4')
    text_color = colors.HexColor('#212121')
    light_text = colors.HexColor('#757575')
    
    # Custom styles
    style_name = ParagraphStyle(
        name='Name',
        parent=styles['Heading1'],
        fontSize=26,
        alignment=TA_LEFT,
        spaceAfter=10,
        fontName='Helvetica-Bold',
        textColor=primary_color,
    )
    
    style_section_heading = ParagraphStyle(
        name='SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=4,
        fontName='Helvetica-Bold',
        textColor=primary_color,
        alignment=TA_LEFT,
    )
    
    style_normal = ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,  
        fontName='Helvetica',
        textColor=text_color,
    )
    
    style_contact = ParagraphStyle(
        name='Contact',
        parent=styles['Normal'],
        alignment=TA_LEFT,
        fontSize=9,
        leading=12,
        spaceAfter=2,
        textColor=secondary_color,
    )
    
    style_job_title = ParagraphStyle(
        name='JobTitle',
        parent=styles['Normal'],
        fontSize=12,  
        spaceAfter=4,
        fontName='Helvetica-Bold',
        textColor=primary_color,  
    )
    
    style_company = ParagraphStyle(
        name='Company',
        parent=styles['Normal'],
        spaceBefore=2,
        fontSize=10,
        fontName='Helvetica-Bold',  
        textColor=secondary_color,
    )
    
    style_dates = ParagraphStyle(
        name='Dates',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_LEFT,
        fontName='Helvetica-Oblique',
        textColor=light_text,
    )
    
    style_bullet = ParagraphStyle(
        name='Bullet',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        leftIndent=10,
        bulletIndent=0,
        fontName='Helvetica',
        textColor=text_color,
        spaceAfter=1,
    )

    style_tech = ParagraphStyle(
        name='Technologies',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Helvetica-Oblique',
        textColor=light_text,
        spaceAfter=8,
    )
    
    story = []

    # --- Header ---
    if resume_data.name:
        story.append(Paragraph(resume_data.name.upper(), style_name))

    # --- Contact Information ---
    contact_info = []
    if resume_data.email: contact_info.append(resume_data.email)
    if resume_data.phone: contact_info.append(resume_data.phone)
    if resume_data.location: contact_info.append(resume_data.location)
    if contact_info:
        story.append(Paragraph(" | ".join(contact_info), style_contact))
    
    # --- Links ---
    links = []
    if resume_data.links:
        if resume_data.links.linkedin: links.append(f"LinkedIn: {resume_data.links.linkedin}")
        if resume_data.links.github: links.append(f"GitHub: {resume_data.links.github}")
        if resume_data.links.portfolio: links.append(f"Portfolio: {resume_data.links.portfolio}")
    if links:
        story.append(Paragraph(" | ".join(links), style_contact))
    
    # --- Summary ---
    if resume_data.summary:
        story.append(Paragraph("PROFESSIONAL SUMMARY", style_section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2C3E50'), spaceBefore=0, spaceAfter=6))
        
        # Remove leading/trailing double quotes from summary if they exist
        cleaned_summary = resume_data.summary
        if cleaned_summary.startswith('"') and cleaned_summary.endswith('"'):
            cleaned_summary = cleaned_summary[1:-1]
            
        story.append(Paragraph(cleaned_summary, style_normal))
    
    # --- Skills (Grouped) ---
    if resume_data.skills:
        story.append(Paragraph("SKILLS", style_section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2C3E50'), spaceBefore=0, spaceAfter=6))
        grouped = _group_skills(resume_data.skills)
        for heading, items in grouped.items():
            if items:
                story.append(Paragraph(f"<b>{heading}:</b> {', '.join(items)}", style_normal))
                story.append(Spacer(1, 0.04*inch))
    
    # --- Experience ---
    if resume_data.experience:
        story.append(Paragraph("PROFESSIONAL EXPERIENCE", style_section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2C3E50'), spaceBefore=0, spaceAfter=8))
        
        for exp in resume_data.experience:
            job_title = f"{exp.job_title}" if exp.job_title else ""
            if job_title:
                story.append(Paragraph(job_title, style_job_title))
            if exp.company and exp.location:
                story.append(Paragraph(f"{exp.company} | {exp.location}", style_company))
            elif exp.company:
                story.append(Paragraph(exp.company, style_company))
            dates = _format_date_range(exp.start_date, exp.end_date)
            if dates:
                story.append(Paragraph(dates, style_dates))
            story.append(Spacer(1, 0.06*inch))
            
            if exp.description:
                # Trust that description is already formatted with bullets from LLM
                # Check if it contains newlines (pre-formatted bullets)
                if '\n' in exp.description:
                    bullets = exp.description.split('\n')
                    for bullet in bullets:
                        bullet_text = bullet.strip()
                        if bullet_text:
                            # Ensure bullet point formatting
                            if not bullet_text.startswith('•'):
                                bullet_text = f"• {bullet_text}"
                            story.append(Paragraph(bullet_text, style_bullet))
                else:
                    # Single paragraph - add as one bullet
                    if exp.description.strip():
                        bullet_text = exp.description.strip()
                        if not bullet_text.startswith('•'):
                            bullet_text = f"• {bullet_text}"
                        story.append(Paragraph(bullet_text, style_bullet))
            
            story.append(Spacer(1, 0.08*inch))
    
    # --- Education ---
    if resume_data.education:
        story.append(Paragraph("EDUCATION", style_section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2C3E50'), spaceBefore=0, spaceAfter=8))
        
        for edu in resume_data.education:
            # Degree info
            degree_info = f"<b>{edu.degree}</b>"
            if edu.field_of_study: 
                degree_info += f", {edu.field_of_study}"
            
            # Year info
            years = ""
            if edu.start_year and edu.end_year: 
                years = f"{edu.start_year} - {edu.end_year}"
            elif edu.start_year: 
                years = f"Started {edu.start_year}"
            elif edu.end_year: 
                years = f"Graduated {edu.end_year}"
            
            if degree_info:
                story.append(Paragraph(degree_info, style_normal))
            if years:
                story.append(Paragraph(years, style_dates))
            story.append(Paragraph(edu.institution, style_normal))
            story.append(Spacer(1, 0.1*inch))
    
    # --- Projects ---
    if resume_data.projects:
        story.append(Paragraph("PROJECTS", style_section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2C3E50'), spaceBefore=0, spaceAfter=8))
        
        for proj in resume_data.projects:
            story.append(Paragraph(f"<b>{proj.name}</b>", style_job_title))
            
            if proj.description:
                # Trust LLM formatting - check for newlines
                if '\n' in proj.description:
                    bullets = proj.description.split('\n')
                    for bullet in bullets:
                        bullet_text = bullet.strip()
                        if bullet_text:
                            if not bullet_text.startswith('•'):
                                bullet_text = f"• {bullet_text}"
                            story.append(Paragraph(bullet_text, style_bullet))
                else:
                    # Single paragraph
                    if proj.description.strip():
                        bullet_text = proj.description.strip()
                        if not bullet_text.startswith('•'):
                            bullet_text = f"• {bullet_text}"
                        story.append(Paragraph(bullet_text, style_bullet))
            
            if proj.technologies and len(proj.technologies) > 0:
                tech_text = f"<i>Technologies:</i> {', '.join(proj.technologies)}"
                story.append(Paragraph(tech_text, style_tech))
            
            story.append(Spacer(1, 0.15*inch))
    
    # --- Certifications ---
    if resume_data.certifications:
        story.append(Paragraph("CERTIFICATIONS", style_section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2C3E50'), spaceBefore=0, spaceAfter=8))
        
        for cert in resume_data.certifications:
            if cert.name:
                story.append(Paragraph(f"<b>{cert.name}</b>", style_normal))
            if cert.issuer:
                story.append(Paragraph(cert.issuer, style_normal))
            if cert.year:
                story.append(Paragraph(_format_month_year(cert.year), style_dates))
            story.append(Spacer(1, 0.08*inch))
    
    # --- Languages ---
    if resume_data.languages:
        story.append(Paragraph("LANGUAGES", style_section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2C3E50'), spaceBefore=0, spaceAfter=8))
        story.append(Paragraph(", ".join(resume_data.languages), style_normal))
    
    try:
        doc.build(story)
        logging.info("PDF generated successfully.")
    except Exception as e:
        logging.error(f"Error building PDF: {e}")
        raise
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def _format_month_year(text: str) -> str:
    if not text:
        return ""
    patterns = ["%b %Y", "%B %Y", "%Y-%m", "%Y"]
    from datetime import datetime
    for fmt in patterns:
        try:
            if fmt == "%Y":
                dt = datetime(int(text), 1, 1)
            else:
                dt = datetime.strptime(text, fmt)
            return dt.strftime("%b %Y")
        except Exception:
            pass
    return text

def _format_date_range(start: str, end: str) -> str:
    s = _format_month_year(start) if start else ""
    e = _format_month_year(end) if end else ""
    if s and e:
        return f"{s} – {e}"
    if s and not e:
        return f"{s} – Present"
    return ""

def _group_skills(skills: list) -> dict:
    """
    Group skills into meaningful categories for better readability.
    """
    cat = {
        "Programming & Data Tools": [],
        "Data Analysis & Statistics": [],
        "Visualization & BI": [],
        "Databases": [],
        "Machine Learning": [],
        "Domain Skills": [],
        "Tools & Technologies": [],
    }
    
    for s in skills:
        sl = s.lower()
        # Programming & Data Tools
        if any(k in sl for k in ["python", "pandas", "numpy", "scikit", "sklearn", "r ", "julia", "scala"]):
            cat["Programming & Data Tools"].append(s)
        # Visualization & BI
        elif any(k in sl for k in ["power bi", "tableau", "visualization", "dashboard", "looker", "qlik", "plotly", "matplotlib", "seaborn"]):
            cat["Visualization & BI"].append(s)
        # Databases
        elif any(k in sl for k in ["sql", "mysql", "postgresql", "database", "erd", "schema", "mongodb", "nosql", "oracle", "sqlite"]):
            cat["Databases"].append(s)
        # Machine Learning
        elif any(k in sl for k in ["classification", "regression", "random forest", "xgboost", "svm", "model", "ml", "machine learning", "deep learning", "neural", "tensorflow", "pytorch", "keras"]):
            cat["Machine Learning"].append(s)
        # Data Analysis & Statistics
        elif any(k in sl for k in ["data cleaning", "statistical", "statistics", "hypothesis", "a/b test", "kpi", "analysis", "analytics", "trend", "reporting", "data quality", "etl", "data pipeline"]):
            cat["Data Analysis & Statistics"].append(s)
        # Domain Skills
        elif any(k in sl for k in ["quality", "process", "optimization", "root cause", "spc", "manufacturing", "reliability", "testing"]):
            cat["Domain Skills"].append(s)
        # Tools & Technologies
        elif any(k in sl for k in ["excel", "git", "github", "jupyter", "notebook", "aws", "azure", "gcp", "docker", "kubernetes", "airflow"]):
            cat["Tools & Technologies"].append(s)
        # Default - put in first appropriate category
        else:
            cat["Tools & Technologies"].append(s)
    
    # Return only non-empty categories in logical order
    ordered_result = {}
    order = [
        "Programming & Data Tools",
        "Data Analysis & Statistics",
        "Machine Learning",
        "Visualization & BI",
        "Databases",
        "Tools & Technologies",
        "Domain Skills"
    ]
    
    for key in order:
        if cat[key]:
            ordered_result[key] = cat[key]
    
    return ordered_result