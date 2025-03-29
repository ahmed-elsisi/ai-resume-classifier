import fitz
import re
import spacy
import language_tool_python
from sentence_transformers import SentenceTransformer, util
from dateparser import parse

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
model = SentenceTransformer('all-MiniLM-L6-v2')
tool = language_tool_python.LanguageTool('en-US')

experience_date_patterns = [
    r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\s?[-–]\s?(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{4}|Present|Current)',
    r'(\d{4})\s?[-–]\s?(\d{4}|Present|Current)'
]

def extract_experience_years(text):
    experience_periods = []
    today = parse("today")

    for pattern in experience_date_patterns:
        matches = re.findall(pattern, text, re.I)
        for start, end in matches:
            start_date = parse(start)
            end_date = parse(end) if "Present" not in end and "Current" not in end else today
            if start_date and end_date:
                experience_periods.append((start_date, end_date))

    experience_periods.sort()
    total_experience_years = 0
    last_end_date = None

    for start_date, end_date in experience_periods:
        if last_end_date is None or start_date > last_end_date:
            total_experience_years += (end_date.year - start_date.year)
            last_end_date = end_date
        else:
            if end_date > last_end_date:
                total_experience_years += (end_date.year - last_end_date.year)
                last_end_date = end_date

    return max(0, total_experience_years)

def get_special_terms(text):
    doc = nlp(text)
    special_terms = set(ent.text for ent in doc.ents)
    acronyms = re.findall(r'\b[A-Z]{2,}(?:-[A-Z]+)*\b', text)
    special_terms.update(acronyms)
    return special_terms

def count_dynamic_filtered_errors(text, tool):
    matches = tool.check(text)
    special_terms = get_special_terms(text)
    filtered_errors = [match for match in matches if match.context[match.offsetInContext: match.offsetInContext + match.errorLength] not in special_terms]
    return len(filtered_errors)

def extract_entities(doc, entity_type):
    return [ent.text for ent in doc.ents if ent.label_ == entity_type]

def comprehensive_resume_extraction_with_ner(pdf_path, job_desc_embedding):
    pdf_document = fitz.open(pdf_path)
    full_text = ''
    linkedin_url = 'Not found'

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        full_text += page.get_text()
        if linkedin_url == 'Not found':
            links = page.get_links()
            for link in links:
                uri = link.get('uri', '').lower()
                if re.search(r'linkedin\.com/(in|pub|profile|company)/[\w-]+', uri):
                    linkedin_url = uri
                    break

    full_text = re.sub(r'\u2022|\*|- ', '', full_text)
    doc = nlp(full_text)
    locations = extract_entities(doc, 'GPE')

    basic_info = {
        'Email': re.search(r'\b[\w.-]+?@\w+?\.\w+?\b', full_text).group().strip() if re.search(r'\b[\w.-]+?@\w+?\.\w+?\b', full_text) else '',
        'Phone_Number': re.search(r'(\+?\d{1,3}\s?)?(\d{2,4}[\s-]?){2,5}\d{2,4}', full_text).group().strip() if re.search(r'(\+?\d{1,3}\s?)?(\d{2,4}[\s-]?){2,5}\d{2,4}', full_text) else '',
        'Location': locations[0] if locations else '',
        'LinkedIn': linkedin_url,
        'Number_of_Pages': pdf_document.page_count,
        'Number_of_Words': len(full_text.split())
    }

    education_level = 'Not found'
    if re.search(r'Ph\.?D|Doctorate', full_text, re.I):
        education_level = "PhD"
    elif re.search(r'Master', full_text, re.I):
        education_level = "Master's"
    elif re.search(r'B\.E\.|B\.Sc\.|Bachelor', full_text, re.I):
        education_level = "Bachelor's"

    experience_years = extract_experience_years(full_text)
    internships_count = len(re.findall(r'\bIntern(ship)?\b', full_text, re.I))

    exp_section = re.findall(r'EXPERIENCE(.*?)(SKILLS|EDUCATION|CERTIFICATIONS|PROJECTS|$)', full_text, re.I | re.S)
    experience_text = ' '.join([match[0].strip().replace('\n', ' ') for match in exp_section]) if exp_section else 'Not found'
    print(experience_text, 'exp')
    edu_section = re.findall(r'EDUCATION\s*&\s*CERTIFICATIONS(.*?)(SKILLS|EDUCATION|PROJECTS|EXPERIENCE$)', full_text, re.I | re.S)
    edu_text = ' '.join([match[0].strip().replace('\n', ' ') for match in edu_section]) if edu_section else 'Not found'
    print(edu_text, 'edu')
    skills_section_match = re.findall(r'SKILLS\s*(.*?)(LANGUAGES|EDUCATION|CERTIFICATIONS|PROJECTS|EXPERIENCE|$)',
                                     full_text, re.I | re.S)
    skills_text = ' '.join([match[0].strip().replace('\n', ' ') for match in skills_section_match]) if skills_section_match else ''
    print(skills_text, 'skill')
    certifications_text_match = re.findall(r'(CERTIFICATIONS|COURSES)(.*?)(SKILLS|LANGUAGES|TOOLS|$)', full_text, re.I | re.S)
    cert_text = ' '.join([match[1].strip().replace('\n', ' ') for match in certifications_text_match]) if certifications_text_match else ''
    print(cert_text, 'cert')
    embeddings = model.encode([skills_text, cert_text, experience_text, edu_text])
    scores = [round(float(util.cos_sim(embed, job_desc_embedding)), 3) for embed in embeddings]
    skills_score, cert_score, exp_score, edu_score = scores

    grammatical_errors_count = round(count_dynamic_filtered_errors(full_text, tool) / 10, 2)

    features = {
        **basic_info,
        'Education_Level': 1 if education_level == "Bachelor's" else 2 if education_level == "Master's" else 3,
        'Experience_Years': experience_years,
        'Internships_Count': internships_count,
        'Skills_Score': skills_score * 1.5,
        'Industry_Relevance_Score': round((skills_score + exp_score) / 2, 3) * 1.5,
        'Education_Score': edu_score * 2,
        'Certifications_Score': cert_score,
        'Experience_Score': exp_score * 1.5,
        'Extracurricular_Activities': "Yes" if re.search(r'EXTRACURRICULAR|VOLUNTEER', full_text, re.I) else "No",
        'Grammatical_Mistakes_Count': grammatical_errors_count
    }

    return features

def extract_resume_features(job_description: str, pdf_path: str) -> dict:
    job_desc_embedding = model.encode(job_description)
    return comprehensive_resume_extraction_with_ner(pdf_path, job_desc_embedding)
