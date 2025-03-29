import fitz
import re
import spacy
import language_tool_python
from sentence_transformers import SentenceTransformer, util
from dateutil.parser import parse
from datetime import datetime

# Lazy-load globals
model = None
nlp = None
tool = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def get_nlp():
    global nlp
    if nlp is None:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            import os
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    return nlp

# def get_tool():
#     global tool
#     if tool is None:
#         import language_tool_python
#         tool = language_tool_python.LanguageTool('en-US')
#     return tool

experience_date_patterns = [
    # Flexible MM/YYYY or M/YYYY
    r'(\d{1,2}[/.\\-]\d{4})\s*[-–to]+\s*(\d{1,2}[/.\\-]\d{4}|Present|Current)',

    # Full month names
    r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)[.,]?\s?\d{4})\s*[-–to]+\s*(?:\b(?:January|February|March|April|May|June|July|August|September|October|November|December)[.,]?\s?\d{4}|Present|Current)',

    # Abbreviated month names
    r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[.,]?\s?\d{4})\s*[-–to]+\s*(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[.,]?\s?\d{4}|Present|Current)',

    # Year ranges
    r'(\b\d{4})\s*[-–to]+\s*(\d{4}|Present|Current)',

    # 2-digit month/year shorthand (e.g., 03/23)
    r'(\d{2}[/.\\-]?\d{2})\s*[-–to]+\s*(\d{2}[/.\\-]?\d{2}|Present|Current)',

    # Since/From with optional month
    r'\b(Since|From)\s+(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.,]?\s*)?\d{4}',

    # Standalone "Present" or "Current"
    r'\b(Current|Present)\b'
]

def extract_experience_years(text):
    experience_periods = []
    today = datetime.today()

    for pattern in experience_date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple) and len(match) == 2:
                start_str, end_str = match
            else:
                continue  # skip single or malformed entries

            try:
                start_date = parse(start_str.lower(), fuzzy=True)
                end_date = parse(end_str.lower(), fuzzy=True) if not re.search(r'present|current', end_str, re.I) else today
                if start_date < end_date:
                    experience_periods.append((start_date, end_date))
            except Exception:
                continue

    # Merge overlapping periods
    experience_periods.sort()
    merged_periods = []

    for start, end in experience_periods:
        if not merged_periods or start > merged_periods[-1][1]:
            merged_periods.append([start, end])
        else:
            merged_periods[-1][1] = max(merged_periods[-1][1], end)

    # Calculate total experience in months
    total_months = 0
    for start, end in merged_periods:
        months = (end.year - start.year) * 12 + (end.month - start.month)
        total_months += max(0, months)

    return round(total_months / 12, 2)

# def count_dynamic_filtered_errors(text):
#     matches = get_tool().check(text)
#     special_terms = get_special_terms(text)

#     filtered_errors = []
#     for match in matches:
#         error_word = match.context[match.offsetInContext: match.offsetInContext + match.errorLength]
#         if error_word in special_terms:
#             continue
#         filtered_errors.append(match)

#     return len(filtered_errors)

def semantic_similarity(texts, job_desc_embedding):
    embeddings = get_model().encode(texts)
    from sentence_transformers import util
    return [round(float(util.cos_sim(embed, job_desc_embedding)), 3) for embed in embeddings]

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

    full_text = re.sub(r'^\s*[-•*▪‣➤►❖✔️]+\s*', '', full_text, flags=re.MULTILINE)

    doc = get_nlp()(full_text)
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
    
    exp_section = re.findall(r'EXPERIENCE(.*?)(SKILLS|EDUCATION|CERTIFICATIONS|PROJECTS|$)', full_text, re.I | re.S)
    experience_text = ' '.join(
        [match[0].strip().replace('\n', ' ') for match in exp_section]) if exp_section else 'Not found'
    
    internships_count = len(re.findall(r'\b(Intern(ship)?|Trainee)\b', experience_text, re.IGNORECASE))
    experience_years = extract_experience_years(experience_text)
    
    edu_section = re.findall(r'EDUCATION\s*&?\s*CERTIFICATIONS?\s*(.*?)(SKILLS|PROJECTS|EXPERIENCE|LANGUAGES|$)', full_text, re.I | re.S)
    edu_text = ' '.join([m[0].strip().replace('\n', ' ') for m in edu_section]) if edu_section else 'Not found'

    skills_section = re.findall(r'SKILLS\s*(.*?)(LANGUAGES|EDUCATION|CERTIFICATIONS|PROJECTS|EXPERIENCE|$)', full_text, re.I | re.S)
    skills_text = ' '.join([m[0].strip().replace('\n', ' ') for m in skills_section]) if skills_section else ''

    cert_section = re.findall(r'(CERTIFICATIONS|COURSES)\s*(.*?)(SKILLS|LANGUAGES|TOOLS|$)', full_text, re.I | re.S)
    cert_text = ' '.join([m[1].strip().replace('\n', ' ') for m in cert_section]) if cert_section else ''

    scores = semantic_similarity([skills_text, cert_text, experience_text, edu_text], job_desc_embedding)
    skills_score, cert_score, exp_score, edu_score = scores

    # grammatical_errors_count = round(count_dynamic_filtered_errors(full_text) / 10, 2)
    grammatical_errors_count = 0

    features = {
        **basic_info,
        'Education_Level': 1 if education_level == "Bachelor's" else 2 if education_level == "Master's" else 3,
        'Experience_Years': experience_years,
        'Internships_Count': internships_count,
        'Skills_Score': skills_score*2,
        'Industry_Relevance_Score': round((skills_score + exp_score) / 2, 3)*2,
        'Education_Score': edu_score*2,
        'Certifications_Score': cert_score,
        'Experience_Score': exp_score*1.5,
        'Extracurricular_Activities': "Yes" if re.search(r'EXTRACURRICULAR|VOLUNTEER', full_text, re.I) else "No",
        'Grammatical_Mistakes_Count': grammatical_errors_count
    }

    return features

def extract_resume_features(job_description, pdf_path):
    job_desc_embedding = get_model().encode(job_description)
    return comprehensive_resume_extraction_with_ner(pdf_path, job_desc_embedding)
