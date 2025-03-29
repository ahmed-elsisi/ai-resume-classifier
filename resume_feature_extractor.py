import fitz
import re
from dateparser import parse

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

def get_tool():
    global tool
    if tool is None:
        import language_tool_python
        tool = language_tool_python.LanguageTool('en-US')
    return tool

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
        elif end_date > last_end_date:
            total_experience_years += (end_date.year - last_end_date.year)
            last_end_date = end_date

    return max(0, total_experience_years)

def get_special_terms(text):
    doc = get_nlp()(text)
    special_terms = set(ent.text.split() for ent in doc.ents)
    acronyms = re.findall(r'\b[A-Z]{2,}(?:-[A-Z]+)*\b', text)
    special_terms.update(acronyms)
    return special_terms

def count_dynamic_filtered_errors(text):
    matches = get_tool().check(text)
    special_terms = get_special_terms(text)

    filtered_errors = []
    for match in matches:
        error_word = match.context[match.offsetInContext: match.offsetInContext + match.errorLength]
        if error_word in special_terms:
            continue
        filtered_errors.append(match)

    return len(filtered_errors)

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

    full_text = re.sub(r'•|\*|- ', '', full_text)

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

    experience_years = extract_experience_years(full_text)
    internships_count = len(re.findall(r'\bIntern(ship)?\b', full_text, re.I))

    exp_section = re.findall(r'EXPERIENCE\s*(.*?)(NOTABLE ACHIEVEMENTS|EDUCATION|CERTIFICATIONS|PROJECTS|SKILLS|$)', full_text, re.I | re.S)
    experience_text = ' '.join([m[0].strip().replace('\n', ' ') for m in exp_section]) if exp_section else 'Not found'

    edu_section = re.findall(r'EDUCATION\s*&?\s*CERTIFICATIONS?\s*(.*?)(SKILLS|PROJECTS|EXPERIENCE|LANGUAGES|$)', full_text, re.I | re.S)
    edu_text = ' '.join([m[0].strip().replace('\n', ' ') for m in edu_section]) if edu_section else 'Not found'

    skills_section = re.findall(r'SKILLS\s*(.*?)(LANGUAGES|EDUCATION|CERTIFICATIONS|PROJECTS|EXPERIENCE|$)', full_text, re.I | re.S)
    skills_text = ' '.join([m[0].strip().replace('\n', ' ') for m in skills_section]) if skills_section else ''

    cert_section = re.findall(r'(CERTIFICATIONS|COURSES)\s*(.*?)(SKILLS|LANGUAGES|TOOLS|$)', full_text, re.I | re.S)
    cert_text = ' '.join([m[1].strip().replace('\n', ' ') for m in cert_section]) if cert_section else ''

    scores = semantic_similarity([skills_text, cert_text, experience_text, edu_text], job_desc_embedding)
    skills_score, cert_score, exp_score, edu_score = scores

    grammatical_errors_count = round(count_dynamic_filtered_errors(full_text) / 10, 2)

    features = {
        **basic_info,
        'Education_Level': 1 if education_level == "Bachelor's" else 2 if education_level == "Master's" else 3,
        'Experience_Years': experience_years,
        'Internships_Count': internships_count,
        'Skills_Score': skills_score*2,
        'Industry_Relevance_Score': round((skills_score + exp_score) / 2, 3)*2,
        'Education_Score': edu_score*2,
        'Certifications_Score': cert_score,
        'Experience_Score': exp_score*2,
        'Extracurricular_Activities': "Yes" if re.search(r'EXTRACURRICULAR|VOLUNTEER', full_text, re.I) else "No",
        'Grammatical_Mistakes_Count': grammatical_errors_count
    }

    return features

def extract_resume_features(job_description, pdf_path):
    job_desc_embedding = get_model().encode(job_description)
    return comprehensive_resume_extraction_with_ner(pdf_path, job_desc_embedding)
