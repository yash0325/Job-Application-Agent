import streamlit as st
import os
import PyPDF2
import docx
from tempfile import NamedTemporaryFile
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ----------- Util: Text Extraction -----------
def extract_text(file, file_type):
    if file_type in ("txt", "md"):
        return file.read().decode("utf-8")
    elif file_type == "pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif file_type == "docx":
        # Save file to disk for python-docx compatibility
        with NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        doc_file = docx.Document(tmp_path)
        os.remove(tmp_path)
        return "\n".join([para.text for para in doc_file.paragraphs])
    else:
        return ""

def get_file_type(file):
    filename = file.name.lower()
    if filename.endswith(".txt"):
        return "txt"
    elif filename.endswith(".pdf"):
        return "pdf"
    elif filename.endswith(".md"):
        return "md"
    elif filename.endswith(".docx"):
        return "docx"
    else:
        return ""

# ----------- Streamlit UI -----------
st.set_page_config(page_title="AI Job Application Agent", layout="centered")
st.title("🧑‍💼 AI Job Application Assistant")
st.markdown(
    "Upload your **resume** and a **job description** (in `.txt`, `.pdf`, `.md`, or `.docx`).\n"
    "The app analyzes both, identifies skill gaps, generates a tailored resume, and writes a cover letter using OpenAI's GPT-4o."
)

resume_file = st.file_uploader("Upload Resume (.txt, .pdf, .md, .docx)", type=["txt", "pdf", "md", "docx"])
job_desc_file = st.file_uploader("Upload Job Description (.txt, .pdf, .md, .docx)", type=["txt", "pdf", "md", "docx"])

api_key = st.text_input("Enter your OpenAI API key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# ----------- On Button Click -----------
if st.button("Run AI Job Agent"):
    if not resume_file or not job_desc_file or not api_key:
        st.warning("Please upload both files and provide your OpenAI API key.")
        st.stop()

    resume_type = get_file_type(resume_file)
    job_type = get_file_type(job_desc_file)

    try:
        resume_text = extract_text(resume_file, resume_type)
    except Exception as e:
        st.error(f"Error reading resume: {e}")
        st.stop()

    try:
        job_desc_text = extract_text(job_desc_file, job_type)
    except Exception as e:
        st.error(f"Error reading job description: {e}")
        st.stop()

    if not resume_text.strip() or not job_desc_text.strip():
        st.error("One or both files are empty or could not be read. Please check your uploads.")
        st.stop()

    # ---------- Prompts ----------
    resume_prompt = """You are a resume analysis agent. Given the following resume, extract:
- Key technical skills
- Key soft skills
- Relevant work experience (job titles and industries)
Resume:
{resume}
Extracted Information:
"""

    job_desc_prompt = """You are a job description analysis agent. Given the following job posting, extract:
- Must-have technical skills
- Must-have soft skills
- Required experience (titles, years, industries)
Job Description:
{job_desc}
Extracted Requirements:
"""

    gap_prompt = """You are an expert job application coach. Given the extracted resume info and job description requirements, compare the two and identify:
- Which requirements are met by the resume
- Which requirements are missing or weak
Resume Information:
{resume_info}
Job Requirements:
{job_requirements}
Comparison:
"""

    resume_modifier_prompt = """
You are an expert resume editor and career coach. Using the provided resume, job description, and gap analysis, rewrite the resume to best fit the job. You may:
- Reorder or rephrase bullets to match job requirements
- Emphasize relevant skills and experiences
- Move less relevant content down or omit it for brevity
- Ensure all information remains true to the original resume (do not invent experience or skills)
- Output a complete, well-formatted resume (keep to 1-2 pages, if possible)

Original Resume:
{resume}

Job Description:
{job_desc}

Gap Analysis:
{gap_analysis}

Modified Resume:
"""

    cover_letter_prompt = """You are a cover letter generator. Write a tailored cover letter (max 200 words) using the provided resume and job description. Highlight the strongest matches, and mention genuine interest in the role. Be concise and professional.
Resume Information:
{resume_info}
Job Requirements:
{job_requirements}
Gap Analysis:
{gap_analysis}
Cover Letter:
"""

    # ----------- LLM Setup -----------
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4, streaming=False)

    resume_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["resume"], template=resume_prompt)
    )

    job_desc_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["job_desc"], template=job_desc_prompt)
    )

    gap_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["resume_info", "job_requirements"], template=gap_prompt)
    )

    resume_modifier_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["resume", "job_desc", "gap_analysis"],
            template=resume_modifier_prompt
        )
    )

    cover_letter_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["resume_info", "job_requirements", "gap_analysis"],
            template=cover_letter_prompt
        )
    )

    with st.spinner("Analyzing resume and job description..."):
        # Step 1: Analyze resume
        resume_info = resume_chain.run(resume=resume_text)
        st.subheader("Resume Information Extracted")
        st.code(resume_info, language="markdown")

        # Step 2: Analyze job description
        job_requirements = job_desc_chain.run(job_desc=job_desc_text)
        st.subheader("Job Requirements Extracted")
        st.code(job_requirements, language="markdown")

        # Step 3: Gap analysis
        gap_analysis = gap_chain.run(
            resume_info=resume_info,
            job_requirements=job_requirements
        )
        st.subheader("Gap Analysis")
        st.code(gap_analysis, language="markdown")

        # Step 4: Modify resume to fit job
        modified_resume = resume_modifier_chain.run(
            resume=resume_text,
            job_desc=job_desc_text,
            gap_analysis=gap_analysis
        )
        st.subheader("Modified Resume (AI-Tailored for Job Description)")
        st.code(modified_resume, language="markdown")
        st.download_button("Download Modified Resume", modified_resume, file_name="modified_resume.txt")

        # Step 5: Generate cover letter (using extracted info and gap analysis)
        cover_letter = cover_letter_chain.run(
            resume_info=resume_info,
            job_requirements=job_requirements,
            gap_analysis=gap_analysis
        )
        st.subheader("Tailored Cover Letter")
        st.write(cover_letter)
        st.download_button("Download Cover Letter", cover_letter, file_name="cover_letter.txt")
        st.success("Done! You can copy or download the results above for your application.")

st.caption("Built with 🦜 LangChain + Streamlit")
