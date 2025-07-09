import streamlit as st
import io
import PyPDF2
import docx
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ----------- Util: Text Extraction -----------
def extract_text(file, file_type):
    try:
        if file_type == "txt":
            return file.read().decode("utf-8")
        elif file_type == "pdf":
            file.seek(0)
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif file_type == "md":
            return file.read().decode("utf-8")
        elif file_type == "docx":
            file.seek(0)
            doc_file = docx.Document(io.BytesIO(file.read()))
            return "\n".join([para.text for para in doc_file.paragraphs])
        else:
            return ""
    except Exception as e:
        return f"ERROR: {e}"

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
    """
    Upload your **resume** and a **job description** (`.txt`, `.pdf`, `.md`, `.docx`).  
    The app analyzes both, identifies skill gaps, and generates a tailored cover letter using OpenAI's GPT-4o.
    """
)

with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
        1. Upload your resume.
        2. Upload the job description.
        3. Click **Run AI Job Agent** and review the results!
        4. Download your tailored cover letter.

        ---
        #### Maintainers/Developers:
        - Set your OpenAI API key in the Streamlit Cloud UI under the **Secrets** section as:
            ```toml
            [openai]
            api_key = "sk-..."
            ```
        - No need to add secrets.toml to your repo.
        """
    )
    st.caption("All data is processed in-memory. The OpenAI API key is securely managed on Streamlit Cloud.")

resume_file = st.file_uploader("Upload Resume (.txt, .pdf, .md, .docx)", type=["txt", "pdf", "md", "docx"])
job_desc_file = st.file_uploader("Upload Job Description (.txt, .pdf, .md, .docx)", type=["txt", "pdf", "md", "docx"])

run_button = st.button("Run AI Job Agent", type="primary")

if run_button:
    if not resume_file or not job_desc_file:
        st.warning("Please upload both files.")
        st.stop()

    resume_type = get_file_type(resume_file)
    job_type = get_file_type(job_desc_file)

    resume_text = extract_text(resume_file, resume_type)
    job_desc_text = extract_text(job_desc_file, job_type)

    if resume_text.startswith("ERROR:"):
        st.error(f"Error reading resume: {resume_text}")
        st.stop()
    if job_desc_text.startswith("ERROR:"):
        st.error(f"Error reading job description: {job_desc_text}")
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
    try:
        api_key = st.secrets["openai"]["api_key"]
        llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o", temperature=0.4, streaming=False)
    except Exception as e:
        st.error(f"Failed to load OpenAI API key from secrets: {e}")
        st.stop()

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
    cover_letter_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["resume_info", "job_requirements", "gap_analysis"],
            template=cover_letter_prompt
        )
    )

    with st.spinner("Analyzing resume and job description..."):
        # Step 1: Analyze resume
        try:
            resume_info = resume_chain.run(resume=resume_text)
            st.subheader("Resume Information Extracted")
            st.code(resume_info, language="markdown")
        except Exception as e:
            st.error(f"LLM error while analyzing resume: {e}")
            st.stop()

        # Step 2: Analyze job description
        try:
            job_requirements = job_desc_chain.run(job_desc=job_desc_text)
            st.subheader("Job Requirements Extracted")
            st.code(job_requirements, language="markdown")
        except Exception as e:
            st.error(f"LLM error while analyzing job description: {e}")
            st.stop()

        # Step 3: Gap analysis
        try:
            gap_analysis = gap_chain.run(
                resume_info=resume_info,
                job_requirements=job_requirements
            )
            st.subheader("Gap Analysis")
            st.code(gap_analysis, language="markdown")
        except Exception as e:
            st.error(f"LLM error during gap analysis: {e}")
            st.stop()

        # Step 4: Generate cover letter
        try:
            cover_letter = cover_letter_chain.run(
                resume_info=resume_info,
                job_requirements=job_requirements,
                gap_analysis=gap_analysis
            )
            st.subheader("Tailored Cover Letter")
            st.write(cover_letter)
            st.download_button(
                label="Download Cover Letter",
                data=cover_letter,
                file_name="cover_letter.txt",
                mime="text/plain"
            )
            st.success("Done! You can copy or download the cover letter above for your application.")
        except Exception as e:
            st.error(f"LLM error while generating cover letter: {e}")

st.caption("Built with 🦜 LangChain + Streamlit | See README for deployment instructions.")
