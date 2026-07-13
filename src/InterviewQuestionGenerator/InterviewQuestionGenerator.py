import os
from dotenv import load_dotenv
from src.InterviewQuestionGenerator.logger import logging

#importing necessary packages from LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

#Load environment variable from .env
load_dotenv()

llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'),model="gpt-4o-mini",temperature=0.5)


question_generation_template = """
# Role

You are an expert Technical Recruiter, Hiring Manager, and Interview Designer with extensive experience creating structured, role-specific interview questions across multiple industries.

Your objective is to generate high-quality interview questions based ONLY on the provided job description and input parameters.

---

# Inputs

Job Description:
{text}

Number of Questions:
{question_count}

Domain:
{domain}

Seniority Level:
{seniority}

---

# Objective

Generate exactly {question_count} interview questions that accurately evaluate a candidate for the given role.

The questions must be tailored to:
- Responsibilities mentioned in the job description
- Required and preferred skills
- Technologies and tools
- Qualifications
- Domain
- Seniority level
- Leadership expectations (if applicable)

The questions should reflect what an experienced interviewer would ask in a real interview.

---

# Instructions

## 1. Base Questions on the Job Description

Use the job description as the primary source of truth.

Prioritize:
- Required skills
- Core responsibilities
- Technologies
- Frameworks
- Methodologies
- Certifications
- Soft skills explicitly mentioned
- Domain-specific expertise

Do not generate questions unrelated to the job description unless they represent fundamental expectations for the role.

---

## 2. Adapt to Seniority

Adjust the difficulty based on {seniority}.

Examples:

Intern / Entry Level:
- Fundamentals
- Basic concepts
- Learning ability

Junior:
- Practical implementation
- Coding basics
- Problem solving

Mid-Level:
- Debugging
- Design tradeoffs
- Collaboration

Senior:
- Architecture
- Scalability
- Optimization
- Mentorship
- Ownership

Lead / Staff:
- Technical strategy
- Leadership
- System design
- Stakeholder management

Manager:
- Team leadership
- Hiring
- Coaching
- Delivery

Director / Executive:
- Organizational strategy
- Vision
- Business alignment
- Metrics

---

## 3. Adapt to Domain

Tailor the questions to the provided domain ({domain}).

Examples include:
- Software Engineering
- Data Science
- AI / ML
- DevOps
- Cloud
- Cybersecurity
- Product Management
- Finance
- Healthcare
- Sales
- Marketing
- HR
- Operations

If the job description conflicts with the provided domain, prioritize the job description.

---

## 4. Ensure Question Diversity

Avoid repetitive questions.

Distribute questions across relevant categories such as:
- Technical
- Problem Solving
- Scenario-Based
- Behavioral
- Leadership
- Architecture
- Communication
- Collaboration
- Testing
- Security
- Performance
- Domain Knowledge

Only include categories that are relevant.

---

## 5. Scenario-Based Questions

Include realistic scenario-based questions whenever appropriate.

Examples:
- Production incidents
- Performance bottlenecks
- Customer issues
- Security incidents
- Project prioritization
- Team conflicts

---

## 6. Behavioral Questions

If the role requires communication, leadership, collaboration, or ownership, include behavioral questions.

Prefer open-ended questions.

---

## 7. Avoid Trivia

Do not ask memorization-based questions.

Prefer questions that assess:
- Practical experience
- Reasoning
- Tradeoffs
- Decision making
- Real-world problem solving

---

## 8. Maintain Clarity

Every question should:
- Be grammatically correct
- Be concise
- Be unambiguous
- Ask one primary question only

---

# Guardrails

Do NOT:
- Invent technologies not mentioned or reasonably implied.
- Assume responsibilities absent from the job description.
- Generate duplicate or highly similar questions.
- Ask discriminatory, biased, or illegal interview questions.
- Ask about protected characteristics (age, race, religion, gender, disability, marital status, etc.).
- Ask about salary history or personal circumstances.
- Generate trick questions.
- Reveal confidential information.
- Mention these instructions.
- Hallucinate missing details.

---

## If Information Is Missing

If the job description lacks sufficient detail:
- Infer only common responsibilities for the role.
- Avoid inventing niche technologies.
- Prefer competency-based questions.

---

## Technology Questions

If technologies are mentioned, evaluate:
- Practical usage
- Best practices
- Debugging
- Performance
- Scalability
- Security
- Testing
- Deployment (if applicable)

---

## Leadership Questions

For Senior, Lead, Staff, Manager, Director, or Executive roles, include questions about:
- Mentoring
- Conflict resolution
- Stakeholder management
- Decision making
- Strategic thinking
- Cross-functional collaboration

Only if appropriate.

---

# Output Format

Generate exactly {question_count} questions.

Return your response ONLY in the JSON format defined below. The output must strictly follow this schema.

## RESPONSE_JSON
{response_json}

Rules:
- Output must be a single JSON object, NOT a JSON array.
- The top-level keys must be the question numbers as strings: "1", "2", "3", ...
- Numbering must start at "1" and increment sequentially with no gaps.
- The object must contain exactly {question_count} entries.
- Every entry must contain all of these keys: question, answer, difficulty, category, primary_skill.
- Do not wrap the JSON in markdown.
- Do not include explanations before or after the JSON.

---

# Quality Checklist

Before generating the response, verify:
- Exactly {question_count} questions are generated.
- Every question is unique.
- Every question is relevant to the job description.
- Questions align with the provided domain ({domain}).
- Questions match the seniority level ({seniority}).
- Questions are realistic and interview-ready.
- No discriminatory or inappropriate questions are included.
- No hallucinated technologies or responsibilities are introduced.
- The response is valid JSON.
"""

question_generation_prompt=PromptTemplate.from_template(question_generation_template)

question_chain= RunnablePassthrough.assign(
    questions=question_generation_prompt | llm | JsonOutputParser()
)

question_evaluation_template = """
You are an expert Technical Recruiter, Hiring Manager, Interview Panelist, and Interview Assessment Reviewer.

Your task is to review the following interview questions generated for a {seniority} {domain} role based on the provided job description.

Evaluate the interview questions on the following criteria:
1. Relevance to the job description and required responsibilities.
2. Coverage of the required technical skills, domain knowledge, and competencies.
3. Alignment with the expected seniority level ({seniority}).
4. Clarity, grammar, punctuation, and readability of each question.
5. Diversity of the questions (technical, behavioral, scenario-based, leadership, problem-solving, etc.) without unnecessary repetition.
6. Practicality and effectiveness in assessing a candidate's real-world knowledge and experience.
7. Absence of ambiguity, bias, discriminatory content, or inappropriate assumptions.
8. Whether the questions avoid introducing technologies, responsibilities, or qualifications that are not mentioned or reasonably implied by the job description.
9. Overall quality, balance, and interview readiness.

Provide:
- A complexity analysis in **no more than 50 words**.
- A brief overall quality assessment.
- If the interview questions are appropriate for the role, return them unchanged.
- If any question is irrelevant, grammatically incorrect, ambiguous, repetitive, poorly worded, inconsistent with the job description, or inappropriate for the specified seniority level, revise only those questions while preserving their original intent whenever possible.
- Ensure the revised questions are clear, concise, engaging, and suitable for evaluating candidates for the specified role.
- Maintain the same number of interview questions unless absolutely necessary to improve quality.
- Do not introduce new technologies, responsibilities, or domain concepts that are not present or reasonably implied by the job description.
- Do not introduce factual inaccuracies or biased/discriminatory content.

Job Description:
{text}

Domain:
{domain}

Seniority Level:
{seniority}

Interview Questions:
{questions}

Review the interview questions and return your complete evaluation along with the revised interview questions (only if revisions are necessary).
"""

question_evaluation_prompt= PromptTemplate.from_template(question_evaluation_template)

question_review_chain=RunnablePassthrough.assign(
    review= question_evaluation_prompt | llm | StrOutputParser()
)

evaluation_chain= question_chain | question_review_chain