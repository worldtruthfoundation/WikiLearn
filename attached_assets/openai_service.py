import openai
import logging
from config import OPENAI_API_KEY

log = logging.getLogger(__name__)

# Initialize OpenAI client with timeout settings
client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=60.0,  # 60 second timeout
    max_retries=2  # Retry failed requests twice
) if OPENAI_API_KEY else None

def generate_summary(article_title, english_level):
    """Generate an article summary using OpenAI API"""
    if not client:
        raise Exception("OpenAI API key not configured")
    
    # Define complexity based on English level
    complexity = {
        'elementary': 'Use simple vocabulary and grammar suitable for A1-A2 English level. Use short sentences (maximum 8-10 words). Avoid complex words, idioms, and phrasal verbs. Use simple present and past tenses only. Explain any technical terms.',
        'intermediate': 'Use vocabulary and grammar suitable for B1-B2 English level. Use a mix of simple and compound sentences. Include some common idioms and phrasal verbs, but explain them if they are important to understanding. Use a variety of tenses.',
        'professional': 'Use advanced vocabulary and grammar suitable for C1-C2 English level. Use complex sentence structures, academic vocabulary, varied tenses, and specialized terminology appropriate to the subject. Do not simplify content.'
    }.get(english_level, 'Use vocabulary and grammar suitable for B1-B2 English level.')
    
    prompt = f"""
    Write a comprehensive summary about "{article_title}" for {english_level} level English learners.
    
    {complexity}
    
    Format requirements:
    1. Begin with a clear title
    2. Structure the content with clear paragraphs and headings
    3. Conclude with 2-3 sentences summarizing the main points
    
    Make the summary engaging, educational, and factually accurate. Maintain all important information while adapting the language to the appropriate level.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except openai.APIConnectionError as e:
        log.error(f"OpenAI API connection error: {e}")
        return f"<h1>Summary: {article_title}</h1><p>Unable to generate AI summary due to connection issues. Please try again later or check your internet connection.</p>"
    except openai.RateLimitError as e:
        log.error(f"OpenAI API rate limit error: {e}")
        return f"<h1>Summary: {article_title}</h1><p>Rate limit exceeded. Please wait a moment before trying again.</p>"
    except openai.APIError as e:
        log.error(f"OpenAI API error: {e}")
        return f"<h1>Summary: {article_title}</h1><p>Unable to generate AI summary. Please try again later.</p>"
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        return f"<h1>Summary: {article_title}</h1><p>An unexpected error occurred. Please try again later.</p>"

def generate_lesson(article_content, english_level):
    """Generate a comprehensive lesson using OpenAI API"""
    if not client:
        raise Exception("OpenAI API key not configured")
    
    level_description = {
        'elementary': 'A1-A2 level (elementary) - Use simple vocabulary and grammar. Focus on basic sentence structures, common everyday words, and simple present and past tenses.',
        'intermediate': 'B1-B2 level (intermediate) - Use a moderate range of vocabulary and grammar structures. Include some idioms and phrasal verbs with explanations. Use various tenses and conditional forms.',
        'professional': 'C1-C2 level (professional) - Use advanced vocabulary, complex grammar structures, idiomatic expressions, and academic language. Challenge the learner with sophisticated content.'
    }.get(english_level, 'B1-B2 level (intermediate)')
    
    prompt = f"""
    Create a comprehensive English language lesson plan based on the following Wikipedia article, suitable for {level_description} English learners.
    
    Your lesson plan must include exactly these sections, formatted with clean HTML:
    
    <h1>English Lesson: [Topic]</h1>
    
    <h2>1. Vocabulary (Exactly 10 New Words)</h2>
    - Create a well-formatted HTML table with exactly 10 new vocabulary words from the article
    - For each word, include: Word, Part of Speech, Definition, Example Sentence 
    - Choose words appropriate for the {english_level} level English learner
    
    <h2>2. Reading Comprehension</h2>
    <h3>Summary</h3>
    - Create a level-appropriate summary of the article (3-4 paragraphs)
    
    <h3>True/False Questions</h3>
    - Create 5 true/false questions about the article
    - Format with expandable answers: <details><summary>Show answer</summary>True/False</details>
    
    <h3>Wh- Questions</h3>
    - Create 5 questions (What/Who/Where/When/Why/How)
    - Format with expandable answers: <details><summary>Show answer</summary>Answer</details>
    
    <h2>3. Word Matching Exercise</h2>
    - Create a two-column table (Word | Definition)
    - Add an Answer Key with expandable section
    
    <h2>4. Fill in the Blanks</h2>
    - Create one paragraph with 8-10 blanks
    - Show the Word Bank as an unordered list
    - Add expandable completed text section
    
    <h2>5. Discussion (Speaking Practice)</h2>
    - Provide 3 discussion questions related to the article topic
    - For each question, include 2-3 guiding points
    
    <h2>6. Essay Theme</h2>
    - Provide a writing prompt related to the article topic
    - Include specific requirements (word count, elements to include, etc.)
    - Tailor the difficulty to the {english_level} level
    
    Use proper HTML formatting with tables for all exercises. Make the lesson engaging, practical, and focused on real language use.
    
    Article content (first 2000 characters):
    {article_content[:2000]}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except openai.APIConnectionError as e:
        log.error(f"OpenAI API connection error: {e}")
        return "<h1>English Lesson</h1><p>Unable to generate lesson due to connection issues. Please check your OpenAI API key and try again later.</p>"
    except openai.RateLimitError as e:
        log.error(f"OpenAI API rate limit error: {e}")
        return "<h1>English Lesson</h1><p>Rate limit exceeded. Please wait a moment before trying again.</p>"
    except openai.APIError as e:
        log.error(f"OpenAI API error: {e}")
        return "<h1>English Lesson</h1><p>Unable to generate lesson. Please verify your OpenAI API key and try again.</p>"
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        return "<h1>English Lesson</h1><p>An unexpected error occurred. Please try again later.</p>"

def generate_exercise(article_title, english_level, exercise_type):
    """Generate specific exercises using OpenAI API"""
    if not client:
        raise Exception("OpenAI API key not configured")
    
    prompts = {
        'grammar': f"Create grammar exercises based on '{article_title}' for {english_level} level learners. Include 10 exercises with explanations.",
        'vocabulary': f"Create vocabulary exercises based on '{article_title}' for {english_level} level learners. Include word definitions, synonyms, and usage examples.",
        'extra': f"Create additional practice exercises based on '{article_title}' for {english_level} level learners. Include creative writing prompts and critical thinking questions."
    }
    
    prompt = prompts.get(exercise_type, prompts['extra'])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except openai.APIConnectionError as e:
        log.error(f"OpenAI API connection error: {e}")
        return f"<h1>{exercise_type.title()} Exercise</h1><p>Unable to generate exercises due to connection issues. Please try again later.</p>"
    except openai.RateLimitError as e:
        log.error(f"OpenAI API rate limit error: {e}")
        return f"<h1>{exercise_type.title()} Exercise</h1><p>Rate limit exceeded. Please wait a moment before trying again.</p>"
    except openai.APIError as e:
        log.error(f"OpenAI API error: {e}")
        return f"<h1>{exercise_type.title()} Exercise</h1><p>Unable to generate exercises. Please verify your OpenAI API key and try again.</p>"
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        return f"<h1>{exercise_type.title()} Exercise</h1><p>An unexpected error occurred. Please try again later.</p>"