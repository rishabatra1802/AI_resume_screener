import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def calculate_match_percentage(job_requirements, resume_text):
    # Preprocess both texts
    processed_requirements = preprocess_text(job_requirements)
    processed_resume = preprocess_text(resume_text)
    
    # Create vectors
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([processed_requirements, processed_resume])
    
    # Calculate similarity
    similarity = cosine_similarity(vectors)[0][1]
    
    return similarity * 100

def main():
    st.title("AI Resume Screening System 📄")
    
    # Load sample resumes
    df = pd.read_csv('sample_resumes.csv')
    
    # Job Requirements Input
    st.header("Job Requirements")
    job_requirements = st.text_area(
        "Enter the job requirements:",
        "Example: Looking for a Python developer with 3+ years of experience in Machine Learning and Data Analysis. Must have knowledge of SQL and AWS."
    )
    
    if st.button("Screen Resumes"):
        st.header("Results")
        
        # Calculate match percentage for each resume
        results = []
        for _, row in df.iterrows():
            resume_text = f"{row['Skills']} {row['Experience']} years experience {row['Education']}"
            match_percentage = calculate_match_percentage(job_requirements, resume_text)
            results.append({
                'Name': row['Name'],
                'Match Percentage': match_percentage,
                'Skills': row['Skills'],
                'Experience': row['Experience'],
                'Education': row['Education'],
                'Location': row['Location']
            })
        
        # Sort results by match percentage
        results = sorted(results, key=lambda x: x['Match_Percentage'], reverse=True)
        
        # Display results
        for i, result in enumerate(results, 1):
            with st.expander(f"{i}. {result['Name']} - Match: {result['Match Percentage']:.1f}%"):
                st.write(f"**Skills:** {result['Skills']}")
                st.write(f"**Experience:** {result['Experience']} years")
                st.write(f"**Education:** {result['Education']}")
                st.write(f"**Location:** {result['Location']}")
                
                # Create a progress bar for match percentage
                st.progress(result['Match Percentage'] / 100)

if __name__ == "__main__":
    main()