import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction import _stop_words

# Sample data structure - replace with your actual data loading
def load_course_data():
    """Load your course data - replace this with your actual data loading"""
    # For now, load from CSV
    try:
        df = pd.read_csv('data/courses_data_all.csv', encoding='utf-8')
        return df
    except FileNotFoundError:
        print("Course data not found. Please run data_clean_courses.py first.")
        return pd.DataFrame()

def prepare_documents(df):
    """Combine relevant text fields into a single document for embedding"""
    documents = []

    for _, row in df.iterrows():
        # Combine multiple text fields to create comprehensive document
        document_parts = []

        document_parts.append(str(row['courseCode']))

        if pd.notna(row.get('courseTitle')):
            document_parts.append(str(row['courseTitle']))

        if pd.notna(row.get('courseDescription')):
            document_parts.append(str(row['courseDescription']))

        if pd.notna(row.get('ilo')):
            document_parts.append(str(row['ilo']))

        if pd.notna(row.get('preRequisite')):
            document_parts.append(str(row['preRequisite']))

        # Combine all parts
        full_document = ". ".join(document_parts)
        documents.append(full_document)

    return documents

def setup_bertopic():
    """Set up BERTopic with all-MiniLM-L6-v2"""

    # Initialize sentence transformer
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Configure BERTopic components - FIXED HDBSCAN parameters
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    # FIX: Enable prediction_data for HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True  # This is crucial
    )

    # Define custom stop words - English + your custom words
    custom_stop_words = list(_stop_words.ENGLISH_STOP_WORDS)  # Base English stop words

    # Add your custom stop words
    additional_stop_words = [
        'course', 'courses', 'student', 'students', 'learning', 'teach', 'teaching',
        'study', 'studies', 'program', 'programs', 'university', 'college',
        'education', 'educational', 'academic', 'credit', 'credits',
        'coursetitle', 'coursecode', 'prerequisite', 'prerequisites',
        'cilo', 'objective', 'objectives', 'outcome', 'outcomes',
        'skill', 'skills', 'knowledge', 'understanding', 'ability',
        'develop', 'development', 'provide', 'provided', 'including',
        'required', 'requirement', 'requirements', 'basic', 'fundamental',
        'introduction', 'introductory', 'advanced', 'principle', 'principles',
        'research', 'qualitative', 'quantitative', 'equation', 'linear',
        'differential', 'texts',
        'cilo', 'ilo'
    ]

    custom_stop_words.extend(additional_stop_words)

    vectorizer_model = CountVectorizer(stop_words=custom_stop_words, min_df=2, ngram_range=(1, 2))

    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        language='english',
        calculate_probabilities=True,
        verbose=True
    )

    return topic_model

def main():
    # Load your data
    print("Loading course data...")
    df = load_course_data()
    if df.empty:
        return None, None, None, []

    # Prepare documents for topic modeling
    print("Preparing documents...")
    documents = prepare_documents(df)

    # Initialize BERTopic
    print("Setting up BERTopic...")
    topic_model = setup_bertopic()

    # Fit the model
    print("Fitting BERTopic model...")
    topics, probabilities = topic_model.fit_transform(documents)

    # Add topics back to original dataframe
    df['topic'] = topics
    df['topic_probability'] = probabilities.max(axis=1) if probabilities is not None else None

    # Display results
    print("\nTopic Information:")
    topic_info = topic_model.get_topic_info()
    print(topic_info)

    # Save results
    df.to_csv('data/courses_with_topics.csv', index=False)
    topic_model.save("data/bertopic_model")

    return df, topic_model, topic_info, documents

# Updated utility functions
def explore_topics(topic_model, specific_topic=None):
    """Explore specific topics or all topics"""
    if specific_topic is not None:
        print(f"\nTopic {specific_topic} keywords:")
        print(topic_model.get_topic(specific_topic))
    else:
        topic_info = topic_model.get_topic_info()
        for topic_id in topic_info['Topic'].values:
            if topic_id != -1:  # Skip outlier topic
                print(f"\nTopic {topic_id}:")
                print(topic_model.get_topic(topic_id))

def find_similar_courses(topic_model, course_index, documents, top_n=5):
    """Find courses similar to a specific course"""
    try:
        similar_docs, similarity = topic_model.find_similar_documents(
            documents[course_index],
            documents,
            top_n=top_n
        )

        print(f"\nCourses similar to: {documents[course_index][:100]}...")
        for i, (doc_idx, sim_score) in enumerate(zip(similar_docs, similarity)):
            print(f"{i+1}. Similarity: {sim_score:.3f} - {documents[doc_idx][:100]}...")
    except Exception as e:
        print(f"Error finding similar documents: {e}")
        print("Using alternative similarity approach...")
        alternative_find_similar(topic_model, course_index, documents, top_n)

def alternative_find_similar(topic_model, course_index, documents, top_n=5):
    """Alternative method to find similar courses using embeddings"""
    # Get embeddings for all documents
    embeddings = topic_model.embedding_model.embed(documents)

    # Calculate cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    target_embedding = embeddings[course_index].reshape(1, -1)
    similarities = cosine_similarity(target_embedding, embeddings)[0]

    # Get top N most similar (excluding the document itself)
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

    print(f"\nCourses similar to: {documents[course_index][:100]}...")
    for i, idx in enumerate(similar_indices):
        print(f"{i+1}. Similarity: {similarities[idx]:.3f} - {documents[idx][:100]}...")

if __name__ == "__main__":
    df, topic_model, topic_info, documents = main()

    if topic_model is not None:
        # Explore the first few topics
        explore_topics(topic_model, specific_topic=0)

        # Example: Find similar courses to the first one
        if len(documents) > 1:
            find_similar_courses(topic_model, 0, documents)