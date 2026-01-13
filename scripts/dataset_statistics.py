"""
Dataset Statistics Generator

This script analyzes the benchmark synthetic datasets and generates comprehensive
statistics and metadata files for each dataset.
"""

import pandas as pd
import json
import os
from pathlib import Path
from collections import Counter
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

def analyze_individual_conversations(df):
    """Analyze each conversation individually to understand topics and themes."""
    if 'conversation_id' not in df.columns or 'message' not in df.columns:
        return {"error": "Required columns not found for conversation analysis"}
    
    conversation_analyses = {}
    conversation_summaries = []
    
    # Process each conversation individually
    for conv_id in df['conversation_id'].unique():
        conv_data = df[df['conversation_id'] == conv_id].sort_values('turn_index' if 'turn_index' in df.columns else df.index)
        
        # Combine all messages in the conversation
        full_conversation = ' '.join(conv_data['message'].dropna().astype(str))
        
        # Extract key information about this conversation
        analysis = {
            "conversation_id": str(conv_id),
            "total_turns": len(conv_data),
            "total_words": len(full_conversation.split()),
            "key_terms": [],
            "conversation_summary": ""
        }
        
        # Extract key terms (simple approach - most frequent meaningful words)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', full_conversation.lower())
        
        # Basic stopwords
        stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 
                     'each', 'which', 'their', 'time', 'about', 'when', 'what', 'where', 'there',
                     'here', 'your', 'would', 'could', 'should', 'might', 'must', 'shall', 'need',
                     'want', 'like', 'just', 'also', 'even', 'only', 'well', 'know', 'think',
                     'make', 'take', 'come', 'good', 'help', 'work', 'look', 'first', 'last'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_freq = Counter(filtered_words)
        
        # Get top 10 key terms for this conversation
        analysis["key_terms"] = [word for word, count in word_freq.most_common(10)]
        
        # Create a simple summary based on first user message (usually contains the main topic)
        user_messages = conv_data[conv_data['role'] == 'user']['message'] if 'role' in conv_data.columns else conv_data['message']
        if len(user_messages) > 0:
            first_message = str(user_messages.iloc[0])
            # Take first sentence or first 100 characters as summary
            summary = first_message.split('.')[0][:100] + "..." if len(first_message) > 100 else first_message
            analysis["conversation_summary"] = summary
        
        conversation_analyses[str(conv_id)] = analysis
        conversation_summaries.append({
            "id": str(conv_id),
            "summary": analysis["conversation_summary"],
            "key_terms": analysis["key_terms"][:5],  # Top 5 terms
            "turns": analysis["total_turns"]
        })
    
    return {
        "total_conversations": len(conversation_analyses),
        "individual_analyses": conversation_analyses,
        "conversation_summaries": conversation_summaries[:20],  # Show first 20 for overview
        "method": "Individual conversation analysis"
    }

def extract_conversation_topics(df, num_topics=10):
    """Extract topics by analyzing individual conversations and then clustering them."""
    try:
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback to basic English stopwords if NLTK fails
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                     'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
                     'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                     'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                     'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                     'further', 'then', 'once', 'help', 'need', 'want', 'like', 'know', 'think', 'make',
                     'take', 'come', 'good', 'work', 'look', 'first', 'last', 'time', 'also', 'well'}
    
    if 'conversation_id' not in df.columns or 'message' not in df.columns:
        return {"error": "Required columns not found for topic extraction"}
    
    # Get individual conversation analysis first
    individual_analysis = analyze_individual_conversations(df)
    
    # Combine all messages per conversation for clustering
    conversation_data = []
    conversation_ids = []
    
    for conv_id in df['conversation_id'].unique():
        conv_messages = df[df['conversation_id'] == conv_id]['message'].dropna().astype(str)
        full_conversation = ' '.join(conv_messages)
        conversation_data.append(full_conversation)
        conversation_ids.append(str(conv_id))
    
    if len(conversation_data) < 2:
        return {
            "individual_conversations": individual_analysis,
            "error": "Not enough conversations for topic clustering"
        }
    
    try:
        # Create TF-IDF vectors for clustering
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words=list(stop_words),
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        
        tfidf_matrix = vectorizer.fit_transform(conversation_data)
        
        # Perform clustering
        n_clusters = min(num_topics, len(conversation_data) // 3, 8)  # Max 8 topics
        if n_clusters < 2:
            n_clusters = 2
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics with example conversations
        topics = {}
        for i in range(n_clusters):
            # Get the centroid for this cluster
            centroid = kmeans.cluster_centers_[i]
            # Get top terms
            top_indices = centroid.argsort()[-8:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            # Get conversations in this cluster
            cluster_conversations = [conversation_ids[j] for j, label in enumerate(cluster_labels) if label == i]
            cluster_count = len(cluster_conversations)
            
            # Get example conversation summaries for this topic
            example_conversations = []
            for conv_id in cluster_conversations[:3]:  # Show up to 3 examples
                if conv_id in individual_analysis["individual_analyses"]:
                    conv_info = individual_analysis["individual_analyses"][conv_id]
                    example_conversations.append({
                        "conversation_id": conv_id,
                        "summary": conv_info["conversation_summary"][:150] + "..." if len(conv_info["conversation_summary"]) > 150 else conv_info["conversation_summary"],
                        "turns": conv_info["total_turns"]
                    })
            
            topics[f"topic_{i+1}"] = {
                "top_terms": top_terms,
                "conversation_count": int(cluster_count),
                "percentage": round(cluster_count / len(conversation_data) * 100, 2),
                "example_conversations": example_conversations
            }
        
        return {
            "total_topics_identified": n_clusters,
            "topics": topics,
            "individual_conversations": individual_analysis,
            "method": "Individual conversation analysis + TF-IDF clustering"
        }
        
    except Exception as e:
        # Return individual analysis even if clustering fails
        return {
            "individual_conversations": individual_analysis,
            "clustering_error": f"Topic clustering failed: {str(e)}",
            "method": "Individual conversation analysis only"
        }

def analyze_turn_distribution(df):
    """Analyze the distribution of turns per conversation."""
    if 'conversation_id' not in df.columns or 'turn_index' not in df.columns:
        return {"error": "Required columns not found for turn analysis"}
    
    # Get max turn index per conversation (assuming turn_index starts from 1)
    turns_per_conv = df.groupby('conversation_id')['turn_index'].max()
    
    # Create distribution
    turn_counts = turns_per_conv.value_counts().sort_index()
    
    distribution = {
        "statistics": {
            "mean": float(turns_per_conv.mean()),
            "median": float(turns_per_conv.median()),
            "std": float(turns_per_conv.std()),
            "min": int(turns_per_conv.min()),
            "max": int(turns_per_conv.max()),
            "total_conversations": len(turns_per_conv)
        },
        "distribution": {
            int(k): int(v) for k, v in turn_counts.items()
        },
        "percentiles": {
            "25th": float(turns_per_conv.quantile(0.25)),
            "50th": float(turns_per_conv.quantile(0.50)),
            "75th": float(turns_per_conv.quantile(0.75)),
            "90th": float(turns_per_conv.quantile(0.90)),
            "95th": float(turns_per_conv.quantile(0.95))
        }
    }
    
    return distribution

def analyze_coding_dataset(file_path):
    """Analyze the coding dataset."""
    print(f"Analyzing coding dataset: {file_path}")
    
    # Try different CSV parsing options to handle potential formatting issues
    try:
        df = pd.read_csv(file_path)
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(file_path, quoting=1)  # QUOTE_ALL
        except pd.errors.ParserError:
            df = pd.read_csv(file_path, sep=',', quotechar='"', escapechar='\\', on_bad_lines='skip')
    
    stats = {
        "dataset_name": "coding",
        "description": "Synthetic dataset containing coding-related conversations",
        "file_path": str(file_path),
        "turn_distribution": {},
        "conversation_topics": {}
    }
    
    # Analyze turn distribution
    stats["turn_distribution"] = analyze_turn_distribution(df)
    
    # Extract conversation topics
    print("  Extracting conversation topics...")
    try:
        stats["conversation_topics"] = extract_conversation_topics(df, num_topics=8)
    except Exception as e:
        print(f"    Warning: Topic extraction failed: {str(e)}")
        stats["conversation_topics"] = {"error": f"Topic extraction failed: {str(e)}"}
    
    return stats

def analyze_personal_chat_dataset(file_path):
    """Analyze the personal chat dataset."""
    print(f"Analyzing personal chat dataset: {file_path}")
    
    # Try different CSV parsing options to handle potential formatting issues
    try:
        df = pd.read_csv(file_path)
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(file_path, quoting=1)  # QUOTE_ALL
        except pd.errors.ParserError:
            df = pd.read_csv(file_path, sep=',', quotechar='"', escapechar='\\', on_bad_lines='skip')
    
    stats = {
        "dataset_name": "personal_chat",
        "description": "Synthetic dataset containing personal chat conversations",
        "file_path": str(file_path),
        "turn_distribution": {},
        "conversation_topics": {}
    }
    
    # Analyze turn distribution
    stats["turn_distribution"] = analyze_turn_distribution(df)
    
    # Extract conversation topics
    print("  Extracting conversation topics...")
    stats["conversation_topics"] = extract_conversation_topics(df, num_topics=8)
    
    return stats

def analyze_safety_protocol_dataset(file_path):
    """Analyze the safety protocol dataset."""
    print(f"Analyzing safety protocol dataset: {file_path}")
    
    # Try different CSV parsing options to handle potential formatting issues
    try:
        df = pd.read_csv(file_path)
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(file_path, quoting=1)  # QUOTE_ALL
        except pd.errors.ParserError:
            df = pd.read_csv(file_path, sep=',', quotechar='"', escapechar='\\', on_bad_lines='skip')
    
    stats = {
        "dataset_name": "safety_protocol",
        "description": "Synthetic dataset containing safety protocol documents",
        "file_path": str(file_path),
        "document_topics": {},
        "document_statistics": {}
    }
    
    # For safety protocols, we'll analyze document topics differently since it's not conversational
    if 'content' in df.columns:
        # Create a modified dataframe for topic analysis
        # Treat each document as a "conversation" with single message
        topic_df = pd.DataFrame({
            'conversation_id': df['document_id'] if 'document_id' in df.columns else range(len(df)),
            'message': df['content']
        })
        
        print("  Extracting document topics...")
        stats["document_topics"] = extract_conversation_topics(topic_df, num_topics=6)
        
        # Basic document statistics
        content_lengths = df['content'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        stats["document_statistics"] = {
            "total_documents": len(df),
            "word_count_stats": {
                "mean": float(content_lengths.mean()),
                "median": float(content_lengths.median()),
                "min": int(content_lengths.min()),
                "max": int(content_lengths.max())
            }
        }
        
        # Extract common themes from titles if available
        if 'title' in df.columns:
            all_titles = ' '.join(df['title'].dropna().astype(str))
            words = re.findall(r'\b[a-zA-Z]{4,}\b', all_titles.lower())
            # Basic stopwords filtering
            stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'about'}
            filtered_words = [word for word in words if word not in stop_words]
            common_words = Counter(filtered_words).most_common(15)
            stats["common_title_themes"] = dict(common_words)
    
    return stats

def save_metadata(stats, output_path):
    """Save statistics as JSON metadata file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to: {output_path}")

def main():
    """Main function to analyze all datasets."""
    base_path = Path("dataset/benchmark_synthetic_dataset")
    
    datasets = [
        {
            "name": "coding",
            "path": base_path / "coding" / "coding.csv",
            "analyzer": analyze_coding_dataset
        },
        {
            "name": "personal_chat", 
            "path": base_path / "personal_chat" / "personal_chat.csv",
            "analyzer": analyze_personal_chat_dataset
        },
        {
            "name": "safety_protocol",
            "path": base_path / "safety_protocol" / "safety_protocol.csv", 
            "analyzer": analyze_safety_protocol_dataset
        }
    ]
    
    all_stats = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {dataset['name']} dataset")
        print(f"{'='*50}")
        
        if not dataset["path"].exists():
            print(f"Warning: File not found: {dataset['path']}")
            continue
            
        try:
            stats = dataset["analyzer"](dataset["path"])
            all_stats[dataset["name"]] = stats
            
            # Save individual metadata file
            metadata_path = dataset["path"].parent / f"{dataset['name']}_metadata.json"
            save_metadata(stats, metadata_path)
            
        except Exception as e:
            print(f"Error analyzing {dataset['name']}: {str(e)}")
    
    # Save combined metadata
    combined_metadata_path = base_path / "combined_metadata.json"
    save_metadata(all_stats, combined_metadata_path)
    
    print(f"\n{'='*50}")
    print("Analysis complete!")
    print(f"{'='*50}")
    
    # Print summary
    for name, stats in all_stats.items():
        print(f"\n{name.upper()} DATASET SUMMARY:")
        
        if 'turn_distribution' in stats and 'statistics' in stats['turn_distribution']:
            turn_stats = stats['turn_distribution']['statistics']
            print(f"  Turn Distribution:")
            print(f"    Total conversations: {turn_stats.get('total_conversations', 'N/A')}")
            print(f"    Avg turns per conversation: {turn_stats.get('mean', 0):.1f}")
            print(f"    Median turns: {turn_stats.get('median', 0):.1f}")
            print(f"    Min-Max turns: {turn_stats.get('min', 0)}-{turn_stats.get('max', 0)}")
        
        if 'conversation_topics' in stats and 'total_topics_identified' in stats['conversation_topics']:
            topic_stats = stats['conversation_topics']
            print(f"  Conversation Topics: {topic_stats.get('total_topics_identified', 0)} identified")
            if 'topics' in topic_stats:
                for topic_id, topic_info in list(topic_stats['topics'].items())[:3]:  # Show first 3 topics
                    print(f"    {topic_id}: {topic_info.get('percentage', 0):.1f}% - {', '.join(topic_info.get('top_terms', [])[:5])}")
        
        if 'document_topics' in stats and 'total_topics_identified' in stats['document_topics']:
            topic_stats = stats['document_topics']
            print(f"  Document Topics: {topic_stats.get('total_topics_identified', 0)} identified")
            if 'document_statistics' in stats:
                doc_stats = stats['document_statistics']
                print(f"  Total documents: {doc_stats.get('total_documents', 'N/A')}")
                if 'word_count_stats' in doc_stats:
                    wc_stats = doc_stats['word_count_stats']
                    print(f"  Avg words per document: {wc_stats.get('mean', 0):.0f}")

if __name__ == "__main__":
    main()
