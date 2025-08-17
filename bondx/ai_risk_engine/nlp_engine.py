"""
Natural Language Processing & Sentiment Analysis Engine

This module implements sophisticated NLP systems that process financial news,
earnings calls, management discussions, rating reports, and regulatory filings
to extract sentiment and fundamental insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import re
import json
from pathlib import Path

# NLP imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Text processing
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Deep Learning
from transformers import (
    pipeline, AutoTokenizer, AutoModel, 
    AutoModelForSequenceClassification, AutoModelForTokenClassification
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Types of financial documents"""
    NEWS_ARTICLE = "news_article"
    EARNINGS_CALL = "earnings_call"
    MANAGEMENT_DISCUSSION = "management_discussion"
    RATING_REPORT = "rating_report"
    REGULATORY_FILING = "regulatory_filing"
    PROSPECTUS = "prospectus"
    ANNUAL_REPORT = "annual_report"

class SentimentType(Enum):
    """Sentiment classifications"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class EntityType(Enum):
    """Types of entities"""
    ISSUER = "issuer"
    FINANCIAL_INSTRUMENT = "financial_instrument"
    PERSON = "person"
    ORGANIZATION = "organization"
    REGULATORY_BODY = "regulatory_body"
    LOCATION = "location"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"

@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    compound_score: float
    positive_score: float
    negative_score: float
    neutral_score: float
    sentiment_label: SentimentType
    confidence: float
    timestamp: datetime

@dataclass
class EntityExtraction:
    """Entity extraction result"""
    entity_text: str
    entity_type: EntityType
    confidence: float
    start_pos: int
    end_pos: int
    context: str

@dataclass
class TopicModeling:
    """Topic modeling result"""
    topic_id: int
    topic_keywords: List[str]
    topic_weight: float
    topic_coherence: float
    documents: List[str]

class NLPEngine:
    """
    Sophisticated NLP system for financial text analysis
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        # Load financial domain models
        self._load_financial_models()
        
        # Initialize sentiment analyzers
        self._initialize_sentiment_analyzers()
        
        # Financial terminology and patterns
        self._initialize_financial_patterns()
        
    def _initialize_nlp_components(self):
        """Initialize basic NLP components"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            
            # Initialize NLTK components
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
                
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
    
    def _load_financial_models(self):
        """Load financial domain-specific models"""
        try:
            # Load financial sentiment model
            self.financial_sentiment_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # Use CPU by default
            )
            
            # Load financial NER model
            self.financial_ner_model = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                device=-1
            )
            
            # Load sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
        except Exception as e:
            logger.warning(f"Financial models loading failed: {e}")
            self.financial_sentiment_model = None
            self.financial_ner_model = None
            self.sentence_transformer = None
    
    def _initialize_sentiment_analyzers(self):
        """Initialize sentiment analysis tools"""
        try:
            # VADER sentiment analyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Custom financial sentiment patterns
            self._add_financial_sentiment_patterns()
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzers: {e}")
    
    def _add_financial_sentiment_patterns(self):
        """Add financial domain-specific sentiment patterns"""
        try:
            # Financial positive words
            financial_positive = {
                'bullish': 2.0,
                'surge': 2.0,
                'rally': 2.0,
                'outperform': 1.5,
                'beat': 1.5,
                'exceed': 1.5,
                'strong': 1.0,
                'growth': 1.0,
                'profit': 1.0,
                'dividend': 0.5,
                'upgrade': 1.5,
                'positive': 1.0
            }
            
            # Financial negative words
            financial_negative = {
                'bearish': -2.0,
                'plunge': -2.0,
                'crash': -2.0,
                'underperform': -1.5,
                'miss': -1.5,
                'decline': -1.0,
                'weak': -1.0,
                'loss': -1.0,
                'downgrade': -1.5,
                'negative': -1.0,
                'default': -2.0,
                'bankruptcy': -3.0
            }
            
            # Update VADER lexicon
            self.vader_analyzer.lexicon.update(financial_positive)
            self.vader_analyzer.lexicon.update(financial_negative)
            
        except Exception as e:
            logger.error(f"Error adding financial sentiment patterns: {e}")
    
    def _initialize_financial_patterns(self):
        """Initialize financial terminology and patterns"""
        self.financial_terms = {
            'ratings': ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-'],
            'financial_metrics': ['EBITDA', 'ROE', 'ROA', 'P/E', 'P/B', 'Debt/Equity', 'Current Ratio'],
            'regulatory_bodies': ['RBI', 'SEBI', 'IRDAI', 'PFRDA', 'FSSAI'],
            'currencies': ['INR', 'USD', 'EUR', 'GBP', 'JPY'],
            'time_periods': ['Q1', 'Q2', 'Q3', 'Q4', 'FY', 'H1', 'H2']
        }
        
        # Financial regex patterns
        self.financial_patterns = {
            'percentage': r'\d+\.?\d*\s*%',
            'currency': r'[₹$€£¥]\s*\d+[\d,]*\.?\d*',
            'rating': r'\b(?:AAA|AA\+|AA|AA-|A\+|A|A-|BBB\+|BBB|BBB-|BB\+|BB|BB-|B\+|B|B-|C|D)\b',
            'financial_ratio': r'\b(?:ROE|ROA|P/E|P/B|Debt/Equity|Current Ratio)\b',
            'quarter': r'\b(?:Q[1-4]|FY|H[1-2])\b'
        }
    
    def preprocess_text(self, text: str, document_type: DocumentType = None) -> str:
        """
        Preprocess text for analysis
        
        Args:
            text: Raw text input
            document_type: Type of financial document
            
        Returns:
            Preprocessed text
        """
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Basic cleaning
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
            text = re.sub(r'[^\w\s\.\,\!\?\-\%\$]', '', text)  # Remove special characters
            
            # Document type specific preprocessing
            if document_type == DocumentType.EARNINGS_CALL:
                text = self._preprocess_earnings_call(text)
            elif document_type == DocumentType.RATING_REPORT:
                text = self._preprocess_rating_report(text)
            elif document_type == DocumentType.REGULATORY_FILING:
                text = self._preprocess_regulatory_filing(text)
            
            # Tokenization and lemmatization
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and lemmatize
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
            
            return ' '.join(processed_tokens)
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    def _preprocess_earnings_call(self, text: str) -> str:
        """Preprocess earnings call transcripts"""
        try:
            # Remove speaker labels
            text = re.sub(r'^[A-Z\s]+:', '', text, flags=re.MULTILINE)
            
            # Remove timestamps
            text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', '', text)
            
            # Remove question numbers
            text = re.sub(r'Q\d+', '', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing earnings call: {e}")
            return text
    
    def _preprocess_rating_report(self, text: str) -> str:
        """Preprocess rating reports"""
        try:
            # Remove rating symbols
            text = re.sub(r'[A-Z]{1,3}[+-]?', '', text)
            
            # Remove page numbers
            text = re.sub(r'Page \d+', '', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing rating report: {e}")
            return text
    
    def _preprocess_regulatory_filing(self, text: str) -> str:
        """Preprocess regulatory filings"""
        try:
            # Remove form numbers
            text = re.sub(r'Form [A-Z0-9-]+', '', text)
            
            # Remove section references
            text = re.sub(r'Section \d+', '', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing regulatory filing: {e}")
            return text
    
    def analyze_sentiment(
        self,
        text: str,
        method: str = "ensemble",
        document_type: DocumentType = None
    ) -> SentimentScore:
        """
        Analyze sentiment of financial text
        
        Args:
            text: Text to analyze
            method: Sentiment analysis method
            document_type: Type of financial document
            
        Returns:
            Sentiment analysis result
        """
        try:
            if not text:
                return self._create_default_sentiment()
            
            # Preprocess text
            processed_text = self.preprocess_text(text, document_type)
            
            if method == "vader":
                return self._vader_sentiment_analysis(processed_text)
            elif method == "financial":
                return self._financial_sentiment_analysis(processed_text)
            elif method == "textblob":
                return self._textblob_sentiment_analysis(processed_text)
            elif method == "ensemble":
                return self._ensemble_sentiment_analysis(processed_text)
            else:
                raise ValueError(f"Unsupported sentiment method: {method}")
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._create_default_sentiment()
    
    def _vader_sentiment_analysis(self, text: str) -> SentimentScore:
        """VADER sentiment analysis"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine sentiment label
            compound = scores['compound']
            if compound >= 0.5:
                sentiment_label = SentimentType.VERY_POSITIVE
            elif compound >= 0.1:
                sentiment_label = SentimentType.POSITIVE
            elif compound <= -0.5:
                sentiment_label = SentimentType.VERY_NEGATIVE
            elif compound <= -0.1:
                sentiment_label = SentimentType.NEGATIVE
            else:
                sentiment_label = SentimentType.NEUTRAL
            
            return SentimentScore(
                compound_score=compound,
                positive_score=scores['pos'],
                negative_score=scores['neg'],
                neutral_score=scores['neu'],
                sentiment_label=sentiment_label,
                confidence=abs(compound),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in VADER sentiment analysis: {e}")
            return self._create_default_sentiment()
    
    def _financial_sentiment_analysis(self, text: str) -> SentimentScore:
        """Financial domain-specific sentiment analysis"""
        try:
            if not self.financial_sentiment_model:
                return self._vader_sentiment_analysis(text)
            
            # Use FinBERT for financial sentiment
            result = self.financial_sentiment_model(text[:512])  # Limit length
            
            # Map FinBERT labels to our sentiment types
            label_mapping = {
                'positive': SentimentType.POSITIVE,
                'negative': SentimentType.NEGATIVE,
                'neutral': SentimentType.NEUTRAL
            }
            
            sentiment_label = label_mapping.get(result[0]['label'], SentimentType.NEUTRAL)
            confidence = result[0]['score']
            
            # Convert to our format
            if sentiment_label == SentimentType.POSITIVE:
                compound = confidence
                positive_score = confidence
                negative_score = 1 - confidence
                neutral_score = 0.0
            elif sentiment_label == SentimentType.NEGATIVE:
                compound = -confidence
                positive_score = 1 - confidence
                negative_score = confidence
                neutral_score = 0.0
            else:
                compound = 0.0
                positive_score = 0.0
                negative_score = 0.0
                neutral_score = confidence
            
            return SentimentScore(
                compound_score=compound,
                positive_score=positive_score,
                negative_score=negative_score,
                neutral_score=neutral_score,
                sentiment_label=sentiment_label,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in financial sentiment analysis: {e}")
            return self._vader_sentiment_analysis(text)
    
    def _textblob_sentiment_analysis(self, text: str) -> SentimentScore:
        """TextBlob sentiment analysis"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convert polarity to our scale
            compound = polarity
            positive_score = max(0, polarity)
            negative_score = max(0, -polarity)
            neutral_score = 1 - abs(polarity)
            
            # Determine sentiment label
            if polarity >= 0.3:
                sentiment_label = SentimentType.POSITIVE
            elif polarity <= -0.3:
                sentiment_label = SentimentType.NEGATIVE
            else:
                sentiment_label = SentimentType.NEUTRAL
            
            return SentimentScore(
                compound_score=compound,
                positive_score=positive_score,
                negative_score=negative_score,
                neutral_score=neutral_score,
                sentiment_label=sentiment_label,
                confidence=subjectivity,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in TextBlob sentiment analysis: {e}")
            return self._create_default_sentiment()
    
    def _ensemble_sentiment_analysis(self, text: str) -> SentimentScore:
        """Ensemble sentiment analysis combining multiple methods"""
        try:
            # Get sentiment from all methods
            vader_sentiment = self._vader_sentiment_analysis(text)
            financial_sentiment = self._financial_sentiment_analysis(text)
            textblob_sentiment = self._textblob_sentiment_analysis(text)
            
            # Weight the methods (financial gets higher weight for financial text)
            weights = {'vader': 0.3, 'financial': 0.5, 'textblob': 0.2}
            
            # Calculate weighted average
            compound_score = (
                vader_sentiment.compound_score * weights['vader'] +
                financial_sentiment.compound_score * weights['financial'] +
                textblob_sentiment.compound_score * weights['textblob']
            )
            
            positive_score = (
                vader_sentiment.positive_score * weights['vader'] +
                financial_sentiment.positive_score * weights['financial'] +
                textblob_sentiment.positive_score * weights['textblob']
            )
            
            negative_score = (
                vader_sentiment.negative_score * weights['vader'] +
                financial_sentiment.negative_score * weights['financial'] +
                textblob_sentiment.negative_score * weights['textblob']
            )
            
            neutral_score = (
                vader_sentiment.neutral_score * weights['vader'] +
                financial_sentiment.neutral_score * weights['financial'] +
                textblob_sentiment.neutral_score * weights['textblob']
            )
            
            # Determine ensemble sentiment label
            if compound_score >= 0.3:
                sentiment_label = SentimentType.POSITIVE
            elif compound_score <= -0.3:
                sentiment_label = SentimentType.NEGATIVE
            else:
                sentiment_label = SentimentType.NEUTRAL
            
            # Calculate ensemble confidence
            confidence = (vader_sentiment.confidence + financial_sentiment.confidence + textblob_sentiment.confidence) / 3
            
            return SentimentScore(
                compound_score=compound_score,
                positive_score=positive_score,
                negative_score=negative_score,
                neutral_score=neutral_score,
                sentiment_label=sentiment_label,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble sentiment analysis: {e}")
            return self._vader_sentiment_analysis(text)
    
    def extract_entities(self, text: str) -> List[EntityExtraction]:
        """
        Extract named entities from financial text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities
        """
        try:
            entities = []
            
            # Use spaCy for entity extraction
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Map spaCy entity types to our types
                entity_type = self._map_spacy_entity_type(ent.label_)
                
                # Get context around entity
                start = max(0, ent.start_char - 50)
                end = min(len(text), ent.end_char + 50)
                context = text[start:end]
                
                entity = EntityExtraction(
                    entity_text=ent.text,
                    entity_type=entity_type,
                    confidence=ent.prob,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    context=context
                )
                
                entities.append(entity)
            
            # Extract financial-specific entities using regex
            financial_entities = self._extract_financial_entities(text)
            entities.extend(financial_entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _map_spacy_entity_type(self, spacy_type: str) -> EntityType:
        """Map spaCy entity types to our entity types"""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'MONEY': EntityType.CURRENCY,
            'PERCENT': EntityType.PERCENTAGE,
            'CARDINAL': EntityType.PERCENTAGE,
            'FAC': EntityType.ORGANIZATION,
            'PRODUCT': EntityType.FINANCIAL_INSTRUMENT
        }
        
        return mapping.get(spacy_type, EntityType.ORGANIZATION)
    
    def _extract_financial_entities(self, text: str) -> List[EntityExtraction]:
        """Extract financial-specific entities using regex patterns"""
        entities = []
        
        try:
            # Extract ratings
            rating_matches = re.finditer(self.financial_patterns['rating'], text, re.IGNORECASE)
            for match in rating_matches:
                entity = EntityExtraction(
                    entity_text=match.group(),
                    entity_type=EntityType.FINANCIAL_INSTRUMENT,
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=text[max(0, match.start()-30):match.end()+30]
                )
                entities.append(entity)
            
            # Extract percentages
            percentage_matches = re.finditer(self.financial_patterns['percentage'], text)
            for match in percentage_matches:
                entity = EntityExtraction(
                    entity_text=match.group(),
                    entity_type=EntityType.PERCENTAGE,
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=text[max(0, match.start()-30):match.end()+30]
                )
                entities.append(entity)
            
            # Extract currencies
            currency_matches = re.finditer(self.financial_patterns['currency'], text)
            for match in currency_matches:
                entity = EntityExtraction(
                    entity_text=match.group(),
                    entity_type=EntityType.CURRENCY,
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=text[max(0, match.start()-30):match.end()+30]
                )
                entities.append(entity)
            
        except Exception as e:
            logger.error(f"Error extracting financial entities: {e}")
        
        return entities
    
    def extract_topics(
        self,
        documents: List[str],
        n_topics: int = 5,
        method: str = "lda"
    ) -> List[TopicModeling]:
        """
        Extract topics from financial documents
        
        Args:
            documents: List of document texts
            n_topics: Number of topics to extract
            method: Topic modeling method
            
        Returns:
            List of extracted topics
        """
        try:
            if not documents:
                return []
            
            # Preprocess documents
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            
            # Vectorize documents
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            
            doc_term_matrix = vectorizer.fit_transform(processed_docs)
            feature_names = vectorizer.get_feature_names_out()
            
            if method == "lda":
                return self._extract_topics_lda(doc_term_matrix, feature_names, n_topics, documents)
            elif method == "nmf":
                return self._extract_topics_nmf(doc_term_matrix, feature_names, n_topics, documents)
            else:
                raise ValueError(f"Unsupported topic modeling method: {method}")
                
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    def _extract_topics_lda(
        self,
        doc_term_matrix,
        feature_names,
        n_topics: int,
        documents: List[str]
    ) -> List[TopicModeling]:
        """Extract topics using LDA"""
        try:
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
            
            lda.fit(doc_term_matrix)
            
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                # Get top keywords
                top_keywords_idx = topic.argsort()[-10:][::-1]
                top_keywords = [feature_names[i] for i in top_keywords_idx]
                
                # Calculate topic weights for each document
                doc_topics = lda.transform(doc_term_matrix)
                topic_weights = doc_topics[:, topic_idx]
                
                # Get documents most associated with this topic
                top_doc_indices = topic_weights.argsort()[-5:][::-1]
                top_docs = [documents[i][:100] + "..." for i in top_doc_indices]
                
                # Calculate topic coherence (simplified)
                topic_coherence = np.mean(topic_weights)
                
                topic = TopicModeling(
                    topic_id=topic_idx,
                    topic_keywords=top_keywords,
                    topic_weight=topic_coherence,
                    topic_coherence=topic_coherence,
                    documents=top_docs
                )
                
                topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error in LDA topic extraction: {e}")
            return []
    
    def _extract_topics_nmf(
        self,
        doc_term_matrix,
        feature_names,
        n_topics: int,
        documents: List[str]
    ) -> List[TopicModeling]:
        """Extract topics using NMF"""
        try:
            nmf = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200
            )
            
            nmf.fit(doc_term_matrix)
            
            topics = []
            for topic_idx, topic in enumerate(nmf.components_):
                # Get top keywords
                top_keywords_idx = topic.argsort()[-10:][::-1]
                top_keywords = [feature_names[i] for i in top_keywords_idx]
                
                # Calculate topic weights for each document
                doc_topics = nmf.transform(doc_term_matrix)
                topic_weights = doc_topics[:, topic_idx]
                
                # Get documents most associated with this topic
                top_doc_indices = topic_weights.argsort()[-5:][::-1]
                top_docs = [documents[i][:100] + "..." for i in top_doc_indices]
                
                # Calculate topic coherence (simplified)
                topic_coherence = np.mean(topic_weights)
                
                topic = TopicModeling(
                    topic_id=topic_idx,
                    topic_keywords=top_keywords,
                    topic_weight=topic_coherence,
                    topic_coherence=topic_coherence,
                    documents=top_docs
                )
                
                topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error in NMF topic extraction: {e}")
            return []
    
    def analyze_sentiment_trends(
        self,
        documents: List[str],
        timestamps: List[datetime],
        window_size: int = 7
    ) -> pd.DataFrame:
        """
        Analyze sentiment trends over time
        
        Args:
            documents: List of document texts
            timestamps: List of corresponding timestamps
            window_size: Rolling window size in days
            
        Returns:
            DataFrame with sentiment trends
        """
        try:
            if len(documents) != len(timestamps):
                raise ValueError("Documents and timestamps must have same length")
            
            # Analyze sentiment for each document
            sentiments = []
            for doc, ts in zip(documents, timestamps):
                sentiment = self.analyze_sentiment(doc, method="ensemble")
                sentiments.append({
                    'timestamp': ts,
                    'compound_score': sentiment.compound_score,
                    'positive_score': sentiment.positive_score,
                    'negative_score': sentiment.negative_score,
                    'neutral_score': sentiment.neutral_score,
                    'sentiment_label': sentiment.sentiment_label.value
                })
            
            # Create DataFrame
            df = pd.DataFrame(sentiments)
            df = df.sort_values('timestamp')
            
            # Calculate rolling averages
            df['compound_rolling'] = df['compound_score'].rolling(window=window_size).mean()
            df['positive_rolling'] = df['positive_score'].rolling(window=window_size).mean()
            df['negative_rolling'] = df['negative_score'].rolling(window=window_size).mean()
            
            # Calculate sentiment momentum
            df['sentiment_momentum'] = df['compound_score'].diff()
            df['sentiment_momentum_rolling'] = df['sentiment_momentum'].rolling(window=window_size).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment trends: {e}")
            return pd.DataFrame()
    
    def _create_default_sentiment(self) -> SentimentScore:
        """Create default sentiment score"""
        return SentimentScore(
            compound_score=0.0,
            positive_score=0.0,
            negative_score=0.0,
            neutral_score=1.0,
            sentiment_label=SentimentType.NEUTRAL,
            confidence=0.0,
            timestamp=datetime.now()
        )
