"""
Intelligent Advisory System (RAG Implementation)

This module implements a Retrieval-Augmented Generation system that provides
intelligent, context-aware investment advice by combining large language models
with domain-specific knowledge bases.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import json
import hashlib
from pathlib import Path

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
import faiss
from sentence_transformers import SentenceTransformer

# LLM integration
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.retrievers import VectorStoreRetriever
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Custom imports
from .nlp_engine import NLPEngine, DocumentType
from .risk_scoring import RiskScoringEngine, RiskScore

logger = logging.getLogger(__name__)

class AdvisoryType(Enum):
    """Types of investment advice"""
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    MARKET_ANALYSIS = "market_analysis"
    INVESTMENT_RECOMMENDATION = "investment_recommendation"
    EDUCATIONAL = "educational"
    ALERT = "alert"

class UserProfile(Enum):
    """User profile types"""
    RETAIL_INVESTOR = "retail_investor"
    INSTITUTIONAL_INVESTOR = "institutional_investor"
    PROFESSIONAL_TRADER = "professional_trader"
    FINANCIAL_ADVISOR = "financial_advisor"

class RiskTolerance(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class InvestmentQuery:
    """Investment query structure"""
    query_text: str
    user_profile: UserProfile
    risk_tolerance: RiskTolerance
    investment_horizon: str
    investment_amount: float
    current_portfolio: Dict = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class InvestmentAdvice:
    """Investment advice structure"""
    advice_type: AdvisoryType
    title: str
    summary: str
    detailed_explanation: str
    recommendations: List[str]
    risk_assessment: Dict
    supporting_data: Dict
    confidence_score: float
    timestamp: datetime
    sources: List[str]

@dataclass
class KnowledgeBaseEntry:
    """Knowledge base entry structure"""
    content: str
    content_type: str
    metadata: Dict
    embedding: List[float]
    source: str
    timestamp: datetime
    tags: List[str]

class IntelligentAdvisorySystem:
    """
    Intelligent advisory system using RAG for bond investment advice
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize components
        self.nlp_engine = NLPEngine()
        self.risk_engine = RiskScoringEngine()
        
        # Initialize vector database
        self._initialize_vector_database()
        
        # Initialize LLM components
        self._initialize_llm_components()
        
        # Initialize conversation memory
        self._initialize_conversation_memory()
        
        # Load knowledge base
        self._load_knowledge_base()
        
        # Initialize advisory templates
        self._initialize_advisory_templates()
        
    def _initialize_vector_database(self):
        """Initialize vector database for knowledge storage"""
        try:
            # Initialize ChromaDB
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            # Create collections
            self.bond_knowledge_collection = self.chroma_client.create_collection(
                name="bond_knowledge",
                metadata={"description": "Bond investment knowledge base"}
            )
            
            self.market_data_collection = self.chroma_client.create_collection(
                name="market_data",
                metadata={"description": "Market data and analytics"}
            )
            
            self.regulatory_collection = self.chroma_client.create_collection(
                name="regulatory_info",
                metadata={"description": "Regulatory information and compliance"}
            )
            
            # Initialize sentence transformer for embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            self.chroma_client = None
            self.embedding_model = None
    
    def _initialize_llm_components(self):
        """Initialize LLM components"""
        try:
            # OpenAI configuration
            openai.api_key = self.config.get('openai_api_key')
            
            # LangChain components
            self.llm = ChatOpenAI(
                model_name=self.config.get('llm_model', 'gpt-3.5-turbo'),
                temperature=0.1,
                max_tokens=1000
            )
            
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.config.get('openai_api_key')
            )
            
            # Create vector stores
            self.bond_knowledge_store = Chroma(
                collection_name="bond_knowledge",
                embedding_function=self.embeddings
            )
            
            self.market_data_store = Chroma(
                collection_name="market_data",
                embedding_function=self.embeddings
            )
            
            logger.info("LLM components initialized successfully")
            
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
            self.llm = None
            self.embeddings = None
    
    def _initialize_conversation_memory(self):
        """Initialize conversation memory systems"""
        try:
            # Conversation buffer memory
            self.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Conversation summary memory for long conversations
            self.summary_memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True
            )
            
            # User session memory
            self.user_sessions = {}
            
        except Exception as e:
            logger.error(f"Error initializing conversation memory: {e}")
    
    def _load_knowledge_base(self):
        """Load initial knowledge base"""
        try:
            # Load bond investment knowledge
            self._load_bond_knowledge()
            
            # Load market data knowledge
            self._load_market_knowledge()
            
            # Load regulatory knowledge
            self._load_regulatory_knowledge()
            
            logger.info("Knowledge base loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
    
    def _load_bond_knowledge(self):
        """Load bond investment knowledge"""
        try:
            bond_knowledge = [
                {
                    "content": "Government bonds are debt securities issued by governments to finance public spending. They are generally considered low-risk investments with predictable returns.",
                    "metadata": {"type": "educational", "topic": "government_bonds", "risk_level": "low"},
                    "tags": ["government_bonds", "debt_securities", "low_risk"]
                },
                {
                    "content": "Corporate bonds are debt securities issued by corporations to raise capital. They offer higher yields than government bonds but carry higher credit risk.",
                    "metadata": {"type": "educational", "topic": "corporate_bonds", "risk_level": "medium"},
                    "tags": ["corporate_bonds", "debt_securities", "credit_risk"]
                },
                {
                    "content": "Duration measures a bond's price sensitivity to interest rate changes. Longer duration bonds are more sensitive to rate changes.",
                    "metadata": {"type": "educational", "topic": "duration", "risk_level": "medium"},
                    "tags": ["duration", "interest_rate_risk", "bond_pricing"]
                },
                {
                    "content": "Credit ratings assess the creditworthiness of bond issuers. AAA is the highest rating, while D indicates default.",
                    "metadata": {"type": "educational", "topic": "credit_ratings", "risk_level": "low"},
                    "tags": ["credit_ratings", "creditworthiness", "default_risk"]
                }
            ]
            
            for knowledge in bond_knowledge:
                self._add_to_knowledge_base(
                    knowledge["content"],
                    "bond_knowledge",
                    knowledge["metadata"],
                    knowledge["tags"]
                )
                
        except Exception as e:
            logger.error(f"Error loading bond knowledge: {e}")
    
    def _load_market_knowledge(self):
        """Load market data knowledge"""
        try:
            market_knowledge = [
                {
                    "content": "The Indian bond market is influenced by RBI monetary policy, inflation expectations, and global economic conditions.",
                    "metadata": {"type": "market_analysis", "topic": "indian_bond_market", "region": "india"},
                    "tags": ["indian_bond_market", "rbi", "monetary_policy", "inflation"]
                },
                {
                    "content": "Yield curve analysis helps understand market expectations for interest rates and economic growth.",
                    "metadata": {"type": "market_analysis", "topic": "yield_curve", "complexity": "intermediate"},
                    "tags": ["yield_curve", "interest_rates", "economic_growth", "market_expectations"]
                }
            ]
            
            for knowledge in market_knowledge:
                self._add_to_knowledge_base(
                    knowledge["content"],
                    "market_data",
                    knowledge["metadata"],
                    knowledge["tags"]
                )
                
        except Exception as e:
            logger.error(f"Error loading market knowledge: {e}")
    
    def _load_regulatory_knowledge(self):
        """Load regulatory knowledge"""
        try:
            regulatory_knowledge = [
                {
                    "content": "SEBI regulates the Indian securities market, including bond trading and investor protection.",
                    "metadata": {"type": "regulatory", "topic": "sebi_regulation", "jurisdiction": "india"},
                    "tags": ["sebi", "regulation", "securities_market", "investor_protection"]
                },
                {
                    "content": "RBI regulates the money market and government securities market in India.",
                    "metadata": {"type": "regulatory", "topic": "rbi_regulation", "jurisdiction": "india"},
                    "tags": ["rbi", "regulation", "money_market", "government_securities"]
                }
            ]
            
            for knowledge in regulatory_knowledge:
                self._add_to_knowledge_base(
                    knowledge["content"],
                    "regulatory_info",
                    knowledge["metadata"],
                    knowledge["tags"]
                )
                
        except Exception as e:
            logger.error(f"Error loading regulatory knowledge: {e}")
    
    def _add_to_knowledge_base(
        self,
        content: str,
        collection_name: str,
        metadata: Dict,
        tags: List[str]
    ):
        """Add content to knowledge base"""
        try:
            if not self.chroma_client:
                return
            
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Add to collection
            collection = getattr(self, f"{collection_name}_collection")
            collection.add(
                documents=[content],
                metadatas=[metadata],
                embeddings=[embedding],
                ids=[f"{collection_name}_{len(collection.get()['ids'])}"]
            )
            
        except Exception as e:
            logger.error(f"Error adding to knowledge base: {e}")
    
    def _initialize_advisory_templates(self):
        """Initialize advisory response templates"""
        self.advisory_templates = {
            AdvisoryType.RISK_ASSESSMENT: {
                "title": "Risk Assessment for {bond_name}",
                "summary": "Comprehensive risk analysis based on multiple factors",
                "template": """
                Based on our analysis of {bond_name}, here's the risk assessment:
                
                **Overall Risk Score**: {risk_score}/100
                
                **Risk Breakdown**:
                - Credit Risk: {credit_risk}/100
                - Interest Rate Risk: {interest_rate_risk}/100
                - Liquidity Risk: {liquidity_risk}/100
                
                **Key Risk Factors**:
                {risk_factors}
                
                **Recommendations**:
                {recommendations}
                """
            },
            AdvisoryType.PORTFOLIO_OPTIMIZATION: {
                "title": "Portfolio Optimization Recommendations",
                "summary": "Strategic portfolio allocation and optimization advice",
                "template": """
                **Portfolio Optimization Analysis**
                
                **Current Portfolio Risk**: {current_risk}/100
                **Target Risk Level**: {target_risk}/100
                
                **Recommended Changes**:
                {recommendations}
                
                **Expected Impact**:
                {expected_impact}
                """
            },
            AdvisoryType.MARKET_ANALYSIS: {
                "title": "Market Analysis and Outlook",
                "summary": "Current market conditions and future outlook",
                "template": """
                **Market Analysis Report**
                
                **Current Market Conditions**:
                {market_conditions}
                
                **Key Drivers**:
                {key_drivers}
                
                **Outlook**:
                {outlook}
                
                **Investment Implications**:
                {implications}
                """
            }
        }
    
    def process_investment_query(
        self,
        query: InvestmentQuery,
        user_id: str = None
    ) -> InvestmentAdvice:
        """
        Process investment query and generate advice
        
        Args:
            query: Investment query from user
            user_id: User identifier for session management
            
        Returns:
            Investment advice response
        """
        try:
            # Store user session
            if user_id:
                self._update_user_session(user_id, query)
            
            # Analyze query intent
            intent = self._analyze_query_intent(query.query_text)
            
            # Retrieve relevant knowledge
            relevant_knowledge = self._retrieve_relevant_knowledge(query, intent)
            
            # Generate advice based on intent
            if intent == "risk_assessment":
                advice = self._generate_risk_assessment(query, relevant_knowledge)
            elif intent == "portfolio_optimization":
                advice = self._generate_portfolio_advice(query, relevant_knowledge)
            elif intent == "market_analysis":
                advice = self._generate_market_analysis(query, relevant_knowledge)
            elif intent == "educational":
                advice = self._generate_educational_content(query, relevant_knowledge)
            else:
                advice = self._generate_general_advice(query, relevant_knowledge)
            
            # Update conversation memory
            self._update_conversation_memory(query, advice)
            
            return advice
            
        except Exception as e:
            logger.error(f"Error processing investment query: {e}")
            return self._create_error_advice(str(e))
    
    def _analyze_query_intent(self, query_text: str) -> str:
        """Analyze the intent of the user query"""
        try:
            query_lower = query_text.lower()
            
            # Risk-related keywords
            risk_keywords = ['risk', 'safe', 'dangerous', 'volatile', 'stable', 'secure']
            if any(keyword in query_lower for keyword in risk_keywords):
                return "risk_assessment"
            
            # Portfolio-related keywords
            portfolio_keywords = ['portfolio', 'allocation', 'diversify', 'balance', 'mix']
            if any(keyword in query_lower for keyword in portfolio_keywords):
                return "portfolio_optimization"
            
            # Market-related keywords
            market_keywords = ['market', 'trend', 'outlook', 'forecast', 'condition']
            if any(keyword in query_lower for keyword in market_keywords):
                return "market_analysis"
            
            # Educational keywords
            educational_keywords = ['what is', 'how to', 'explain', 'learn', 'understand']
            if any(keyword in query_lower for keyword in educational_keywords):
                return "educational"
            
            return "general"
            
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return "general"
    
    def _retrieve_relevant_knowledge(
        self,
        query: InvestmentQuery,
        intent: str
    ) -> List[Dict]:
        """Retrieve relevant knowledge from vector database"""
        try:
            relevant_knowledge = []
            
            # Search in relevant collections
            if intent == "risk_assessment":
                collections = ["bond_knowledge", "market_data"]
            elif intent == "portfolio_optimization":
                collections = ["bond_knowledge", "market_data"]
            elif intent == "market_analysis":
                collections = ["market_data", "regulatory_info"]
            else:
                collections = ["bond_knowledge", "market_data", "regulatory_info"]
            
            for collection_name in collections:
                try:
                    collection = getattr(self, f"{collection_name}_collection")
                    
                    # Query collection
                    results = collection.query(
                        query_texts=[query.query_text],
                        n_results=5
                    )
                    
                    for i, doc in enumerate(results['documents'][0]):
                        relevant_knowledge.append({
                            'content': doc,
                            'metadata': results['metadatas'][0][i],
                            'collection': collection_name,
                            'relevance_score': 1.0 / (i + 1)  # Simple relevance scoring
                        })
                        
                except Exception as e:
                    logger.warning(f"Error querying collection {collection_name}: {e}")
                    continue
            
            # Sort by relevance
            relevant_knowledge.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return relevant_knowledge[:10]  # Return top 10 results
            
        except Exception as e:
            logger.error(f"Error retrieving relevant knowledge: {e}")
            return []
    
    def _generate_risk_assessment(
        self,
        query: InvestmentQuery,
        knowledge: List[Dict]
    ) -> InvestmentAdvice:
        """Generate risk assessment advice"""
        try:
            # Extract bond information from query
            bond_info = self._extract_bond_info(query.query_text)
            
            # Calculate risk scores (simplified)
            risk_scores = self._calculate_risk_scores(bond_info)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(risk_scores, query.risk_tolerance)
            
            # Create advice
            advice = InvestmentAdvice(
                advice_type=AdvisoryType.RISK_ASSESSMENT,
                title=f"Risk Assessment for {bond_info.get('name', 'Bond')}",
                summary=f"Overall risk score: {risk_scores['overall']}/100",
                detailed_explanation=self._format_risk_explanation(risk_scores, knowledge),
                recommendations=recommendations,
                risk_assessment=risk_scores,
                supporting_data={'knowledge_sources': len(knowledge)},
                confidence_score=0.85,
                timestamp=datetime.now(),
                sources=[k['collection'] for k in knowledge[:3]]
            )
            
            return advice
            
        except Exception as e:
            logger.error(f"Error generating risk assessment: {e}")
            return self._create_error_advice("Error generating risk assessment")
    
    def _extract_bond_info(self, query_text: str) -> Dict:
        """Extract bond information from query text"""
        # Simple extraction - in practice, use more sophisticated NLP
        bond_info = {
            'name': 'Unknown Bond',
            'type': 'corporate',
            'rating': 'BBB',
            'maturity': '5 years'
        }
        
        # Extract rating if mentioned
        import re
        rating_match = re.search(r'\b(?:AAA|AA\+|AA|AA-|A\+|A|A-|BBB\+|BBB|BBB-)\b', query_text, re.IGNORECASE)
        if rating_match:
            bond_info['rating'] = rating_match.group()
        
        # Extract maturity if mentioned
        maturity_match = re.search(r'(\d+)\s*(?:year|yr)', query_text, re.IGNORECASE)
        if maturity_match:
            bond_info['maturity'] = f"{maturity_match.group(1)} years"
        
        return bond_info
    
    def _calculate_risk_scores(self, bond_info: Dict) -> Dict:
        """Calculate risk scores for bond"""
        try:
            # Simplified risk calculation
            rating_scores = {
                'AAA': 10, 'AA+': 15, 'AA': 20, 'AA-': 25,
                'A+': 30, 'A': 35, 'A-': 40,
                'BBB+': 45, 'BBB': 50, 'BBB-': 55,
                'BB+': 60, 'BB': 65, 'BB-': 70,
                'B+': 75, 'B': 80, 'B-': 85,
                'C': 90, 'D': 100
            }
            
            credit_risk = rating_scores.get(bond_info.get('rating', 'BBB'), 50)
            
            # Maturity risk (longer maturity = higher risk)
            maturity_years = int(bond_info.get('maturity', '5').split()[0])
            maturity_risk = min(30, maturity_years * 3)
            
            # Overall risk (weighted average)
            overall_risk = (credit_risk * 0.7) + (maturity_risk * 0.3)
            
            return {
                'overall': round(overall_risk),
                'credit_risk': credit_risk,
                'maturity_risk': maturity_risk,
                'liquidity_risk': 25,  # Default moderate liquidity risk
                'interest_rate_risk': 30  # Default moderate interest rate risk
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk scores: {e}")
            return {
                'overall': 50,
                'credit_risk': 50,
                'maturity_risk': 50,
                'liquidity_risk': 50,
                'interest_rate_risk': 50
            }
    
    def _generate_risk_recommendations(
        self,
        risk_scores: Dict,
        risk_tolerance: RiskTolerance
    ) -> List[str]:
        """Generate risk-based recommendations"""
        recommendations = []
        
        overall_risk = risk_scores['overall']
        
        if risk_tolerance == RiskTolerance.CONSERVATIVE:
            if overall_risk > 40:
                recommendations.append("Consider lower-risk alternatives like government bonds")
                recommendations.append("Limit exposure to this bond in your portfolio")
            elif overall_risk > 25:
                recommendations.append("Monitor this bond closely for any rating changes")
                recommendations.append("Consider diversifying with other low-risk bonds")
            else:
                recommendations.append("This bond fits well with your conservative risk profile")
        
        elif risk_tolerance == RiskTolerance.MODERATE:
            if overall_risk > 60:
                recommendations.append("This bond may be too risky for moderate risk tolerance")
                recommendations.append("Consider reducing position size")
            elif overall_risk > 40:
                recommendations.append("Monitor credit quality and market conditions")
                recommendations.append("Ensure adequate portfolio diversification")
            else:
                recommendations.append("This bond aligns with your moderate risk profile")
        
        elif risk_tolerance == RiskTolerance.AGGRESSIVE:
            if overall_risk > 80:
                recommendations.append("High risk bond - ensure you understand the risks")
                recommendations.append("Monitor closely for any negative developments")
            else:
                recommendations.append("This bond fits your aggressive risk profile")
        
        # Add general recommendations
        if risk_scores['credit_risk'] > 60:
            recommendations.append("High credit risk - monitor issuer financial health")
        
        if risk_scores['maturity_risk'] > 50:
            recommendations.append("Long maturity - consider interest rate sensitivity")
        
        return recommendations
    
    def _format_risk_explanation(self, risk_scores: Dict, knowledge: List[Dict]) -> str:
        """Format risk explanation using knowledge base"""
        explanation = f"""
        **Risk Analysis Summary**
        
        The bond has an overall risk score of {risk_scores['overall']}/100, which indicates {'low' if risk_scores['overall'] < 30 else 'moderate' if risk_scores['overall'] < 60 else 'high'} risk.
        
        **Risk Breakdown:**
        - **Credit Risk**: {risk_scores['credit_risk']}/100 - {'Low' if risk_scores['credit_risk'] < 30 else 'Moderate' if risk_scores['credit_risk'] < 60 else 'High'} credit risk indicates the likelihood of default by the issuer.
        - **Maturity Risk**: {risk_scores['maturity_risk']}/100 - {'Low' if risk_scores['maturity_risk'] < 30 else 'Moderate' if risk_scores['maturity_risk'] < 60 else 'High'} maturity risk reflects sensitivity to interest rate changes.
        - **Liquidity Risk**: {risk_scores['liquidity_risk']}/100 - {'Low' if risk_scores['liquidity_risk'] < 30 else 'Moderate' if risk_scores['liquidity_risk'] < 60 else 'High'} liquidity risk indicates how easily the bond can be sold.
        - **Interest Rate Risk**: {risk_scores['interest_rate_risk']}/100 - {'Low' if risk_scores['interest_rate_risk'] < 30 else 'Moderate' if risk_scores['interest_rate_risk'] < 60 else 'High'} interest rate risk shows sensitivity to changes in market rates.
        
        **Key Insights:**
        """
        
        # Add insights from knowledge base
        for i, k in enumerate(knowledge[:3]):
            explanation += f"\n- {k['content'][:100]}..."
        
        return explanation
    
    def _generate_portfolio_advice(
        self,
        query: InvestmentQuery,
        knowledge: List[Dict]
    ) -> InvestmentAdvice:
        """Generate portfolio optimization advice"""
        try:
            # Analyze current portfolio
            portfolio_analysis = self._analyze_portfolio(query.current_portfolio)
            
            # Generate optimization recommendations
            recommendations = self._generate_portfolio_recommendations(
                portfolio_analysis, query.risk_tolerance, query.investment_horizon
            )
            
            advice = InvestmentAdvice(
                advice_type=AdvisoryType.PORTFOLIO_OPTIMIZATION,
                title="Portfolio Optimization Recommendations",
                summary=f"Portfolio risk score: {portfolio_analysis.get('risk_score', 50)}/100",
                detailed_explanation=self._format_portfolio_explanation(portfolio_analysis, knowledge),
                recommendations=recommendations,
                risk_assessment=portfolio_analysis,
                supporting_data={'portfolio_size': len(query.current_portfolio)},
                confidence_score=0.80,
                timestamp=datetime.now(),
                sources=[k['collection'] for k in knowledge[:3]]
            )
            
            return advice
            
        except Exception as e:
            logger.error(f"Error generating portfolio advice: {e}")
            return self._create_error_advice("Error generating portfolio advice")
    
    def _analyze_portfolio(self, portfolio: Dict) -> Dict:
        """Analyze current portfolio composition"""
        try:
            if not portfolio:
                return {'risk_score': 50, 'diversification': 'low', 'sector_exposure': {}}
            
            # Calculate portfolio metrics
            total_value = sum(portfolio.values())
            risk_score = 50  # Default moderate risk
            
            # Analyze sector exposure
            sector_exposure = {}
            for bond, value in portfolio.items():
                # Simplified sector classification
                if 'gov' in bond.lower():
                    sector = 'government'
                elif 'corp' in bond.lower():
                    sector = 'corporate'
                else:
                    sector = 'other'
                
                sector_exposure[sector] = sector_exposure.get(sector, 0) + value
            
            # Normalize sector exposure
            sector_exposure = {k: v/total_value for k, v in sector_exposure.items()}
            
            return {
                'risk_score': risk_score,
                'diversification': 'high' if len(sector_exposure) > 2 else 'low',
                'sector_exposure': sector_exposure,
                'total_value': total_value
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return {'risk_score': 50, 'diversification': 'low', 'sector_exposure': {}}
    
    def _generate_portfolio_recommendations(
        self,
        portfolio_analysis: Dict,
        risk_tolerance: RiskTolerance,
        investment_horizon: str
    ) -> List[str]:
        """Generate portfolio optimization recommendations"""
        recommendations = []
        
        # Diversification recommendations
        if portfolio_analysis['diversification'] == 'low':
            recommendations.append("Increase portfolio diversification across different sectors and bond types")
            recommendations.append("Consider adding government bonds for stability")
        
        # Risk tolerance recommendations
        if risk_tolerance == RiskTolerance.CONSERVATIVE:
            if portfolio_analysis['risk_score'] > 40:
                recommendations.append("Reduce exposure to high-risk bonds")
                recommendations.append("Increase allocation to government and high-grade corporate bonds")
        
        # Investment horizon recommendations
        if 'long' in investment_horizon.lower():
            recommendations.append("Consider laddering bond maturities for consistent income")
            recommendations.append("Monitor interest rate sensitivity for longer-term bonds")
        
        return recommendations
    
    def _format_portfolio_explanation(
        self,
        portfolio_analysis: Dict,
        knowledge: List[Dict]
    ) -> str:
        """Format portfolio explanation"""
        explanation = f"""
        **Portfolio Analysis Summary**
        
        Your current portfolio has a risk score of {portfolio_analysis['risk_score']}/100 and {'high' if portfolio_analysis['diversification'] == 'high' else 'low'} diversification.
        
        **Current Allocation:**
        """
        
        for sector, exposure in portfolio_analysis['sector_exposure'].items():
            explanation += f"\n- {sector.title()}: {exposure:.1%}"
        
        explanation += "\n\n**Key Insights:**"
        
        # Add insights from knowledge base
        for i, k in enumerate(knowledge[:2]):
            explanation += f"\n- {k['content'][:100]}..."
        
        return explanation
    
    def _generate_market_analysis(
        self,
        query: InvestmentQuery,
        knowledge: List[Dict]
    ) -> InvestmentAdvice:
        """Generate market analysis advice"""
        try:
            # Extract market insights from knowledge
            market_insights = self._extract_market_insights(knowledge)
            
            advice = InvestmentAdvice(
                advice_type=AdvisoryType.MARKET_ANALYSIS,
                title="Market Analysis and Outlook",
                summary="Current market conditions and investment implications",
                detailed_explanation=self._format_market_analysis(market_insights, knowledge),
                recommendations=self._generate_market_recommendations(market_insights),
                risk_assessment={'market_volatility': 'medium'},
                supporting_data={'analysis_date': datetime.now().isoformat()},
                confidence_score=0.75,
                timestamp=datetime.now(),
                sources=[k['collection'] for k in knowledge[:3]]
            )
            
            return advice
            
        except Exception as e:
            logger.error(f"Error generating market analysis: {e}")
            return self._create_error_advice("Error generating market analysis")
    
    def _extract_market_insights(self, knowledge: List[Dict]) -> Dict:
        """Extract market insights from knowledge base"""
        insights = {
            'market_conditions': 'Stable',
            'key_drivers': ['RBI policy', 'Inflation expectations'],
            'outlook': 'Moderate growth expected',
            'implications': 'Favorable for bond investments'
        }
        
        # Extract insights from knowledge content
        for k in knowledge:
            content = k['content'].lower()
            if 'volatile' in content or 'uncertainty' in content:
                insights['market_conditions'] = 'Volatile'
            elif 'growth' in content or 'positive' in content:
                insights['outlook'] = 'Positive growth outlook'
        
        return insights
    
    def _generate_market_recommendations(self, market_insights: Dict) -> List[str]:
        """Generate market-based recommendations"""
        recommendations = []
        
        if market_insights['market_conditions'] == 'Volatile':
            recommendations.append("Consider defensive positioning with high-quality bonds")
            recommendations.append("Monitor market developments closely")
        else:
            recommendations.append("Current market conditions support bond investments")
            recommendations.append("Consider gradual position building")
        
        return recommendations
    
    def _format_market_analysis(
        self,
        market_insights: Dict,
        knowledge: List[Dict]
    ) -> str:
        """Format market analysis"""
        explanation = f"""
        **Market Analysis Report**
        
        **Current Market Conditions**: {market_insights['market_conditions']}
        
        **Key Drivers**:
        """
        
        for driver in market_insights['key_drivers']:
            explanation += f"\n- {driver}"
        
        explanation += f"""
        
        **Outlook**: {market_insights['outlook']}
        
        **Investment Implications**: {market_insights['implications']}
        
        **Supporting Analysis**:
        """
        
        # Add insights from knowledge base
        for i, k in enumerate(knowledge[:2]):
            explanation += f"\n- {k['content'][:100]}..."
        
        return explanation
    
    def _generate_educational_content(
        self,
        query: InvestmentQuery,
        knowledge: List[Dict]
    ) -> InvestmentAdvice:
        """Generate educational content"""
        try:
            # Extract educational content from knowledge
            educational_content = self._extract_educational_content(query, knowledge)
            
            advice = InvestmentAdvice(
                advice_type=AdvisoryType.EDUCATIONAL,
                title="Educational Content: Understanding Bond Investments",
                summary="Learn about bond investment fundamentals",
                detailed_explanation=educational_content,
                recommendations=["Continue learning about bond markets", "Consult with financial advisors"],
                risk_assessment={'complexity': 'low'},
                supporting_data={'content_type': 'educational'},
                confidence_score=0.90,
                timestamp=datetime.now(),
                sources=[k['collection'] for k in knowledge[:3]]
            )
            
            return advice
            
        except Exception as e:
            logger.error(f"Error generating educational content: {e}")
            return self._create_error_advice("Error generating educational content")
    
    def _extract_educational_content(
        self,
        query: InvestmentQuery,
        knowledge: List[Dict]
    ) -> str:
        """Extract educational content from knowledge base"""
        content = """
        **Understanding Bond Investments**
        
        Bonds are debt securities that represent a loan from an investor to a borrower (typically a corporation or government).
        
        **Key Concepts:**
        """
        
        # Add content from knowledge base
        for i, k in enumerate(knowledge[:3]):
            content += f"\n- {k['content']}"
        
        content += """
        
        **Why Invest in Bonds?**
        - Regular income through interest payments
        - Capital preservation (for high-quality bonds)
        - Portfolio diversification
        - Predictable returns
        
        **Risk Considerations:**
        - Credit risk (issuer default)
        - Interest rate risk (price sensitivity)
        - Liquidity risk (ease of selling)
        - Inflation risk (purchasing power)
        """
        
        return content
    
    def _generate_general_advice(
        self,
        query: InvestmentQuery,
        knowledge: List[Dict]
    ) -> InvestmentAdvice:
        """Generate general investment advice"""
        try:
            advice = InvestmentAdvice(
                advice_type=AdvisoryType.INVESTMENT_RECOMMENDATION,
                title="General Investment Guidance",
                summary="General advice based on your profile and market conditions",
                detailed_explanation=self._format_general_advice(query, knowledge),
                recommendations=["Consider your risk tolerance", "Diversify your portfolio", "Monitor market conditions"],
                risk_assessment={'general_risk': 'moderate'},
                supporting_data={'query_type': 'general'},
                confidence_score=0.70,
                timestamp=datetime.now(),
                sources=[k['collection'] for k in knowledge[:3]]
            )
            
            return advice
            
        except Exception as e:
            logger.error(f"Error generating general advice: {e}")
            return self._create_error_advice("Error generating general advice")
    
    def _format_general_advice(
        self,
        query: InvestmentQuery,
        knowledge: List[Dict]
    ) -> str:
        """Format general advice"""
        explanation = f"""
        **Investment Guidance for {query.user_profile.value.replace('_', ' ').title()}**
        
        Based on your {query.risk_tolerance.value} risk tolerance and {query.investment_horizon} investment horizon, here are some general considerations:
        
        **Risk Profile**: {query.risk_tolerance.value.title()} risk tolerance suggests {'conservative' if query.risk_tolerance == RiskTolerance.CONSERVATIVE else 'moderate' if query.risk_tolerance == RiskTolerance.MODERATE else 'aggressive'} investment approach.
        
        **Investment Horizon**: {query.investment_horizon} horizon allows for {'short-term' if 'short' in query.investment_horizon.lower() else 'medium-term' if 'medium' in query.investment_horizon.lower() else 'long-term'} planning.
        
        **Key Insights**:
        """
        
        # Add insights from knowledge base
        for i, k in enumerate(knowledge[:2]):
            explanation += f"\n- {k['content'][:100]}..."
        
        return explanation
    
    def _update_user_session(self, user_id: str, query: InvestmentQuery):
        """Update user session with latest query"""
        try:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {
                    'queries': [],
                    'preferences': {},
                    'last_activity': datetime.now()
                }
            
            self.user_sessions[user_id]['queries'].append({
                'query': query.query_text,
                'timestamp': query.timestamp,
                'advice_type': 'pending'
            })
            
            self.user_sessions[user_id]['last_activity'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating user session: {e}")
    
    def _update_conversation_memory(self, query: InvestmentQuery, advice: InvestmentAdvice):
        """Update conversation memory with query and advice"""
        try:
            # Add to conversation memory
            self.conversation_memory.save_context(
                {"input": query.query_text},
                {"output": advice.summary}
            )
            
        except Exception as e:
            logger.error(f"Error updating conversation memory: {e}")
    
    def _create_error_advice(self, error_message: str) -> InvestmentAdvice:
        """Create error advice when processing fails"""
        return InvestmentAdvice(
            advice_type=AdvisoryType.EDUCATIONAL,
            title="Error Processing Request",
            summary="Unable to process your request at this time",
            detailed_explanation=f"An error occurred while processing your request: {error_message}. Please try again or contact support.",
            recommendations=["Try rephrasing your question", "Check your input format", "Contact support if the issue persists"],
            risk_assessment={},
            supporting_data={'error': error_message},
            confidence_score=0.0,
            timestamp=datetime.now(),
            sources=[]
        )
    
    def get_conversation_history(self, user_id: str = None) -> List[Dict]:
        """Get conversation history for user"""
        try:
            if user_id and user_id in self.user_sessions:
                return self.user_sessions[user_id]['queries']
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def add_custom_knowledge(
        self,
        content: str,
        metadata: Dict,
        tags: List[str],
        collection_name: str = "bond_knowledge"
    ):
        """Add custom knowledge to the system"""
        try:
            self._add_to_knowledge_base(content, collection_name, metadata, tags)
            logger.info(f"Added custom knowledge to {collection_name}")
        except Exception as e:
            logger.error(f"Error adding custom knowledge: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        try:
            status = {
                'vector_database': 'active' if self.chroma_client else 'inactive',
                'llm_components': 'active' if self.llm else 'inactive',
                'knowledge_base_size': 0,
                'active_users': len(self.user_sessions),
                'conversation_memory': 'active' if self.conversation_memory else 'inactive',
                'last_updated': datetime.now().isoformat()
            }
            
            # Get knowledge base size
            if self.chroma_client:
                try:
                    bond_collection = getattr(self, 'bond_knowledge_collection', None)
                    if bond_collection:
                        status['knowledge_base_size'] = len(bond_collection.get()['ids'])
                except Exception as e:
                    logger.warning(f"Error getting knowledge base size: {e}")
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
