import pandas as pd
from transformers import pipeline
import torch
import re
from tqdm import tqdm
import warnings
import numpy as np
from typing import Tuple, Dict, List
import logging

warnings.filterwarnings('ignore')

class EnhancedReviewClassifier:
    def __init__(self, confidence_threshold: float = 0.8):
        self.device = 0 if torch.cuda.is_available() else -1
        self.confidence_threshold = confidence_threshold
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🚀 Loading enhanced classification model...")
        self.logger.info(f"Device: {'GPU' if self.device >= 0 else 'CPU'}")
        
        # Initialize classifier with better error handling
        self._initialize_classifier()
        
        # Enhanced category definitions
        self.categories = [
            "genuine visitor review with personal experience about the castle or museum",
            "commercial advertisement or promotional spam content",
            "vulgar profanity sexual or violent inappropriate content",
            "personal contact information email or phone number",
            "opinion from someone who never actually visited the location"
        ]
        
        # Label mapping
        self.label_map = {
            "genuine visitor review with personal experience about the castle or museum": "authentic",
            "commercial advertisement or promotional spam content": "advertisement",
            "vulgar profanity sexual or violent inappropriate content": "inappropriate",
            "personal contact information email or phone number": "personal-info", 
            "opinion from someone who never actually visited the location": "never-visited"
        }
        
        # Enhanced keyword sets
        self.authentic_indicators = {
            'visit_words': ['visited', 'went', 'tour', 'toured', 'explored', 'walked through'],
            'experience_words': ['experience', 'saw', 'enjoyed', 'loved', 'liked', 'impressed'],
            'location_specific': ['museum', 'castle', 'chapultepec', 'exhibition', 'gallery', 'rooms'],
            'descriptive': ['beautiful', 'amazing', 'wonderful', 'spectacular', 'impressive', 'historic'],
            'practical': ['admission', 'entrance', 'crowded', 'weekend', 'facilities', 'parking', 'hours']
        }
        
        # Weighted scoring for authentic reviews
        self.authentic_weights = {
            'visit_words': 2.0,
            'experience_words': 1.5, 
            'location_specific': 1.8,
            'descriptive': 1.0,
            'practical': 1.3
        }
    
    def _initialize_classifier(self):
        """Initialize the ML classifier with fallback options."""
        models_to_try = [
            "facebook/bart-large-mnli",
            "cross-encoder/nli-distilroberta-base", 
            "microsoft/DialoGPT-medium"  # Additional fallback
        ]
        
        for model_name in models_to_try:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=self.device,
                    return_all_scores=True
                )
                self.logger.info(f"✅ Successfully loaded: {model_name}")
                break
            except Exception as e:
                self.logger.warning(f"Failed to load {model_name}: {e}")
                continue
        else:
            raise RuntimeError("Could not load any classification model")
    
    def calculate_authentic_score(self, text: str) -> float:
        """Calculate weighted authentic score based on keyword presence."""
        text_lower = text.lower()
        total_score = 0.0
        
        for category, words in self.authentic_indicators.items():
            word_count = sum(1 for word in words if word in text_lower)
            category_score = word_count * self.authentic_weights[category]
            total_score += category_score
        
        # Normalize by text length to prevent bias toward longer texts
        text_length_factor = min(len(text.split()) / 10, 1.0)
        return total_score * text_length_factor
    
    def enhanced_rule_based_check(self, text: str) -> Tuple[str, float]:
        """Enhanced rule-based classification with better scoring."""
        text_lower = text.lower().strip()
        
        # Calculate authentic score
        authentic_score = self.calculate_authentic_score(text)
        if authentic_score >= 3.0:  # Threshold for strong authentic indicators
            return "authentic", min(0.85 + (authentic_score - 3.0) * 0.05, 0.95)
        
        # Enhanced advertisement detection
        ad_patterns = [
            (r'get up to \d+ months? free', 0.95),
            (r'subscribe now|sign up now', 0.90),
            (r'premium|spotify|netflix', 0.85),
            (r'free trial|limited time', 0.90),
            (r'discount|sale|offer', 0.80),
            (r'click here|download.*app', 0.75)
        ]
        
        for pattern, confidence in ad_patterns:
            if re.search(pattern, text_lower):
                return "advertisement", confidence
        
        # Personal information detection
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        phone_pattern = r'[\+]?[1-9]?[\d\s\-\(\)]{7,15}\d'
        
        if re.search(email_pattern, text) or re.search(phone_pattern, text):
            return "personal-info", 0.95
        
        # Never-visited patterns
        never_visited_patterns = [
            (r'haven[\'t]*.*visit.*but', 0.95),
            (r'never.*been.*but.*think', 0.9),
            (r'haven[\'t]*.*opportunity.*visit', 0.9),
            (r'never.*went.*but', 0.85),
            (r'didn[\'t]*.*go.*but.*looks', 0.85),
            (r'from what.*looks.*like.*think', 0.9),
            (r'planning.*visit.*heard', 0.7),
            (r'want.*go.*someday', 0.7)
        ]
        
        for pattern, confidence in never_visited_patterns:
            if re.search(pattern, text_lower):
                return "never-visited", confidence
        
        # Strict inappropriate content detection
        inappropriate_patterns = [
            (r'\b(sex|sexual|porn|nude|naked)\b.*\b(want|have|with|me)\b', 0.95),
            (r'i love sex|who wants.*sex', 0.95),
            (r'\b(kill|murder|attack)\b.*\b(everybody|everyone|people|you)\b', 0.95),
            (r'i want to kill|going to kill', 0.95),
            (r'i fucking hate|fuck.*place|fucking.*sucks', 0.85),
            (r'this.*fucking.*terrible|fucking.*worst', 0.85),
            (r'go to hell|screw you|piss off|fuck off', 0.9)
        ]
        
        for pattern, confidence in inappropriate_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "inappropriate", confidence
        
        return None, 0.0
    
    def ml_classify(self, text: str) -> Tuple[str, float]:
        """ML-based classification with better error handling."""
        try:
            result = self.classifier(text, self.categories)
            best_category = result['labels'][0]
            confidence = result['scores'][0]
            simple_label = self.label_map.get(best_category, "authentic")
            
            # Boost confidence for very short positive reviews
            if len(text.split()) <= 3 and any(word in text.lower() for word in ['beautiful', 'excellent', 'amazing', 'wonderful', 'great']):
                simple_label = "authentic"
                confidence = max(confidence, 0.8)
            
            return simple_label, confidence
        except Exception as e:
            self.logger.warning(f"ML classification error: {e}")
            return "authentic", 0.5  # Default to authentic when in doubt
    
    def classify_review(self, text: str) -> Tuple[str, float]:
        """Hybrid classification with improved logic."""
        if not text or text.strip() == "":
            return "authentic", 0.3
        
        # First try rule-based
        rule_category, rule_confidence = self.enhanced_rule_based_check(text)
        if rule_confidence > self.confidence_threshold:
            return rule_category, rule_confidence
        
        # For very short texts, be more conservative
        if len(text.split()) <= 2:
            positive_words = ['beautiful', 'excellent', 'amazing', 'wonderful', 'great', 'superb']
            if any(word in text.lower() for word in positive_words):
                return "authentic", 0.8
            return "authentic", 0.6  # Short texts are usually authentic
        
        # Fall back to ML
        ml_category, ml_confidence = self.ml_classify(text)
        
        # Override ML if it's clearly wrong
        if ml_category == "inappropriate" and ml_confidence < 0.4:
            authentic_score = self.calculate_authentic_score(text)
            if authentic_score > 1.0:
                return "authentic", 0.7
        
        # Combine rule hints with ML if both are uncertain
        if rule_confidence > 0.5 and ml_confidence < 0.7:
            return rule_category, max(rule_confidence, ml_confidence)
        
        return ml_category, ml_confidence
    
    def classify_dataframe(self, df: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """Classify all reviews with progress tracking."""
        self.logger.info(f"🔍 Classifying {len(df)} reviews...")
        
        predictions = []
        confidences = []
        
        for text in tqdm(df['text'], desc="Processing"):
            category, confidence = self.classify_review(text)
            predictions.append(category)
            confidences.append(confidence)
        
        return predictions, confidences

def analyze_results(df: pd.DataFrame) -> None:
    """Analyze classification results."""
    print(f"\n📊 CLASSIFICATION RESULTS")
    print("=" * 50)
    
    category_counts = df['predicted_category'].value_counts()
    for category, count in category_counts.items():
        pct = count / len(df) * 100
        print(f"{category:15}: {count:2d} ({pct:4.1f}%)")
    
    avg_confidence = df['confidence'].mean()
    high_conf = len(df[df['confidence'] > 0.8])
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    print(f"High confidence (>80%): {high_conf}/{len(df)} ({high_conf/len(df)*100:.1f}%)")

def main():
    # Load data
    print("📁 Loading data...")
    try:
        df = pd.read_csv("../test_reviews.csv")
        df['text'] = df['text'].fillna('').astype(str)
        print(f"✅ Loaded {len(df)} reviews")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Sample data for testing
        sample_data = {
            'text': [
                "Learn about their history, customs, and how they lived!",
                "Get up to 3 months free Spotify Premium if you subscribe now!",
                "Haven't really had the opportunity to visit but from what it looks like i think its awful.",
                "spectacular beyond words.",
                "A beautiful place, the views are unmissable",
                "One of the best museums I've been to✨✨",
                "I liked it but I expected a little more. The anthropology one is better."
            ]
        }
        df = pd.DataFrame(sample_data)
        print(f"✅ Using sample data with {len(df)} reviews")
    
    # Initialize classifier
    classifier = EnhancedReviewClassifier()
    
    # Classify reviews
    predictions, confidences = classifier.classify_dataframe(df)
    
    # Add results to dataframe
    df['predicted_category'] = predictions
    df['confidence'] = confidences
    
    # Analyze results
    analyze_results(df)
    
    # Show all classifications
    print(f"\n📋 ALL CLASSIFICATIONS:")
    print("=" * 70)
    for idx, row in df.iterrows():
        text_preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
        print(f"{idx+1:2d}. {text_preview:55} → {row['predicted_category']:12} ({row['confidence']:.2f})")
    
    # Save results
    output_file = "fixed_classified_reviews.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved results to {output_file}")

if __name__ == "__main__":
    main()