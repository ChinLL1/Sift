import pandas as pd
from transformers import pipeline
import torch
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FixedReviewClassifier:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"🚀 Loading improved classification model...")
        print(f"Device: {'GPU' if self.device >= 0 else 'CPU'}")
        
        # Use a proper zero-shot classification model
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",  # Much better for zero-shot classification
                device=self.device,
                return_all_scores=True
            )
        except Exception as e:
            print(f"Using fallback model due to: {e}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-distilroberta-base",
                device=self.device,
                return_all_scores=True
            )
        
        # Improved category definitions with clearer distinctions
        self.categories = [
            "genuine visitor review with personal experience about the castle or museum",
            "commercial advertisement or promotional spam content",
            "completely unrelated topic about animals personal life or other subjects",
            "vulgar profanity sexual or violent inappropriate content",
            "personal contact information email or phone number",
            "opinion from someone who never actually visited the location"
        ]
        
        # Mapping back to your labels
        self.label_map = {
            "genuine visitor review with personal experience about the castle or museum": "authentic",
            "commercial advertisement or promotional spam content": "advertisement",
            "completely unrelated topic about animals personal life or other subjects": "off-topic",
            "vulgar profanity sexual or violent inappropriate content": "inappropriate", 
            "personal contact information email or phone number": "personal-info",
            "opinion from someone who never actually visited the location": "never-visited"
        }
        
        # Keywords that strongly indicate authentic reviews
        self.authentic_indicators = [
            'visited', 'tour', 'museum', 'castle', 'chapultepec', 'history', 'beautiful', 
            'experience', 'recommend', 'went', 'saw', 'admission', 'crowded', 'weekend',
            'gardens', 'rooms', 'climb', 'entrance', 'facilities', 'preserved', 'maintained'
        ]
    
    def rule_based_check(self, text):
        """Enhanced rule-based classification with better authentic detection."""
        text_lower = text.lower().strip()
        
        # Early authentic detection - if it mentions castle/museum experience
        authentic_score = sum(1 for word in self.authentic_indicators if word in text_lower)
        if authentic_score >= 2:  # Strong indicators of authentic review
            return "authentic", 0.85
        
        # Clear advertisement patterns (unchanged - these work well)
        ad_patterns = [
            r'get up to \d+ months? free',
            r'subscribe now',
            r'premium',
            r'spotify',
            r'free trial',
            r'discount',
            r'click here',
            r'download.*app'
        ]
        
        for pattern in ad_patterns:
            if re.search(pattern, text_lower):
                return "advertisement", 0.95
        
        # Personal information patterns
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        phone_pattern = r'[\+]?[1-9]?[\d\s\-\(\)]{7,15}\d'
        
        if re.search(email_pattern, text) or re.search(phone_pattern, text):
            return "personal-info", 0.95
        
        # Off-topic keywords - but be more careful
        off_topic_words = ['bird', 'caterpillar', 'cat', 'dog', 'guitar', 'movie', 'cooking']
        castle_words = ['castle', 'museum', 'history', 'tour', 'visit', 'chapultepec', 'place']
        
        has_off_topic = any(word in text_lower for word in off_topic_words)
        has_castle_words = any(word in text_lower for word in castle_words)
        
        # Only classify as off-topic if it has off-topic words AND no castle context
        if has_off_topic and not has_castle_words and len(text.split()) > 5:
            return "off-topic", 0.9
        
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
        
        # MUCH stricter inappropriate patterns - only truly offensive content
        inappropriate_patterns = [
            # Sexual content
            (r'\b(sex|sexual|porn|nude|naked)\b.*\b(want|have|with|me)\b', 0.95),
            (r'i love sex|who wants.*sex', 0.95),
            
            # Violence against people
            (r'\b(kill|murder|attack)\b.*\b(everybody|everyone|people|you)\b', 0.95),
            (r'i want to kill|going to kill', 0.95),
            
            # Strong profanity in negative context
            (r'i fucking hate|fuck.*place|fucking.*sucks', 0.85),
            (r'this.*fucking.*terrible|fucking.*worst', 0.85),
            
            # Threats or harassment
            (r'go to hell|screw you|piss off|fuck off', 0.9)
        ]
        
        for pattern, confidence in inappropriate_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "inappropriate", confidence
        
        return None, 0.0
    
    def ml_classify(self, text):
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
            print(f"ML classification error: {e}")
            return "authentic", 0.5  # Default to authentic when in doubt
    
    def classify_review(self, text):
        """Hybrid classification with improved logic."""
        if not text or text.strip() == "":
            return "authentic", 0.3
        
        # First try rule-based
        rule_category, rule_confidence = self.rule_based_check(text)
        if rule_confidence > 0.8:
            return rule_category, rule_confidence
        
        # For very short texts, be more conservative
        if len(text.split()) <= 2:
            if any(word in text.lower() for word in ['beautiful', 'excellent', 'amazing', 'wonderful', 'great', 'superb']):
                return "authentic", 0.8
            return "authentic", 0.6  # Short texts are usually authentic
        
        # Fall back to ML
        ml_category, ml_confidence = self.ml_classify(text)
        
        # Override ML if it's clearly wrong (common misclassifications)
        if ml_category == "inappropriate" and ml_confidence < 0.4:
            # Check if it's actually a normal review
            if any(word in text.lower() for word in self.authentic_indicators):
                return "authentic", 0.7
        
        # If ML is uncertain, boost confidence with rule hints
        if rule_confidence > 0.5 and ml_confidence < 0.7:
            return rule_category, max(rule_confidence, ml_confidence)
        
        return ml_category, ml_confidence
    
    def classify_dataframe(self, df):
        """Classify all reviews with progress tracking."""
        print(f"🔍 Classifying {len(df)} reviews...")
        
        predictions = []
        confidences = []
        
        for text in tqdm(df['text'], desc="Processing"):
            category, confidence = self.classify_review(text)
            predictions.append(category)
            confidences.append(confidence)
        
        return predictions, confidences

def analyze_misclassifications(df):
    """Analyze where the classifier went wrong."""
    print(f"\n🔍 MISCLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    # Look at reviews classified as inappropriate with low confidence
    inappropriate_low_conf = df[(df['predicted_category'] == 'inappropriate') & (df['confidence'] < 0.3)]
    if len(inappropriate_low_conf) > 0:
        print(f"❌ {len(inappropriate_low_conf)} reviews incorrectly classified as 'inappropriate':")
        for idx, row in inappropriate_low_conf.head(5).iterrows():
            print(f"  Row {idx+1}: \"{row['text'][:60]}...\" (conf: {row['confidence']:.3f})")
    
    # Look at very short reviews
    short_reviews = df[df['text'].str.len() < 20]
    print(f"\n📏 {len(short_reviews)} very short reviews (< 20 chars)")
    
    # Look at high confidence predictions
    high_conf = df[df['confidence'] > 0.8]
    print(f"\n✅ {len(high_conf)} high confidence predictions (> 0.8)")

def main():
    # Load data
    print("📁 Loading data...")
    try:
        # Assuming the CSV file is in the current directory
        df = pd.read_csv("../test_reviews.csv")  # Adjust filename as needed
        df['text'] = df['text'].fillna('').astype(str)
        print(f"✅ Loaded {len(df)} reviews")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Create sample data for testing
        sample_data = {
            'text': [
                "Learn about their history, customs, and how they lived!",
                "Get up to 3 months free Spotify Premium if you subscribe now!",
                "Birds are kinda cool, I like them, caterpillars arent that bad either! xD",
                "Havent really had the opportunity to visit but from what it looks like i think its awful.",
                "spectacular beyond words.",
                "A beautiful place, the views are unmissable",
                "One of the best museums I've been to✨✨",
                "I liked it but I expected a little more. The anthropology one is better."
            ]
        }
        df = pd.DataFrame(sample_data)
        print(f"✅ Using sample data with {len(df)} reviews")
    
    # Initialize improved classifier
    classifier = FixedReviewClassifier()
    
    # Classify
    predictions, confidences = classifier.classify_dataframe(df)
    
    # Add results
    df['predicted_category_fixed'] = predictions
    df['confidence_fixed'] = confidences
    
    # Analysis
    print(f"\n📊 IMPROVED RESULTS SUMMARY")
    print("=" * 50)
    
    category_counts = df['predicted_category_fixed'].value_counts()
    for category, count in category_counts.items():
        pct = count / len(df) * 100
        print(f"{category:15}: {count:2d} ({pct:4.1f}%)")
    
    avg_confidence = df['confidence_fixed'].mean()
    high_conf = len(df[df['confidence_fixed'] > 0.8])
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    print(f"High confidence (>80%): {high_conf}/{len(df)} ({high_conf/len(df)*100:.1f}%)")
    
    # Compare with original if available
    if 'predicted_category' in df.columns:
        print(f"\n🔄 COMPARISON WITH ORIGINAL:")
        print("=" * 50)
        
        # Count changes
        changed = df[df['predicted_category'] != df['predicted_category_fixed']]
        print(f"Changed classifications: {len(changed)}/{len(df)} ({len(changed)/len(df)*100:.1f}%)")
        
        # Show some key improvements
        improved = changed[
            (changed['predicted_category'] == 'inappropriate') & 
            (changed['confidence'] < 0.3) &
            (changed['predicted_category_fixed'] == 'authentic')
        ]
        
        if len(improved) > 0:
            print(f"\n✅ Fixed {len(improved)} misclassified 'inappropriate' reviews:")
            for idx, row in improved.head(3).iterrows():
                print(f"  \"{row['text'][:50]}...\"")
                print(f"    OLD: {row['predicted_category']} ({row['confidence']:.3f})")
                print(f"    NEW: {row['predicted_category_fixed']} ({row['confidence_fixed']:.3f})")
                print()
    
    # Analyze remaining issues
    if 'predicted_category' in df.columns:
        analyze_misclassifications(df)
    
    # Show all classifications
    print(f"\n📋 ALL IMPROVED CLASSIFICATIONS:")
    print("=" * 70)
    for idx, row in df.iterrows():
        text_preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
        old_info = ""
        if 'predicted_category' in df.columns:
            old_info = f" (was: {row['predicted_category']})" if row['predicted_category'] != row['predicted_category_fixed'] else ""
        print(f"{idx+1:2d}. {text_preview:55} → {row['predicted_category_fixed']:12} ({row['confidence_fixed']:.2f}){old_info}")
    
    # Save results
    output_file = "fixed_classified_reviews.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved improved results to {output_file}")

if __name__ == "__main__":
    main()