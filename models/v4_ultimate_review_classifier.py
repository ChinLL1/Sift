import pandas as pd
from transformers import pipeline
import torch
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class LocationAwareReviewClassifier:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"🚀 Loading location-aware classification model...")
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
        
        # Enhanced category definitions
        self.categories = [
            "genuine visitor review with personal experience about the location",
            "commercial advertisement or promotional spam content",
            "completely unrelated topic that has nothing to do with this type of location",
            "vulgar profanity sexual or violent inappropriate content",
            "personal contact information email or phone number",
            "opinion from someone who never actually visited the location"
        ]
        
        # Mapping back to your labels
        self.label_map = {
            "genuine visitor review with personal experience about the location": "authentic",
            "commercial advertisement or promotional spam content": "advertisement",
            "completely unrelated topic that has nothing to do with this type of location": "off-topic",
            "vulgar profanity sexual or violent inappropriate content": "inappropriate", 
            "personal contact information email or phone number": "personal-info",
            "opinion from someone who never actually visited the location": "never-visited"
        }
        
        # Location-specific keyword mappings
        self.location_keywords = {
            # Museums and cultural sites
            'museum': ['exhibit', 'artifacts', 'collection', 'curator', 'display', 'gallery', 'history', 'educational', 'guided tour', 'interactive'],
            'castle': ['fortress', 'palace', 'medieval', 'architecture', 'rooms', 'throne', 'royal', 'heritage', 'stone', 'tower', 'courtyard'],
            'park': ['nature', 'trees', 'walking', 'picnic', 'playground', 'gardens', 'peaceful', 'green space', 'outdoor', 'recreation'],
            'restaurant': ['food', 'service', 'menu', 'waiter', 'delicious', 'meal', 'dining', 'cuisine', 'flavor', 'staff', 'atmosphere'],
            'hotel': ['room', 'service', 'staff', 'comfortable', 'clean', 'bed', 'bathroom', 'reception', 'amenities', 'stay'],
            'shopping': ['store', 'shop', 'price', 'quality', 'purchase', 'buy', 'selection', 'sales', 'customer service', 'merchandise'],
            'entertainment': ['fun', 'exciting', 'show', 'performance', 'audience', 'experience', 'venue', 'ticket', 'crowd'],
            'religious': ['prayer', 'worship', 'faith', 'peaceful', 'spiritual', 'ceremony', 'sacred', 'blessing', 'congregation'],
            'educational': ['learning', 'students', 'knowledge', 'information', 'educational', 'teaching', 'academic'],
            'healthcare': ['doctor', 'treatment', 'care', 'medical', 'health', 'professional', 'appointment', 'service'],
            'transport': ['convenient', 'schedule', 'route', 'travel', 'journey', 'transportation', 'access'],
            # Default fallback
            'default': ['visited', 'experience', 'recommend', 'place', 'location', 'good', 'bad', 'nice', 'terrible']
        }
        
        # General authentic indicators
        self.authentic_indicators = [
            'visited', 'tour', 'experience', 'recommend', 'went', 'saw', 'admission', 'crowded', 
            'weekend', 'beautiful', 'amazing', 'terrible', 'disappointed', 'loved', 'enjoyed'
        ]
    
    def get_location_category_key(self, category_info):
        """Extract the most relevant category key from location information."""
        if pd.isna(category_info) or category_info == '':
            return 'default'
        
        category_lower = str(category_info).lower()
        
        # Map common category patterns to our keyword groups
        category_mappings = {
            'museum': ['museum', 'history', 'cultural', 'heritage', 'memorial'],
            'castle': ['castle', 'palace', 'fortress', 'historic site', 'monument'],
            'park': ['park', 'garden', 'nature', 'outdoor', 'recreation'],
            'restaurant': ['restaurant', 'cafe', 'dining', 'food', 'bar', 'bistro'],
            'hotel': ['hotel', 'accommodation', 'lodging', 'resort', 'inn'],
            'shopping': ['shopping', 'mall', 'store', 'retail', 'market'],
            'entertainment': ['entertainment', 'theater', 'cinema', 'venue', 'club'],
            'religious': ['church', 'temple', 'mosque', 'religious', 'worship'],
            'educational': ['school', 'university', 'library', 'educational'],
            'healthcare': ['hospital', 'clinic', 'medical', 'healthcare'],
            'transport': ['station', 'airport', 'transport', 'transit']
        }
        
        for key, patterns in category_mappings.items():
            if any(pattern in category_lower for pattern in patterns):
                return key
        
        return 'default'
    
    def rule_based_check(self, text, location_category=None):
        """Enhanced rule-based classification with location context."""
        text_lower = text.lower().strip()
        
        # Get relevant keywords for this location type
        category_key = self.get_location_category_key(location_category)
        relevant_keywords = self.location_keywords.get(category_key, self.location_keywords['default'])
        
        # Enhanced authentic detection with location context
        authentic_score = 0
        
        # Score for general authentic indicators
        authentic_score += sum(1 for word in self.authentic_indicators if word in text_lower)
        
        # Bonus score for location-specific keywords
        location_score = sum(1 for word in relevant_keywords if word in text_lower)
        authentic_score += location_score * 1.5  # Give extra weight to location-specific terms
        
        # If we have strong indicators of authentic review
        if authentic_score >= 2:
            confidence = min(0.95, 0.7 + (authentic_score * 0.05))
            return "authentic", confidence
        
        # Clear advertisement patterns (unchanged - these work well)
        ad_patterns = [
            r'get up to \d+ months? free',
            r'subscribe now',
            r'premium',
            r'spotify',
            r'free trial',
            r'discount',
            r'click here',
            r'download.*app',
            r'visit.*website',
            r'call.*now'
        ]
        
        for pattern in ad_patterns:
            if re.search(pattern, text_lower):
                return "advertisement", 0.95
        
        # Personal information patterns
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        phone_pattern = r'[\+]?[1-9]?[\d\s\-\(\)]{7,15}\d'
        
        if re.search(email_pattern, text) or re.search(phone_pattern, text):
            return "personal-info", 0.95
        
        # Enhanced off-topic detection with location context
        strong_off_topic_words = ['caterpillar', 'bird', 'animal', 'guitar', 'movie', 'cooking', 'homework', 'school project']
        location_related_words = relevant_keywords + ['place', 'location', 'here', 'there', 'visit']
        
        has_strong_off_topic = any(word in text_lower for word in strong_off_topic_words)
        has_location_context = any(word in text_lower for word in location_related_words)
        
        # Only classify as off-topic if it has strong off-topic indicators AND no location context
        if has_strong_off_topic and not has_location_context and len(text.split()) > 3:
            return "off-topic", 0.9
        
        # Check for very generic off-topic content that doesn't match location type
        if category_key != 'default':
            # For specific location types, check if review mentions completely unrelated topics
            unrelated_topics = {
                'museum': ['taste', 'flavor', 'spicy', 'sweet', 'bed', 'room service'],
                'restaurant': ['exhibits', 'artifacts', 'historical', 'medieval architecture'],
                'hotel': ['exhibits', 'artifacts', 'menu', 'delicious food'],
                'park': ['room service', 'menu', 'exhibits', 'artifacts'],
                'castle': ['room service', 'delicious', 'spicy', 'flavor']
            }
            
            if category_key in unrelated_topics:
                unrelated_words = unrelated_topics[category_key]
                has_unrelated = sum(1 for word in unrelated_words if word in text_lower)
                if has_unrelated >= 2 and not has_location_context:
                    return "off-topic", 0.85
        
        # Never-visited patterns (enhanced)
        never_visited_patterns = [
            (r'haven[\'t]*.*visit.*but', 0.95),
            (r'never.*been.*but.*think', 0.9),
            (r'haven[\'t]*.*opportunity.*visit', 0.9),
            (r'never.*went.*but', 0.85),
            (r'didn[\'t]*.*go.*but.*looks', 0.85),
            (r'from what.*looks.*like.*think', 0.9),
            (r'planning.*visit.*heard', 0.75),
            (r'want.*go.*someday', 0.7),
            (r'hope to visit', 0.7),
            (r'would like.*visit.*heard', 0.75)
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
            
            # Strong profanity in very negative context
            (r'i fucking hate|fuck.*place|fucking.*sucks', 0.85),
            (r'this.*fucking.*terrible|fucking.*worst', 0.85),
            
            # Threats or harassment
            (r'go to hell|screw you|piss off|fuck off', 0.9)
        ]
        
        for pattern, confidence in inappropriate_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return "inappropriate", confidence
        
        return None, 0.0
    
    def ml_classify(self, text, location_category=None):
        """ML-based classification with location context."""
        try:
            # Enhance the text with location context for better classification
            category_key = self.get_location_category_key(location_category)
            
            # Create context-aware prompt
            if category_key != 'default':
                enhanced_text = f"Review about a {category_key}: {text}"
            else:
                enhanced_text = text
            
            result = self.classifier(enhanced_text, self.categories)
            best_category = result['labels'][0]
            confidence = result['scores'][0]
            simple_label = self.label_map.get(best_category, "authentic")
            
            # Boost confidence for short positive reviews that match location context
            if len(text.split()) <= 3:
                category_keywords = self.location_keywords.get(category_key, [])
                general_positive = ['beautiful', 'excellent', 'amazing', 'wonderful', 'great', 'superb', 'fantastic']
                
                has_positive = any(word in text.lower() for word in general_positive)
                has_location_relevance = any(word in text.lower() for word in category_keywords)
                
                if has_positive or has_location_relevance:
                    simple_label = "authentic"
                    confidence = max(confidence, 0.8)
            
            return simple_label, confidence
            
        except Exception as e:
            print(f"ML classification error: {e}")
            return "authentic", 0.5  # Default to authentic when in doubt
    
    def classify_review(self, text, location_category=None):
        """Hybrid classification with location context."""
        if not text or text.strip() == "":
            return "authentic", 0.3
        
        # First try rule-based with location context
        rule_category, rule_confidence = self.rule_based_check(text, location_category)
        if rule_confidence > 0.8:
            return rule_category, rule_confidence
        
        # For very short texts, be more conservative but consider location context
        if len(text.split()) <= 2:
            category_key = self.get_location_category_key(location_category)
            relevant_keywords = self.location_keywords.get(category_key, [])
            
            # Check if short text is relevant to location type
            has_location_relevance = any(word in text.lower() for word in relevant_keywords)
            general_positive = ['beautiful', 'excellent', 'amazing', 'wonderful', 'great', 'superb']
            has_positive = any(word in text.lower() for word in general_positive)
            
            if has_location_relevance or has_positive:
                return "authentic", 0.85
            return "authentic", 0.65  # Short texts are usually authentic
        
        # Fall back to ML with location context
        ml_category, ml_confidence = self.ml_classify(text, location_category)
        
        # Enhanced override logic with location awareness
        if ml_category == "inappropriate" and ml_confidence < 0.4:
            # Check if it's actually a normal review for this location type
            category_key = self.get_location_category_key(location_category)
            relevant_keywords = self.location_keywords.get(category_key, [])
            
            if any(word in text.lower() for word in relevant_keywords + self.authentic_indicators):
                return "authentic", 0.75
        
        # Check if ML classified as off-topic but it's actually relevant to location
        if ml_category == "off-topic" and ml_confidence < 0.7:
            category_key = self.get_location_category_key(location_category)
            relevant_keywords = self.location_keywords.get(category_key, [])
            
            # If text contains location-specific keywords, it's probably on-topic
            location_relevance = sum(1 for word in relevant_keywords if word in text.lower())
            if location_relevance >= 1:
                return "authentic", max(0.7, 1 - ml_confidence)
        
        # If ML is uncertain, boost confidence with rule hints
        if rule_confidence > 0.5 and ml_confidence < 0.7:
            return rule_category, max(rule_confidence, ml_confidence)
        
        return ml_category, ml_confidence
    
    def classify_dataframe(self, df):
        """Classify all reviews with location context and progress tracking."""
        print(f"🔍 Classifying {len(df)} reviews with location context...")
        
        predictions = []
        confidences = []
        
        # Determine which column to use for location category
        category_column = None
        if 'categoryName' in df.columns:
            category_column = 'categoryName'
        elif 'category' in df.columns:
            category_column = 'category'
        elif 'categories/0' in df.columns:
            category_column = 'categories/0'
        
        if category_column:
            print(f"📍 Using '{category_column}' for location context")
        else:
            print("⚠️  No location category column found, proceeding without location context")
        
        for idx, row in tqdm(df.iterrows(), desc="Processing", total=len(df)):
            text = row['text']
            location_category = row[category_column] if category_column else None
            
            category, confidence = self.classify_review(text, location_category)
            predictions.append(category)
            confidences.append(confidence)
        
        return predictions, confidences

def analyze_location_context(df):
    """Analyze how location context affects classifications."""
    print(f"\n🏢 LOCATION CONTEXT ANALYSIS")
    print("=" * 60)
    
    if 'categoryName' in df.columns:
        category_col = 'categoryName'
    elif 'category' in df.columns:
        category_col = 'category'
    else:
        print("No location category column found for analysis")
        return
    
    # Analyze by location type
    location_analysis = df.groupby(category_col).agg({
        'predicted_category_enhanced': lambda x: x.value_counts().to_dict(),
        'confidence_enhanced': 'mean'
    }).round(3)
    
    print(f"Classification distribution by location type:")
    for location_type in df[category_col].unique():
        if pd.notna(location_type):
            subset = df[df[category_col] == location_type]
            print(f"\n📍 {location_type} ({len(subset)} reviews):")
            category_counts = subset['predicted_category_enhanced'].value_counts()
            for cat, count in category_counts.head(3).items():
                pct = count / len(subset) * 100
                avg_conf = subset[subset['predicted_category_enhanced'] == cat]['confidence_enhanced'].mean()
                print(f"   {cat}: {count} ({pct:.1f}%, avg conf: {avg_conf:.2f})")

def analyze_improvements(df):
    """Compare original vs enhanced classifications."""
    if 'predicted_category' not in df.columns:
        return
    
    print(f"\n📈 IMPROVEMENT ANALYSIS")
    print("=" * 60)
    
    # Count changes
    changed = df[df['predicted_category'] != df['predicted_category_enhanced']]
    print(f"Changed classifications: {len(changed)}/{len(df)} ({len(changed)/len(df)*100:.1f}%)")
    
    # Analyze confidence improvements
    conf_improved = df[df['confidence_enhanced'] > df['confidence']]
    print(f"Confidence improved: {len(conf_improved)}/{len(df)} ({len(conf_improved)/len(df)*100:.1f}%)")
    
    avg_conf_old = df['confidence'].mean() if 'confidence' in df.columns else 0
    avg_conf_new = df['confidence_enhanced'].mean()
    print(f"Average confidence: {avg_conf_old:.3f} → {avg_conf_new:.3f}")
    
    # Show key improvements
    key_improvements = changed[
        (changed['predicted_category'] == 'inappropriate') & 
        (changed['confidence'] < 0.4) &
        (changed['predicted_category_enhanced'] == 'authentic')
    ]
    
    if len(key_improvements) > 0:
        print(f"\n✅ Fixed {len(key_improvements)} misclassified reviews:")
        for idx, row in key_improvements.head(3).iterrows():
            location_info = ""
            if 'categoryName' in df.columns and pd.notna(row['categoryName']):
                location_info = f" [{row['categoryName']}]"
            print(f"  \"{row['text'][:50]}...\"{location_info}")
            print(f"    OLD: {row['predicted_category']} ({row['confidence']:.3f})")
            print(f"    NEW: {row['predicted_category_enhanced']} ({row['confidence_enhanced']:.3f})")

def main():
    # Load data
    print("📁 Loading data...")
    try:
        # Ensure path is at /Sift.
        df = pd.read_csv("data/test_reviews.csv")
        df['text'] = df['text'].fillna('').astype(str)
        print(f"✅ Loaded {len(df)} reviews")
        
        # Show available columns
        print(f"📊 Available columns: {', '.join(df.columns.tolist())}")
        
        # Show location category distribution
        if 'categoryName' in df.columns:
            print(f"\n📍 Location types:")
            location_counts = df['categoryName'].value_counts().head(10)
            for loc_type, count in location_counts.items():
                print(f"   {loc_type}: {count}")
        
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
            ],
            'categoryName': [
                "Museum",
                "Museum", 
                "Museum",
                "Castle",
                "Castle",
                "Park",
                "Museum",
                "Museum"
            ]
        }
        df = pd.DataFrame(sample_data)
        print(f"✅ Using sample data with {len(df)} reviews")
    
    # Initialize location-aware classifier
    classifier = LocationAwareReviewClassifier()
    
    # Classify with location context
    predictions, confidences = classifier.classify_dataframe(df)
    
    # Add results
    df['predicted_category_enhanced'] = predictions
    df['confidence_enhanced'] = confidences
    
    # Analysis
    print(f"\n📊 ENHANCED RESULTS SUMMARY")
    print("=" * 50)
    
    category_counts = df['predicted_category_enhanced'].value_counts()
    for category, count in category_counts.items():
        pct = count / len(df) * 100
        print(f"{category:15}: {count:2d} ({pct:4.1f}%)")
    
    avg_confidence = df['confidence_enhanced'].mean()
    high_conf = len(df[df['confidence_enhanced'] > 0.8])
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    print(f"High confidence (>80%): {high_conf}/{len(df)} ({high_conf/len(df)*100:.1f}%)")
    
    # Compare with original if available
    if 'predicted_category' in df.columns:
        analyze_improvements(df)
    
    # Analyze location context effects
    analyze_location_context(df)
    
    # Show all classifications
    print(f"\n📋 ALL ENHANCED CLASSIFICATIONS:")
    print("=" * 90)
    
    # Determine which category column to display
    display_category_col = 'categoryName' if 'categoryName' in df.columns else ('category' if 'category' in df.columns else None)
    
    for idx, row in df.iterrows():
        text_preview = row['text'][:45] + "..." if len(row['text']) > 45 else row['text']
        
        # Show location context
        location_info = ""
        if display_category_col and pd.notna(row[display_category_col]):
            location_info = f"[{row[display_category_col]}] "
        
        # Show comparison if original exists
        change_info = ""
        if 'predicted_category' in df.columns and row['predicted_category'] != row['predicted_category_enhanced']:
            change_info = f" (was: {row['predicted_category']})"
        
        print(f"{idx+1:2d}. {location_info}{text_preview:50} → {row['predicted_category_enhanced']:12} ({row['confidence_enhanced']:.2f}){change_info}")
    
    # Save results
    output_file = "data/enhanced_classified_reviews.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved enhanced results to {output_file}")

if __name__ == "__main__":
    main()