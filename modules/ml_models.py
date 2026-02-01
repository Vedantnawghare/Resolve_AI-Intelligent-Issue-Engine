"""
STAGE 3: ML MODELS - CATEGORY & URGENCY PREDICTION
Production-grade ML models for issue classification

Models:
1. Category Classifier: Logistic Regression (5 classes)
2. Urgency Classifier: Linear SVM (3 priority levels)

Features:
- TF-IDF vectorization
- Model training & persistence
- Confidence scores
- Explainable predictions
- Performance metrics

Author: Senior ML Engineering Team
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)

from config import (
    CATEGORIES,
    PRIORITY_LEVELS,
    ML_CONFIG,
    NLP_CONFIG,
    RANDOM_SEED,
    MODELS_DIR,
    PROCESSED_DATA_PATH
)


# ============================================================================
# TF-IDF VECTORIZER
# Why: Convert text to numerical features for ML models
# ============================================================================

class TextVectorizer:
    """
    Convert cleaned text to TF-IDF features
    
    Why TF-IDF (not Word2Vec, BERT, etc.):
    - Lightweight: No GPU needed, fast training
    - Interpretable: Can see which words matter most
    - Effective: Works well for short text classification
    - Simple: Easy to explain to non-technical stakeholders
    
    TF-IDF = Term Frequency × Inverse Document Frequency
    - Term Frequency: How often word appears in document
    - Inverse Document Frequency: How rare word is across all documents
    
    Result: Common words (the, is, a) get low scores
           Category-specific words (wifi, exam, AC) get high scores
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize TF-IDF vectorizer
        
        Parameters explained:
        - max_features: Keep only top N most important words (prevents overfitting)
        - ngram_range: (1,2) = use single words + word pairs
          Example: "wifi not working" → ["wifi", "not", "working", "wifi not", "not working"]
        - min_df: Ignore words appearing in < N documents (remove rare typos)
        - max_df: Ignore words appearing in > N% documents (remove too common words)
        """
        config = config or NLP_CONFIG
        
        self.vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            stop_words=None,  # We already removed stopwords in preprocessing
            lowercase=False,   # Already lowercased in preprocessing
            token_pattern=r'\b\w+\b'
        )
        
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Learn vocabulary and convert texts to vectors
        
        Real-world example:
        Input texts: ["wifi not working", "printer broken", "wifi down"]
        Vocabulary learned: {wifi: 0, not: 1, working: 2, printer: 3, broken: 4, down: 5}
        
        Output matrix (simplified):
                    wifi  not  working  printer  broken  down
        Text 1:     0.8   0.5   0.5      0.0      0.0     0.0
        Text 2:     0.0   0.0   0.0      0.7      0.7     0.0
        Text 3:     0.8   0.0   0.0      0.0      0.0     0.6
        """
        vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return vectors.toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts using learned vocabulary"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first!")
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get list of vocabulary words"""
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_features_for_text(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get most important words in a text (for explainability)
        
        Why this matters:
        - Shows user WHY model made a prediction
        - "Network" category chosen because: ["wifi", "network", "internet"]
        - Builds trust in the system
        """
        if not self.is_fitted:
            return []
        
        vector = self.transform([text])[0]
        feature_names = self.get_feature_names()
        
        # Get word-score pairs
        word_scores = [(feature_names[i], vector[i]) 
                      for i in range(len(vector)) if vector[i] > 0]
        
        # Sort by score (descending)
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores[:top_n]


# ============================================================================
# CATEGORY CLASSIFIER
# Why: Automatically route issues to correct department
# ============================================================================

class CategoryClassifier:
    """
    Predict issue category using Logistic Regression
    
    Why Logistic Regression:
    - Fast training: Trains in seconds, not hours
    - Probabilistic: Gives confidence scores (crucial for human review routing)
    - Interpretable: Can see feature weights
    - Robust: Works well with limited data
    - Multi-class: Handles 5 categories natively
    
    Categories:
    - Network: WiFi, internet, VPN issues
    - IT Support: Hardware, software, login issues
    - Academic: Courses, grades, assignments
    - Facilities: AC, lights, physical infrastructure
    - Admin: Fees, documents, records
    """
    
    def __init__(self, config: Dict = None):
        config = config or ML_CONFIG['category_model']
        
        self.model = LogisticRegression(
            C=config['C'],                    # Regularization (prevents overfitting)
            max_iter=config['max_iter'],       # Iterations for convergence
            random_state=config['random_state'],
            multi_class='multinomial',         # Use softmax for multi-class
            solver='lbfgs',                    # Optimization algorithm
            class_weight='balanced'            # Handle class imbalance
        )
        
        self.is_trained = False
        self.classes = CATEGORIES
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train category classifier
        
        X_train: TF-IDF vectors (shape: [n_samples, n_features])
        y_train: Category labels (shape: [n_samples])
        
        Returns: Training metrics
        """
        print("\n🎯 Training Category Classifier...")
        
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        
        metrics = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'n_samples': len(y_train),
            'n_features': X_train.shape[1]
        }
        
        print(f"✅ Training complete in {training_time:.2f}s")
        print(f"   Training accuracy: {train_accuracy:.3f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict categories"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict category probabilities
        
        Why probabilities matter:
        - Confidence scoring: Route low-confidence cases to humans
        - Multi-label scenarios: Issue might fit multiple categories
        - Explainability: Show user "80% Network, 20% IT Support"
        
        Example output:
        [
            [0.75, 0.15, 0.05, 0.03, 0.02],  # 75% Network, 15% IT Support, ...
            [0.10, 0.60, 0.20, 0.05, 0.05],  # 60% IT Support
        ]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> List[Dict]:
        """
        Predict with confidence scores and explanations
        
        Real-world usage:
        - High confidence (>85%): Auto-route to department
        - Medium confidence (65-85%): Show suggestion, ask confirmation
        - Low confidence (<65%): Route to human for manual classification
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        results = []
        for i, pred in enumerate(predictions):
            confidence = probabilities[i].max()
            
            # Get top 3 categories
            top_3_idx = np.argsort(probabilities[i])[-3:][::-1]
            top_3_categories = [
                {
                    'category': self.classes[idx],
                    'confidence': probabilities[i][idx]
                }
                for idx in top_3_idx
            ]
            
            results.append({
                'predicted_category': pred,
                'confidence': confidence,
                'top_3_categories': top_3_categories,
                'needs_human_review': confidence < 0.65
            })
        
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Metrics explained:
        - Accuracy: Overall correct predictions
        - F1 Score: Balance of precision & recall (important for imbalanced classes)
        - Per-class metrics: How well we classify each category
        """
        print("\n📊 Evaluating Category Classifier...")
        
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{'='*60}")
        print("CATEGORY CLASSIFIER PERFORMANCE")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=self.classes)}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, 
                                                          target_names=self.classes,
                                                          output_dict=True)
        }


# ============================================================================
# URGENCY CLASSIFIER
# Why: Predict priority level for smart routing
# ============================================================================

class UrgencyClassifier:
    """
    Predict urgency/priority using Linear SVM
    
    Why Linear SVM (not Logistic Regression):
    - Better for nuanced boundaries: P1 vs P2 vs P3 have subtle differences
    - Margin maximization: Finds clearest decision boundaries
    - Handles feature interactions: "exam" + "urgent" together signal P1
    - Robust to outliers: SVM focuses on support vectors
    
    Priority Levels:
    - P1 (High): Urgent, blocking, exam-related
    - P2 (Medium): Important but not urgent
    - P3 (Low): Can wait, informational
    """
    
    def __init__(self, config: Dict = None):
        config = config or ML_CONFIG['urgency_model']
        
        self.model = LinearSVC(
            C=config['C'],                    # Regularization
            max_iter=config['max_iter'],
            random_state=config['random_state'],
            class_weight='balanced',          # Handle priority imbalance
            dual=False                        # Faster for n_samples > n_features
        )
        
        self.is_trained = False
        self.classes = PRIORITY_LEVELS
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train urgency classifier"""
        print("\n⚡ Training Urgency Classifier...")
        
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        
        metrics = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'n_samples': len(y_train),
            'n_features': X_train.shape[1]
        }
        
        print(f"Training complete in {training_time:.2f}s")
        print(f"   Training accuracy: {train_accuracy:.3f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict priority levels"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        return self.model.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision scores (distance from decision boundary)
        
        Why this matters:
        - SVM doesn't give probabilities natively
        - Decision scores show confidence: Large positive = very confident
        - Used for custom confidence scoring
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        return self.model.decision_function(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> List[Dict]:
        """
        Predict priority with confidence estimation
        
        Since LinearSVC doesn't give probabilities, we use decision scores
        and normalize them to approximate confidence
        """
        predictions = self.predict(X)
        decision_scores = self.decision_function(X)
        
        results = []
        for i, pred in enumerate(predictions):
            # For multi-class SVM, decision_function returns matrix
            if decision_scores.ndim > 1:
                scores = decision_scores[i]
                # Normalize scores to pseudo-probabilities using softmax
                exp_scores = np.exp(scores - np.max(scores))
                probabilities = exp_scores / exp_scores.sum()
                confidence = probabilities.max()
            else:
                # Binary case (shouldn't happen with 3 classes)
                confidence = 1.0 / (1.0 + np.exp(-abs(decision_scores[i])))
            
            results.append({
                'predicted_priority': pred,
                'confidence': confidence,
                'needs_human_review': confidence < 0.60
            })
        
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate urgency classifier performance"""
        print("\nEvaluating Urgency Classifier...")
        
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{'='*60}")
        print("URGENCY CLASSIFIER PERFORMANCE")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=self.classes)}")
        
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, 
                                                          target_names=self.classes,
                                                          output_dict=True)
        }


# ============================================================================
# Why: Centralized training, saving, loading, and prediction
# ============================================================================

class ModelManager:
    """
    Manage all ML models in one place
    
    Responsibilities:
    - Train all models together
    - Save/load trained models
    - Coordinate predictions
    - Provide unified interface for UI
    """
    
    def __init__(self):
        self.vectorizer = TextVectorizer()
        self.category_classifier = CategoryClassifier()
        self.urgency_classifier = UrgencyClassifier()
        
        self.is_ready = False
    
    def train_all(self, texts: List[str], categories: List[str], 
                  priorities: List[str], test_size: float = 0.2) -> Dict:
        """
        Train all models on dataset
        
        Pipeline:
        1. Vectorize texts → TF-IDF features
        2. Split train/test
        3. Train category model
        4. Train urgency model
        5. Evaluate both
        6. Save models
        
        Why train/test split:
        - Test on unseen data to measure real-world performance
        - Prevent overfitting (model memorizing training data)
        - Honest evaluation
        """
        print("="*70)
        print("TRAINING ML MODELS")
        print("="*70)
        
        # Step 1: Vectorize
        print("\nVectorizing texts...")
        X = self.vectorizer.fit_transform(texts)
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Vocabulary size: {len(self.vectorizer.get_feature_names())}")
        
        # Step 2: Split data
        print(f"\nSplitting data (test_size={test_size})...")
        
        # For category classification
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
            X, categories, test_size=test_size, random_state=RANDOM_SEED,
            stratify=categories  # Ensure balanced split across categories
        )
        
        # For urgency classification
        X_train_urg, X_test_urg, y_train_urg, y_test_urg = train_test_split(
            X, priorities, test_size=test_size, random_state=RANDOM_SEED,
            stratify=priorities
        )
        
        print(f"   Training samples: {len(X_train_cat)}")
        print(f"   Test samples: {len(X_test_cat)}")
        
        # Step 3: Train category model
        cat_train_metrics = self.category_classifier.train(X_train_cat, y_train_cat)
        
        # Step 4: Train urgency model
        urg_train_metrics = self.urgency_classifier.train(X_train_urg, y_train_urg)
        
        # Step 5: Evaluate
        cat_test_metrics = self.category_classifier.evaluate(X_test_cat, y_test_cat)
        urg_test_metrics = self.urgency_classifier.evaluate(X_test_urg, y_test_urg)
        
        self.is_ready = True
        
        # Compile all metrics
        training_results = {
            'category_model': {
                'train': cat_train_metrics,
                'test': cat_test_metrics
            },
            'urgency_model': {
                'train': urg_train_metrics,
                'test': urg_test_metrics
            },
            'vectorizer': {
                'vocabulary_size': len(self.vectorizer.get_feature_names()),
                'n_features': X.shape[1]
            }
        }
        
        print("\n" + "="*70)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("="*70)
        
        return training_results
    
    def save_models(self, save_dir: Path = None):
        """
        Save all trained models to disk
        
        Why save models:
        - Don't retrain every time app starts (slow!)
        - Persistence: Keep trained models across sessions
        - Deployment: Load pre-trained models in production
        """
        save_dir = save_dir or MODELS_DIR
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving models to {save_dir}...")
        
        # Save vectorizer
        vectorizer_path = save_dir / "vectorizer.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved: {vectorizer_path}")
        
        # Save category classifier
        category_path = save_dir / "category_classifier.pkl"
        with open(category_path, 'wb') as f:
            pickle.dump(self.category_classifier, f)
        print(f"Category classifier saved: {category_path}")
        
        # Save urgency classifier
        urgency_path = save_dir / "urgency_classifier.pkl"
        with open(urgency_path, 'wb') as f:
            pickle.dump(self.urgency_classifier, f)
        print(f"Urgency classifier saved: {urgency_path}")
        
        print("All models saved successfully!")
    
    def load_models(self, load_dir: Path = None):
        """Load pre-trained models from disk"""
        load_dir = load_dir or MODELS_DIR
        load_dir = Path(load_dir)
        
        print(f"\nLoading models from {load_dir}...")
        
        try:
            # Load vectorizer
            vectorizer_path = load_dir / "vectorizer.pkl"
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f" Vectorizer loaded")
            
            # Load category classifier
            category_path = load_dir / "category_classifier.pkl"
            with open(category_path, 'rb') as f:
                self.category_classifier = pickle.load(f)
            print(f"Category classifier loaded")
            
            # Load urgency classifier
            urgency_path = load_dir / "urgency_classifier.pkl"
            with open(urgency_path, 'rb') as f:
                self.urgency_classifier = pickle.load(f)
            print(f"Urgency classifier loaded")
            
            self.is_ready = True
            print("All models loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"❌ Model files not found: {e}")
            print("   Please train models first using train_all()")
            raise
    
    def predict_issue(self, cleaned_text: str) -> Dict:
        """
        Predict category and urgency for a single issue
        
        This is the main prediction function used by the UI
        
        Returns complete prediction with:
        - Category prediction & confidence
        - Priority prediction & confidence
        - Top important words (explainability)
        - Human review flag
        """
        if not self.is_ready:
            raise ValueError("Models not ready! Train or load models first.")
        
        # Vectorize text
        X = self.vectorizer.transform([cleaned_text])
        
        # Predict category
        cat_result = self.category_classifier.predict_with_confidence(X)[0]
        
        # Predict urgency
        urg_result = self.urgency_classifier.predict_with_confidence(X)[0]
        
        # Get important features (for explainability)
        top_features = self.vectorizer.get_top_features_for_text(cleaned_text, top_n=10)
        
        # Compile result
        prediction = {
            'category': cat_result['predicted_category'],
            'category_confidence': cat_result['confidence'],
            'category_top_3': cat_result['top_3_categories'],
            'priority': urg_result['predicted_priority'],
            'priority_confidence': urg_result['confidence'],
            'top_keywords': top_features,
            'needs_human_review': cat_result['needs_human_review'] or urg_result['needs_human_review'],
            'overall_confidence': (cat_result['confidence'] + urg_result['confidence']) / 2
        }
        
        return prediction
    
    def batch_predict(self, cleaned_texts: List[str]) -> List[Dict]:
        """Predict for multiple issues efficiently"""
        if not self.is_ready:
            raise ValueError("Models not ready! Train or load models first.")
        
        results = []
        for text in cleaned_texts:
            result = self.predict_issue(text)
            results.append(result)
        
        return results


# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def train_models_from_dataset():
    """
    Train models using processed dataset
    
    This is the main training script
    """
    print("="*70)
    print("ML MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Load processed dataset
    print(f"\nLoading processed dataset...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"   Dataset shape: {df.shape}")
    
    # Extract features and labels
    texts = df['cleaned_text'].tolist()
    categories = df['category_label'].tolist()
    priorities = df['priority_label'].tolist()
    
    # Initialize model manager
    manager = ModelManager()
    
    # Train all models
    results = manager.train_all(texts, categories, priorities, test_size=0.2)
    
    # Save models
    manager.save_models()
    
    # Show summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"\nCategory Model:")
    print(f"  Training Accuracy: {results['category_model']['train']['train_accuracy']:.3f}")
    print(f"  Test Accuracy: {results['category_model']['test']['accuracy']:.3f}")
    print(f"  F1 Score: {results['category_model']['test']['f1_score']:.3f}")
    
    print(f"\nUrgency Model:")
    print(f"  Training Accuracy: {results['urgency_model']['train']['train_accuracy']:.3f}")
    print(f"  Test Accuracy: {results['urgency_model']['test']['accuracy']:.3f}")
    print(f"  F1 Score: {results['urgency_model']['test']['f1_score']:.3f}")
    
    print(f"\nVocabulary Size: {results['vectorizer']['vocabulary_size']}")
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    
    return manager, results


# ============================================================================
# TESTING & DEMONSTRATION
# ============================================================================

def test_predictions():
    """Test predictions on sample issues"""
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)
    
    # Load trained models
    manager = ModelManager()
    manager.load_models()
    
    # Test cases (already cleaned text)
    test_cases = [
        "wifi not work library",
        "laptop screen broken urgent exam",
        "assignment submit portal down deadline today",
        "ac not work building hot",
        "fee receipt not generate need urgent",
        "network disconnect frequently",
        "printer not work presentation tomorrow",
        "exam login issue start hour",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"TEST CASE #{i}")
        print(f"{'─'*70}")
        print(f"Input: {text}")
        
        prediction = manager.predict_issue(text)
        
        print(f"\n📊 PREDICTION:")
        print(f"   Category: {prediction['category']} ({prediction['category_confidence']:.2%} confidence)")
        print(f"   Priority: {prediction['priority']} ({prediction['priority_confidence']:.2%} confidence)")
        print(f"   Needs Review: {'YES' if prediction['needs_human_review'] else 'NO'}")
        
        print(f"\n TOP KEYWORDS:")
        for word, score in prediction['top_keywords'][:5]:
            print(f"      {word}: {score:.3f}")
        
        print(f"\nTOP 3 CATEGORIES:")
        for cat_info in prediction['category_top_3']:
            print(f"      {cat_info['category']}: {cat_info['confidence']:.2%}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Train models
        train_models_from_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test predictions
        test_predictions()
    else:
        print("Usage:")
        print("  python ml_models.py train  - Train and save models")
        print("  python ml_models.py test   - Test predictions")
