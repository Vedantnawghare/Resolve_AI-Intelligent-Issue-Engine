import sys
sys.path.append('modules')

from modules.ml_models import ModelManager

def main():
    print("="*70)
    print("ML MODEL PREDICTION TESTING")
    print("="*70)
    
    # Load trained models
    print("\nLoading trained models...")
    manager = ModelManager()
    
    try:
        manager.load_models()
    except FileNotFoundError:
        print("\n No trained models found!")
        print("Please run: python train_models.py")
        return
    
    # Test cases with different scenarios
    test_cases = [
        {
            'raw': "mera wifi nahi chal raha library mein",
            'cleaned': "wifi not work library"
        },
        {
            'raw': "laptop ka screen kharab hai urgent exam hai",
            'cleaned': "laptop screen broken urgent exam"
        },
        {
            'raw': "assignment submit nahi ho raha deadline today",
            'cleaned': "assignment submit not work deadline today"
        },
        {
            'raw': "AC not working bahut garmi hai building mein",
            'cleaned': "ac not work hot building"
        },
        {
            'raw': "fee receipt nahi mila need urgent verification",
            'cleaned': "fee receipt not get need urgent verification"
        },
        {
            'raw': "Network keeps disconnecting frequently",
            'cleaned': "network disconnect frequently"
        },
        {
            'raw': "Printer down before presentation tomorrow",
            'cleaned': "printer down presentation tomorrow"
        },
        {
            'raw': "Exam login issue start in 1 hour urgent",
            'cleaned': "exam login issue start hour urgent"
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE #{i}")
        print(f"{'='*70}")
        print(f" Raw Input:     {test['raw']}")
        print(f" Cleaned Input: {test['cleaned']}")
        
        # Get prediction
        prediction = manager.predict_issue(test['cleaned'])
        
        print(f"\n PREDICTION RESULTS:")
        print(f"{'─'*70}")
        
        # Category
        print(f"\n  CATEGORY: {prediction['category']}")
        print(f"   Confidence: {prediction['category_confidence']:.1%}")
        print(f"   Top 3 Predictions:")
        for cat_info in prediction['category_top_3']:
            bar = '█' * int(cat_info['confidence'] * 20)
            print(f"      {cat_info['category']:<15} {bar} {cat_info['confidence']:.1%}")
        
        # Priority
        print(f"\n PRIORITY: {prediction['priority']}")
        priority_label = {'P1': 'High', 'P2': 'Medium', 'P3': 'Low'}[prediction['priority']]
        print(f"   Level: {priority_label}")
        print(f"   Confidence: {prediction['priority_confidence']:.1%}")
        
        # Keywords
        print(f"\n KEY DECISION FACTORS:")
        for word, score in prediction['top_keywords'][:5]:
            bar = '▓' * int(score * 30)
            print(f"      {word:<15} {bar} {score:.3f}")
        
        # Review flag
        if prediction['needs_human_review']:
            print(f"\n  HUMAN REVIEW REQUIRED")
            print(f"   Reason: Low confidence ({prediction['overall_confidence']:.1%})")
        else:
            print(f"\n AUTO-ROUTING APPROVED")
            print(f"   Overall Confidence: {prediction['overall_confidence']:.1%}")
    
    print(f"\n{'='*70}")
    print("TESTING COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
