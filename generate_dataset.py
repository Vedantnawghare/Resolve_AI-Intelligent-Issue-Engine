import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from config import (
    CATEGORIES, PRIORITY_LEVELS, STATUS_OPTIONS, 
    DATASET_SIZE, RANDOM_SEED
)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


ISSUE_TEMPLATES = {
    "Network": [
        "WiFi not working in {location}",
        "Cannot connect to campus network",
        "Internet speed very slow in {location}",
        "Network keeps disconnecting",
        "Unable to access online resources",
        "VPN connection failing",
        "Ethernet port not working in {location}",
        "WiFi password not working",
        "Network down in entire {location}",
        "Can't access university portal due to network issue",
        "Urgent: No internet access before exam",
        "Emergency: Network completely down in {location}",
        "WiFi signal weak, can't submit assignment",
        "Network issue preventing online test access",
        # Hinglish variations
        "mera wifi nahi chal raha {location} mein",
        "internet bahut slow hai plz fix karo",
        "network problem hai urgent exam hai",
        "wifi ka password galat hai kya",
        "pls help internet down hai {location}",
    ],
    
    "IT Support": [
        "Laptop not turning on",
        "Forgot password for student portal",
        "Email account locked",
        "Cannot install required software",
        "Printer not working in {location}",
        "Projector not connecting in classroom {location}",
        "Computer freezing frequently",
        "Need software license activation",
        "Screen display issue on lab computer",
        "Mouse and keyboard not responding",
        "Urgent: Can't access exam portal",
        "Critical: Printer down before deadline",
        "System crashed, lost assignment work",
        "Computer won't start in {location}",
        # Hinglish variations
        "laptop ka screen kharab hai help",
        "password bhul gaya reset karo plz",
        "printer kaam nahi kar raha urgent hai",
        "software install nahi ho raha error aa raha",
        "mujhe account unlock karna hai asap",
    ],
    
    "Academic": [
        "Cannot access course materials",
        "Assignment submission portal not working",
        "Quiz not loading properly",
        "Grades not updated on portal",
        "Missing lecture recording from yesterday",
        "Can't enroll in course",
        "Textbook not available in library",
        "Need extension for assignment",
        "Exam schedule conflict",
        "Course registration issue",
        "Urgent: Assignment deadline today, portal down",
        "Emergency: Can't access exam questions",
        "Exam starts in 1 hour, can't login",
        "Quiz due in 30 minutes, system error",
        # Hinglish variations
        "assignment submit nahi ho raha deadline hai",
        "exam mein login issue hai jaldi dekho",
        "grades update nahi hue abhi tak",
        "course material access nahi ho raha",
        "quiz ka link kaam nahi kar raha urgent",
    ],
    
    "Facilities": [
        "AC not working in {location}",
        "Lights not working in classroom",
        "Door lock broken in {location}",
        "Water leakage in {location}",
        "Elevator out of service",
        "Restroom needs cleaning",
        "Broken chairs in {location}",
        "Whiteboard markers needed",
        "Projector screen damaged",
        "Room too hot/cold in {location}",
        "Urgent: AC broken during exam period",
        "Emergency: Water leak damaging equipment",
        "Multiple lights out in exam hall",
        "Heating not working, very cold",
        # Hinglish variations
        "AC kharab hai {location} bahut garmi hai",
        "pani ka leakage hai urgent fix karo",
        "light nahi hai classroom mein problem",
        "chair toot gaya hai replace karo",
        "fan nahi chal raha bahut heat hai",
    ],
    
    "Admin": [
        "Fee receipt not generated",
        "ID card not working",
        "Document verification delay",
        "Scholarship status inquiry",
        "Transcript request pending",
        "Certificate not issued",
        "Admission query",
        "Hostel allotment issue",
        "Library fine dispute",
        "Attendance discrepancy",
        "Urgent: Need transcript for job application deadline",
        "Emergency: ID card needed for exam",
        "Fee receipt needed immediately for verification",
        "Critical: Document required for interview tomorrow",
        # Hinglish variations
        "fee receipt nahi mila abhi tak",
        "id card kaam nahi kar raha gate pe",
        "certificate kab milega pending hai",
        "admission ka process batao plz",
        "hostel allotment mein problem hai urgent",
    ]
}

LOCATIONS = [
    "Library", "Lab 101", "Lab 202", "Building A", "Building B",
    "Auditorium", "Cafeteria", "Hostel Block C", "Computer Center",
    "Classroom 305", "Seminar Hall", "Admin Block", "Sports Complex"
]

# ============================================================================
# PATTERN GENERATORS
# ============================================================================

def generate_timestamp(base_date, days_range=90):
    """Generate realistic timestamp with patterns"""
    random_days = random.randint(0, days_range)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    
    timestamp = base_date - timedelta(
        days=random_days,
        hours=random_hours,
        minutes=random_minutes
    )
    return timestamp

def is_exam_period(timestamp):
    """Check if timestamp falls in exam season"""
    return timestamp.month in [4, 5, 11, 12]

def assign_priority(issue_text, category, timestamp):
    """Assign priority based on urgency signals and context"""
    text_lower = issue_text.lower()
    
    # High priority conditions
    high_priority_keywords = [
        "urgent", "emergency", "critical", "asap", "immediately",
        "completely down", "not working", "can't access", "exam", 
        "deadline", "today", "jaldi", "turant", "zaroor"
    ]
    
    if any(keyword in text_lower for keyword in high_priority_keywords):
        return "P1"
    
    # Medium priority conditions
    medium_priority_keywords = [
        "slow", "issue", "problem", "not loading", "error",
        "broken", "needs", "required", "dikkat", "pareshani"
    ]
    
    if any(keyword in text_lower for keyword in medium_priority_keywords):
        # Higher chance of P2 during exam period
        if is_exam_period(timestamp):
            return random.choice(["P1", "P2"]) if random.random() < 0.6 else "P2"
        return "P2"
    
    # Default to low priority
    return "P3"

def assign_status(priority, timestamp):
    """Assign status based on priority and time elapsed"""
    days_old = (datetime.now() - timestamp).days
    
    if priority == "P1":
        # High priority issues get resolved faster
        if days_old > 2:
            return "Resolved"
        elif days_old > 1:
            return random.choice(["In Progress", "Resolved"])
        else:
            return random.choice(["Assigned", "In Progress"])
    elif priority == "P2":
        if days_old > 5:
            return "Resolved"
        elif days_old > 3:
            return random.choice(["In Progress", "Resolved"])
        else:
            return random.choice(["New", "Assigned", "In Progress"])
    else:  # P3
        if days_old > 10:
            return "Resolved"
        elif days_old > 5:
            return random.choice(["Assigned", "In Progress", "Resolved"])
        else:
            return random.choice(["New", "Assigned"])

# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_dataset(num_issues=DATASET_SIZE):
    """Generate synthetic dataset with realistic patterns"""
    
    issues = []
    base_date = datetime.now()
    
    # Track recurring issues
    recurring_templates = {}
    
    for i in range(num_issues):
        # Select category with weighted distribution
        # Network and IT Support are more common
        category_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        category = np.random.choice(CATEGORIES, p=category_weights)
        
        # Select template
        template = random.choice(ISSUE_TEMPLATES[category])
        
        # Format template with location if needed
        if "{location}" in template:
            issue_text = template.format(location=random.choice(LOCATIONS))
        else:
            issue_text = template
        
        # Create recurring pattern (20% chance)
        if random.random() < 0.20:
            # Check if we've seen this template before
            template_key = (category, template)
            if template_key in recurring_templates:
                # Use same location for recurring issues
                issue_text = recurring_templates[template_key]
            else:
                recurring_templates[template_key] = issue_text
        
        # Generate timestamp
        timestamp = generate_timestamp(base_date)
        
        # Assign priority
        priority = assign_priority(issue_text, category, timestamp)
        
        # Assign status
        status = assign_status(priority, timestamp)
        
        # Create issue record
        issue = {
            "issue_id": f"ISS-{1000 + i}",
            "issue_text": issue_text,
            "category_label": category,
            "priority_label": priority,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "issue_status": status
        }
        
        issues.append(issue)
    
    # Convert to DataFrame
    df = pd.DataFrame(issues)
    
    # Sort by timestamp (newest first)
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate and save dataset"""
    print(f"Generating {DATASET_SIZE} synthetic issues...")
    
    df = generate_dataset(DATASET_SIZE)
    
    # Save to CSV
    from config import DATASET_PATH
    df.to_csv(DATASET_PATH, index=False)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📁 Saved to: {DATASET_PATH}")
    print(f"\n📊 Dataset Summary:")
    print(f"   Total Issues: {len(df)}")
    print(f"\n   Category Distribution:")
    print(df["category_label"].value_counts())
    print(f"\n   Priority Distribution:")
    print(df["priority_label"].value_counts())
    print(f"\n   Status Distribution:")
    print(df["issue_status"].value_counts())
    
    # Show sample issues
    print(f"\n📝 Sample Issues:")
    print(df[["issue_id", "issue_text", "category_label", "priority_label"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
