import pandas as pd
import json
from sklearn.model_selection import train_test_split

print("="*70)
print("DATA PREPARATION - Splitting True.csv & Fake.csv")
print("="*70)

# Step 1: Load both datasets
print("\n[1/4] Loading datasets...")
try:
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')
    print(f"✓ Loaded {len(true_df)} real news articles")
    print(f"✓ Loaded {len(fake_df)} fake news articles")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Make sure True.csv and Fake.csv are in the same directory")
    exit()

# Step 2: Combine and label
print("\n[2/4] Combining datasets and adding labels...")
true_df['label'] = 0  # 0 = Real
fake_df['label'] = 1  # 1 = Fake

# Get text content (handle both 'text' and 'content' column names)
true_df['text'] = true_df.get('text', true_df.get('content', ''))
fake_df['text'] = fake_df.get('text', fake_df.get('content', ''))

# Keep only text and label columns
true_df = true_df[['text', 'label']]
fake_df = fake_df[['text', 'label']]

# Combine
df = pd.concat([true_df, fake_df], ignore_index=True)
print(f"✓ Combined total: {len(df)} articles")
print(f"  - Real (0): {len(df[df['label'] == 0])}")
print(f"  - Fake (1): {len(df[df['label'] == 1])}")

# Step 3: Split into training and test sets
print("\n[3/4] Splitting data (80% train, 20% test)...")
train_df, test_df = train_test_split(
    df, 
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"✓ Training set: {len(train_df)} articles")
print(f"  - Real: {len(train_df[train_df['label'] == 0])}")
print(f"  - Fake: {len(train_df[train_df['label'] == 1])}")
print(f"✓ Test set: {len(test_df)} articles")
print(f"  - Real: {len(test_df[test_df['label'] == 0])}")
print(f"  - Fake: {len(test_df[test_df['label'] == 1])}")

# Step 4: Save files
print("\n[4/4] Saving files...")

# Save training data as CSV (for the /add_data API)
train_df.to_csv('training_data.csv', index=False, encoding='utf-8')
print("✓ Saved: training_data.csv")

# Save test samples as JSON (for manual testing)
test_samples = {
    "metadata": {
        "total": len(test_df),
        "real_count": len(test_df[test_df['label'] == 0]),
        "fake_count": len(test_df[test_df['label'] == 1]),
        "purpose": "Never seen by the model - for professor verification"
    },
    "real_samples": [
        {
            "id": idx,
            "text": row['text'][:1000],  # First 1000 chars
            "full_text": row['text'],
            "label": "REAL"
        }
        for idx, (_, row) in enumerate(test_df[test_df['label'] == 0].iterrows())
    ],
    "fake_samples": [
        {
            "id": idx,
            "text": row['text'][:1000],  # First 1000 chars
            "full_text": row['text'],
            "label": "FAKE"
        }
        for idx, (_, row) in enumerate(test_df[test_df['label'] == 1].iterrows())
    ]
}

with open('test_samples.json', 'w', encoding='utf-8') as f:
    json.dump(test_samples, f, indent=2, ensure_ascii=False)
print("✓ Saved: test_samples.json")

# Also save a human-readable test file
with open('test_samples.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("TEST SAMPLES - NEVER SEEN BY MODEL\n")
    f.write("Use these to verify model predictions on unseen data\n")
    f.write("="*70 + "\n\n")
    
    f.write("REAL NEWS SAMPLES (Model should predict REAL):\n")
    f.write("-"*70 + "\n\n")
    for idx, (_, row) in enumerate(test_df[test_df['label'] == 0].head(5).iterrows()):
        f.write(f"Sample {idx+1}:\n{row['text'][:500]}...\n\n")
    
    f.write("\n" + "="*70 + "\n\n")
    f.write("FAKE NEWS SAMPLES (Model should predict FAKE):\n")
    f.write("-"*70 + "\n\n")
    for idx, (_, row) in enumerate(test_df[test_df['label'] == 1].head(5).iterrows()):
        f.write(f"Sample {idx+1}:\n{row['text'][:500]}...\n\n")

print("✓ Saved: test_samples.txt")

print("\n" + "="*70)
print("✓ DATA PREPARATION COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Start Training API: python training_api.py")
print("2. Upload training_data.csv via /add_data endpoint")
print("3. Train model via /post /train endpoint")
print("4. Use test_samples.json to verify predictions")
print("5. Show professor the cross-validation scores!")
print("="*70)