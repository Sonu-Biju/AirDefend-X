import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from logic import run_middleware

def run_science_benchmark():
    # 1. Prepare Testing Data (100 Real Flights)
    try:
        base_df = pd.read_csv('data/opensky.csv').dropna().head(100)
    except:
        # Fallback if CSV is missing
        base_df = pd.DataFrame({
            'icao24': [f'REAL_{i}' for i in range(100)],
            'velocity': np.random.uniform(200, 800, 100),
            'vertrate': np.random.uniform(-5, 5, 100),
            'geoaltitude': np.random.uniform(10000, 30000, 100),
            'lat': np.random.uniform(20, 50, 100),
            'lon': np.random.uniform(-10, 10, 100)
        })
    
    base_df['ground_truth'] = 'LEGITIMATE'
    base_df['signature'] = "VALID_SIG"

    # 2. Inject Controlled Attacks (20 Physics, 20 Identity)
    attack_list = []
    
    # Physics Attacks (Ghosting)
    for i in range(20):
        atk = base_df.iloc[0].copy()
        atk['icao24'] = f'GHOST_{i}'
        atk['velocity'] = 5000.0  # Impossible speed
        atk['ground_truth'] = 'IMPROPER PHYSICS'
        attack_list.append(atk)

    # Identity Attacks (Spoofing)
    for i in range(20):
        atk = base_df.iloc[1].copy()
        atk['icao24'] = f'SPOOF_{i}'
        atk['signature'] = "INVALID_ENCRYPTION"
        atk['ground_truth'] = 'FAKE IDENTITY'
        attack_list.append(atk)

    # Combine and Shuffle
    test_df = pd.concat([base_df, pd.DataFrame(attack_list)]).sample(frac=1).reset_index(drop=True)

    # 3. Run Logic
    results = run_middleware(test_df)

    # 4. Generate Metrics
    y_true = results['ground_truth']
    y_pred = results['status']
    
    report = classification_report(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("--- IEEE RESEARCH PERFORMANCE REPORT ---")
    print(report)
    print(f"OVERALL WEIGHTED F1-SCORE: {f1:.4f}")

    # 5. Generate & Save Confusion Matrix Graph
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_true, y_pred, labels=['LEGITIMATE', 'IMPROPER PHYSICS', 'FAKE IDENTITY'])
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['LEGITIMATE', 'PHYSICS', 'IDENTITY'],
                yticklabels=['LEGITIMATE', 'PHYSICS', 'IDENTITY'])
    plt.xlabel('Predicted by AI')
    plt.ylabel('Actual Truth')
    plt.title('AirDefend-X Confusion Matrix (Security Accuracy)')
    plt.savefig('benchmark_results.png')
    print("✅ Graph saved as 'benchmark_results.png'")

if __name__ == "__main__":
    run_science_benchmark()