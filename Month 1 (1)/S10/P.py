from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Simulate Results (Imbalanced Dataset)
# Imagine 0 = Healthy, 1 = Sick
# We have 10 data points.
y_true = [0, 1, 0, 0, 1, 1, 0, 0, 1, 0] # Actual reality
y_pred = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0] # Model prediction

# 2. Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()


precision = tp / (tp + fp)  # Of all predicted sick, how many were actually sick?
recall = tp / (tp + fn)     # Of all actually sick, how many did we find?
f1 = 2 * (precision * recall) / (precision + recall)

print("--- Manual Calculation ---")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")
print("-" * 20)
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")

# 4. Visual Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Healthy', 'Predicted Sick'],
            yticklabels=['Actual Healthy', 'Actual Sick'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 5. Professional Report
print("\n--- Scikit-Learn Report ---")
print(classification_report(y_true, y_pred, target_names=['Healthy', 'Sick']))