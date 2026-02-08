"""Check the value ranges of Boston dataset after normalization."""
import sys
sys.path.append('forgetting_to_improve/forgetting_to_improve')

from helper.data_loader import load_boston

x, y = load_boston()

print("Boston dataset after StandardScaler normalization:")
print(f"  y min: {y.min():.4f}")
print(f"  y max: {y.max():.4f}")
print(f"  y mean: {y.mean():.4f}")
print(f"  y std: {y.std():.4f}")
print(f"\n  Negative values: {(y < 0).sum()} / {len(y)} ({(y < 0).sum()/len(y)*100:.1f}%)")
print(f"  Positive values: {(y > 0).sum()} / {len(y)} ({(y > 0).sum()/len(y)*100:.1f}%)")
print(f"  Zero values: {(y == 0).sum()}")

if (y <= 0).any():
    print("\n⚠️  WARNING: Data contains non-positive values!")
    print("   This will cause issues with relevance pursuit's exponential prior.")
    print("   Solution: Use raw (unnormalized) target values for relevance pursuit,")
    print("   or shift data to be strictly positive.")
