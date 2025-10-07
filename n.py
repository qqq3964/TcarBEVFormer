# n423.py
# English comments only
# Generate scene names from 0001 to 0423

train_detect = [f"n{i:03d}" for i in range(1, 18)]
train_detect = [f"n{i:03d}" for i in range(18, 28)]

print(train_detect)
print(f"Total scenes: {len(train_detect)}")
