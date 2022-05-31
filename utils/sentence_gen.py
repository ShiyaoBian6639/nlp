def gen_semantic(file_path):
    f = open(file_path, "r")
    animals = f.read().split("\n")
    idx = len(animals) // 2
    positive = [f"Doris likes {animals[i]} because they are cute" for i in range(idx)]
    negative = [f"Doris hates {animals[j]} because they are ugly" for j in range(idx, len(animals))]
    label = [0 if i < idx else 1 for i in range(len(animals))]
    return positive, negative, label
