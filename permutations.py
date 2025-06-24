import random

def create_random_permutation(vocab_size):
    destinations = list(range(vocab_size))
    random.shuffle(destinations)
    def permute(i):
        return destinations[i]
    return permute

def create_random_permutation_with_fixed_points(vocab_size, fixed_points):
    p_domain = sorted(set(range(vocab_size)) - set(fixed_points))
    p_range = [t for t in p_domain]
    random.shuffle(p_range)
    permutation = dict(zip(p_domain, p_range))
    def permute(i):
        return permutation.get(i, i)
    return permute


if __name__ == "__main__":
    p = create_random_permutation_with_fixed_points(8, [0, 1, 2, 7])
    for i in range(8):
        print(f"{i} => {p(i)}")
    