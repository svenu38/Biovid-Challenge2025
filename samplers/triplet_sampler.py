import random
from torch.utils.data import Sampler
from collections import defaultdict

class TripletBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: BiovidDataset (must return 'user_label')
            batch_size: Must be divisible by 3
        """
        assert batch_size % 3 == 0, "Batch size must be divisible by 3"
        self.dataset = dataset
        self.batch_size = batch_size
        self.label_to_indices = defaultdict(list)

        for idx in range(len(dataset)):
            label = dataset[idx]['user_label'].item()
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

    def __iter__(self):
        all_indices = []
        for _ in range(len(self.dataset) // (self.batch_size // 3)):
            triplet_indices = []
            for _ in range(self.batch_size // 3):
                anchor_label = random.choice(self.labels)
                negative_label = random.choice([l for l in self.labels if l != anchor_label])

                anchor, positive = random.sample(self.label_to_indices[anchor_label], 2)
                negative = random.choice(self.label_to_indices[negative_label])

                triplet_indices += [anchor, positive, negative]

            all_indices.append(triplet_indices)

        return iter([i for batch in all_indices for i in batch])  # flatten list

    def __len__(self):
        return len(self.dataset) // self.batch_size
