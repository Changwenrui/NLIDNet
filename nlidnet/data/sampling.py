import random
from torch.utils.data import Sampler


class ClassBalancedBatchSampler(Sampler):

    def __init__(
        self,
        dataset,
        num_classes_per_batch=4,
        num_samples_per_class=2,
        batches_per_epoch=None,
        seed=42,
    ):
        self.dataset = dataset
        self.num_classes_per_batch = int(num_classes_per_batch)
        self.num_samples_per_class = int(num_samples_per_class)
        self.batch_size = self.num_classes_per_batch * self.num_samples_per_class
        self.seed = int(seed)
        self.epoch = 0

        self.label_to_indices = {
            int(k): list(v) for k, v in dataset.label_to_indices.items() if len(v) > 0
        }
        self.labels = sorted(self.label_to_indices.keys())

        if len(self.labels) < self.num_classes_per_batch:
            raise RuntimeError(
                f"Not enough classes for balanced sampler: "
                f"{len(self.labels)} < {self.num_classes_per_batch}"
            )

        if batches_per_epoch is None:
            self.batches_per_epoch = max(1, len(dataset) // self.batch_size)
        else:
            self.batches_per_epoch = int(batches_per_epoch)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1000003)

        for _ in range(self.batches_per_epoch):
            chosen_labels = rng.sample(self.labels, self.num_classes_per_batch)
            batch_indices = []

            for label in chosen_labels:
                indices = self.label_to_indices[label]

                if len(indices) >= self.num_samples_per_class:
                    picked = rng.sample(indices, self.num_samples_per_class)
                else:
                    picked = [
                        rng.choice(indices) for _ in range(self.num_samples_per_class)
                    ]

                batch_indices.extend(picked)

            rng.shuffle(batch_indices)
            yield batch_indices
