
import sys,os,glob,shutil,json
import torchvision
import webdataset as wds

for d in ['logs', 'caltech']:
    dd = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], d)
    os.system(f'mkdir -p {dd}; ln -sf {dd} .;')

dataset = torchvision.datasets.Caltech101(root="/data/zhongz2/data/.data", download=False)
with wds.ShardWriter("./caltech/caltech-%06d.tar", maxcount=1000) as sink:
    for index, (pil_image, class_label) in enumerate(dataset):
        width, height = pil_image.size
        sink.write({
            "__key__": "sample%06d" % index,
            "jpg": pil_image,
            "txt": "The label is {}".format(dataset.categories[class_label]),
            "json": {"height": height, "width": width}
        })



import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
training_urls = "/lscratch/35136020/caltech/caltech-{000000..000008}.tar"
# The standard TorchVision transformations.

transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def make_sample(sample, val=False):
    """Take a decoded sample dictionary, augment it, and return an (image, label) tuple."""
    assert not val, "only implemented training dataset for this notebook"
    print(sample.keys())
    image = sample["jpg"]
    label = sample["txt"]
    return transform_train(image), label

# Create the datasets with shard and sample shuffling and decoding.
trainset = wds.WebDataset(
    training_urls, resampled=True, shardshuffle=True
)
trainset = trainset.shuffle(1000).decode("pil").map(make_sample)
# Since this is an IterableDataset, PyTorch requires that we batch in the dataset.
# WebLoader is PyTorch DataLoader with some convenience methods.
trainset = trainset.batched(64)
trainloader = wds.WebLoader(trainset, batch_size=None, num_workers=4)

# Unbatch, shuffle between workers, then rebatch.
trainloader = trainloader.unbatched().shuffle(1000).batched(64)

for data in trainloader:
    break















