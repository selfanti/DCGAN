import torchvision.datasets as dset
import torch
import torchvision.transforms as transforms


def get_dataloader(config):
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataroot=config['dataroot']
    image_size=config['image_size']
    batch_size=config['batch_size']
    ngpu=config['ngpu']
    workers=config['workers']
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    return dataloader,device
