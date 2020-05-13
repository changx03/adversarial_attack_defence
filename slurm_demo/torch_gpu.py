import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))

if device.type == 'cuda':
	print(torch.cuda.get_device_name(0))
