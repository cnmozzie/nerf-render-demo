import matplotlib.pyplot as plt
import torch

tensors_model = torch.jit.load("tensors.pt")
img_rgb = list(tensors_model.parameters())[0]
img_disp = list(tensors_model.parameters())[1]
img_acc = list(tensors_model.parameters())[2]

#print(img_c[0])
plt.imshow(img_rgb)
plt.savefig("fern_rgb.png")

plt.imshow(img_disp)
plt.savefig("fern_disp.png")

plt.imshow(img_acc)
plt.savefig("fern_acc.png")