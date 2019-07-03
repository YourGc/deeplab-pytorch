# coding:utf-8
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
# aug = iaa.Sequential([
#             iaa.Fliplr(0.5),
#             iaa.Flipud(0.5)],random_state=666
#         )
#
#
# def expand_mask(mask):
#     Bsmoke, Corn, Brice = mask.copy(), mask.copy(), mask.copy()
#     # print('---')
#     # print((mask==0).sum(),(mask==1).sum(),(mask==2).sum(),(mask==3).sum())
#     Bsmoke[Bsmoke != 1] = 0
#     Bsmoke[Bsmoke == 1] = 1
#
#     Corn[Corn != 2] = 0
#     Corn[Corn == 2] = 1
#
#     Brice[Brice != 3] = 0
#     Brice[Brice == 3] = 1
#
#     mask[mask != 0] = 2
#     mask[mask == 0] = 1
#     mask[mask == 2] = 0
#     # print(mask.sum(),Bsmoke.sum(),Corn.sum(),Brice.sum())
#     masks = [mask, Bsmoke, Corn, Brice]
#     # print(masks.sum())
#     return masks
#
# img = Image.open(r'F:\AIagriculture\data\train_vis\imgs\1_34.jpg')
# plt.imshow(img)
# plt.show()
# img = np.array(img)
# mask = Image.open(r'F:\AIagriculture\data\train_vis\masks\1_34.png')
# plt.imshow(mask)
# plt.show()
# mask = np.array(mask)
#
# import imgaug as ia
# from imgaug import augmenters as iaa
# import imageio
# import numpy as np
#
# plt.matshow(mask[:,:,0])
# plt.show()
# mask = mask[:,:,0]
# mask[mask>1] = 1
# segmap = ia.SegmentationMapOnImage(mask, shape=mask.shape, nb_classes=2)
# # Define our augmentation pipeline.
# seq = iaa.Sequential([
#     #iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
#     #iaa.Sharpen((0.0, 1.0)),       # sharpen the image
#     iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects heatmaps)
#     #iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects heatmaps)
# ], random_order=True)
#
# # Augment images and heatmaps.
# images_aug = []
# segmaps_aug = []
# for _ in range(5):
#     seq_det = seq.to_deterministic()
#     plt.imshow(seq_det.augment_image(img))# append the augmented img array
#     plt.show()
#     plt.matshow(seq_det.augment_segmentation_maps([segmap])[0].get_arr_int().astype(np.uint8))
#     plt.show()
label = Image.open('1_34_ori.png')
pred = Image.open('33_54.png')
#print((label==pred).sum())

label = np.array(label)
pred = np.array(pred)
print(pred)
print((pred==2).sum())

# plt.imshow(label)
# plt.show()
plt.matshow(pred)
plt.show()