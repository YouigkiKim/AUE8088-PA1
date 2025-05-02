import matplotlib.pyplot as plt


efficientnet_b0 = [  64445184, 0.33399999141693115]
efficientnet_b1 = [  95343360, 0.2337999939918518]
efficientnet_b2 = [  110332042, 0.2607]
efficientnet_v2_s = [472324000, 0.3319999873638153]

resnet_18 = [296296000, 0.363]
resnet_50 = [668107000, 0.4027000069618225]
resnet_101= [1274184000,  0.2630000114440918]

resnet_param = [296296000, 668107000, 1274184000]
resnet_acc = [0.363, 0.4027000069618225, 0.2630000114440918]

efficientnet_param = [64445184, 95343360, 110332042, 472324000]
efficientnet_acc = [0.33399999141693115, 0.2337999939918518, 0.2607, 0.3319999873638153]

alexnet1 = [169830000, 0.276]
alexnet2 = [189763000,  0.19760000705718994]
alexnet3 = [106096000, 0.2703000009059906]

alexnet_param = [169830000, 189763000, 106096000]
alexnet_acc = [0.276, 0.19760000705718994, 0.2703000009059906]

plt.figure(figsize=(10, 5))
# plt.scatter(resnet_18[0], resnet_18[1], label='ResNet18', color='blue', marker='*', s=100)

# plt.scatter(resnet_50[0], resnet_50[1], label='ResNet50', color='blue', marker='o', s=100)
# plt.scatter(resnet_101[0], resnet_101[1], label='ResNet101', color='blue', marker='x', s=100)

plt.scatter(efficientnet_b0[0], efficientnet_b0[1], label='EfficientNetB0', color='red', marker='o', s=100)
plt.scatter(efficientnet_b1[0], efficientnet_b1[1], label='EfficientNetB1', color='red', marker='o', s=100)
plt.scatter(efficientnet_b2[0], efficientnet_b2[1], label='EfficientNetB2', color='red', marker='o', s=100)
plt.scatter(efficientnet_v2_s[0], efficientnet_v2_s[1], label='EfficientNetV2S', color='red', marker='*', s=100)



# plt.plot(resnet_param, resnet_acc, label='ResNet', linestyle='--',  color='blue')
plt.plot(efficientnet_param, efficientnet_acc, label='EfficientNet', linestyle='--', color='red')
# plt.plot(alexnet_param, alexnet_acc, label='AlexNet', linestyle='--', color='green')
plt.legend()
plt.xlabel('Number of Parameters')
plt.ylabel('Accuracy')
plt.title("FLOPs")
plt.grid(True)
plt.show()

