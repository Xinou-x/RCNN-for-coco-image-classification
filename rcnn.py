import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# 数据集路径
data_dir = '.'
image_dir = './images'
data_type = 'train2014'
ann_file = f'{data_dir}/annotations/instances_{data_type}.json'

# 初始化COCO api
coco = COCO(ann_file)

# 获取指定类别的图像
cat_ids = coco.getCatIds(catNms=['person', 'dog', 'cat'])
print("Number of categories: {}, Category_id: {}\n".format(len(cat_ids), cat_ids))

# 获取包含每个类别的图像ID
img_ids = set()
for cat_id in cat_ids:
    img_ids_for_cat = coco.getImgIds(catIds=[cat_id])
    img_ids.update(img_ids_for_cat)
print("Number of img: {}".format(len(img_ids)))

img_info = coco.loadImgs(list(img_ids))

# 定义IoU计算函数
def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# 初始化选择性搜索
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# 存储训练图像和标签
train_images = []
train_labels = []

# 迭代图像信息
for e, img in enumerate(img_info[0:500]):
    print(e)
    try:
        filename = img['file_name']
        img_path = os.path.join(image_dir, data_type)
        img_path = os.path.join(img_path, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: Unable to read image {filename}")
            continue

        ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        gtvalues = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            category_id = ann['category_id']
            gtvalues.append({
                "x1": int(x),
                "x2": int(x + w),
                "y1": int(y),
                "y2": int(y + h),
                "category_id": category_id
            })

        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()

        imout = image.copy()
        counter = 0
        falsecounter = 0
        flag = 0
        fflag = 0
        bflag = 0

        for e, result in enumerate(ssresults):
            if e < 2000 and flag == 0:
                for gtval in gtvalues:
                    x, y, w, h = result
                    iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                    if counter < 30:
                        if iou > 0.70:
                            timage = imout[y:y + h, x:x + w]
                            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(gtval["category_id"])
                            counter += 1
                    else:
                        fflag = 1
                    if falsecounter < 30:
                        if iou < 0.3:
                            timage = imout[y:y + h, x:x + w]
                            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(0)  # 0表示背景
                            falsecounter += 1
                    else:
                        bflag = 1
                if fflag == 1 and bflag == 1:
                    print("inside")
                    flag = 1
    except Exception as e:
        print(e)
        print("error in " + filename)
        continue

X_new = np.array(train_images)
y_new = np.array(train_labels)
print(X_new.shape)

# 修改VGG16模型
input_shape = (224, 224, 3)
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))

# 添加自定义顶层
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(cat_ids) + 1, activation='softmax')(x)
model_final = Model(inputs=base_model.input, outputs=predictions)

# 冻结VGG16的部分卷积层
for layer in base_model.layers:
    layer.trainable = False

opt = Adam(learning_rate=0.0001)
model_final.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model_final.summary())

class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)

lenc = MyLabelBinarizer()
Y = lenc.fit_transform(y_new)

X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.10)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
traindata = trdata.flow(x=X_train, y=y_train)
tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
testdata = tsdata.flow(x=X_test, y=y_test)

checkpoint = ModelCheckpoint("rcnn_vgg16.keras", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

hist = model_final.fit(traindata, steps_per_epoch=10, epochs=15, validation_data=testdata, validation_steps=2, callbacks=[checkpoint, early])

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"])
plt.show()
plt.savefig('chart_loss.png')

# 预测并绘制结果
for e, img in enumerate(img_info[0:1]):
    print(e)
    filename = img['file_name']
    img_path = os.path.join(image_dir, data_type)
    img_path = os.path.join(img_path, filename)
    img = cv2.imread(img_path)
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    imout = img.copy()
    for e, result in enumerate(ssresults):
        if e < 2000:
            x, y, w, h = result
            timage = imout[y:y + h, x:x + w]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
            img_array = np.expand_dims(resized, axis=0)
            pred = model_final.predict(img_array)
            if pred[0].max() > 0.65:
                category = np.argmax(pred[0])
                if category > 0:
                    label = coco.loadCats(cat_ids[category - 1])[0]['name']
                    cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(imout, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(imout, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.savefig('detection_result.png')

