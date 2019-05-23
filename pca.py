#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
#%%
FACE_PATH = "att_faces"  # \\ can be ambiguous
PERSON_NUM = 40
PERSON_FACE_NUM = 10
K = 10  # Number of principle components

raw_img=[]
data_set=[]
data_set_label=[]

def read_data():
    for i in range(1, PERSON_NUM + 1):
        person_path = FACE_PATH + '/s' + str(i)

        for j in range(1, PERSON_FACE_NUM + 1):
            img = cv2.imread(person_path + '/' + str(j) + '.pgm')
            if j == 1:
                raw_img.append(img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img_gray.shape
            img_col = img_gray.reshape(height * width)
            data_set.append(img_col)
            data_set_label.append(i)
    return height, width

#%%
# Import Data
height, width = read_data()
X = np.array(data_set)
Y = np.array(data_set_label)
n_sample, n_feature = X.shape


#%%
# Print some samples
raw_img=np.hstack(raw_img)
cv2.namedWindow("Image")
cv2.imshow('Image',raw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
# A preview of average_face
average_face = np.mean(X, axis=0)
fig = plt.figure()
plt.imshow(average_face.reshape((height, width)), cmap=plt.cm.gray)
plt.title("Average Face", size=12)
plt.xticks(())
plt.yticks(())
plt.show()

#%%
K=20
equalization_X = X - average_face
covariance_X = np.cov(equalization_X.transpose())
svd = TruncatedSVD(n_components=K, random_state=44)
svd.fit(covariance_X)

#%%
plt.figure()
for i in range(1, K + 1):
    plt.subplot(4, 5, i)
    plt.imshow(svd.components_[i - 1].reshape(height, width).reshape((height, width)), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("pca_faces.png",dpi=1000)
plt.show()