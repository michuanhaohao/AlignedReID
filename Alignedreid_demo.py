import torch
from util.FeatureExtractor import FeatureExtractor
from torchvision import transforms
from IPython import embed
import models
from scipy.spatial.distance import cosine, euclidean
from  util.utils import *
from sklearn.preprocessing import normalize

def pool2d(tensor, type= 'max'):
    sz = tensor.size()
    if type == 'max':
        x = torch.nn.functional.max_pool2d(tensor, kernel_size=(sz[2]/8, sz[3]))
    if type == 'mean':
        x = torch.nn.functional.mean_pool2d(tensor, kernel_size=(sz[2]/8, sz[3]))
    x = x[0].cpu().data.numpy()
    x = np.transpose(x,(2,1,0))[0]
    return x

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    use_gpu = torch.cuda.is_available()
    model = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'}, use_gpu=use_gpu,aligned=True)
    checkpoint = torch.load("./log/market1501/alignedreid/checkpoint_ep300.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])

    img_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    exact_list = ['7']
    myexactor = FeatureExtractor(model, exact_list)
    img_path1 = './data/market1501/query/0001_c1s1_001051_00.jpg'
    img_path2 = './data/market1501/query/0001_c2s1_000301_00.jpg'
    img1 = read_image(img_path1)
    img2 = read_image(img_path2)
    img1 = img_to_tensor(img1, img_transform)
    img2 = img_to_tensor(img2, img_transform)
    if use_gpu:
        model = model.cuda()
        img1 = img1.cuda()
        img2 = img2.cuda()
    model.eval()
    f1 = myexactor(img1)
    f2 = myexactor(img2)
    a1 = normalize(pool2d(f1[0], type='max'))
    a2 = normalize(pool2d(f2[0], type='max'))
    dist = np.zeros((8,8))
    for i in range(8):
        temp_feat1 = a1[i]
        for j in range(8):
            temp_feat2 = a2[j]
            dist[i][j] = euclidean(temp_feat1, temp_feat2)
    show_alignedreid(img_path1, img_path2, dist)