from pycocotools.coco import COCO
import glob
import clip
from tqdm import tqdm
import torch
from PIL import Image
import re
import os
import faiss

class CocoDataSet:
    def __init__(self, captions, instance):
        self.instance = COCO(instance)
        self.captions = COCO(captions)

    def loadImgsFromCategories(self, cats):
        """
        Retrieves the images object based on the category names

        Args:
          cats (str): List of category names
        
        Return:
          list: A list of image objects.
        """
        catIDs = self.instance.getCatIds(catNms=cats)
        imgIDs = self.instance.getImgIds(catIds=catIDs)

        return self.instance.loadImgs(imgIDs)
    
    def getImgFromCatetoryName(self, cats_name):
        ids = self.instance.getCatIds(catNms=cats_name)
        img_ids = self.instance.getImgIds(catIds=ids)
        imgs = self.instance.loadImgs(img_ids)
        return imgs
    
    def getImageBlobs(self, path):
        return list(glob.glob(path + "/*.jpg"))

    def idToFile(self, id):
        img = self.instance.loadImgs(id)[0]
        return f"coco/images/val2017/{img['file_name']}"

    def fileToId(self, path):
        f = os.path.basename(path)
        match = re.search(r"(\d+)\.jpg", f)
        image_id = int(match.group(1)) if match else None

        return image_id


class ClipWrapper:
    def __init__(self, model, device):
        m, p = clip.load(model, device=device)
        self.model = m
        self.preprocess = p
        self.device = device

    def generateImageEmbeddings(self, coco, path):
        m = {}
        for img in tqdm(path):
          if not os.path.exists(img):
            continue
          i = self.preprocess(Image.open(img)).unsqueeze(0).to(self.device)
          with torch.no_grad():
            embedding = self.model.encode_image(i)
          embedding /= embedding.norm(dim=-1, keepdim=True)
          m[coco.fileToId(img)] = embedding.cpu().numpy().astype('float32')
        return m
    
    def loadFaissIndex(self, path):
        i = faiss.read_index(path)
        return i
