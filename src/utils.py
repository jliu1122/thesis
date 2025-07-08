from pycocotools.coco import COCO
import glob

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
        return list(glob.glob(path + "/.jpg"))

    def idToFile(self, id):
        img = self.instance.loadImgs(id)[0]
        return f"coco/images/val2017/{img['file_name']}"
