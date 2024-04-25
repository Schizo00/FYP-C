import numpy as np
import pandas as pd
import open3d as o3d
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class VoxDataset(tf.keras.utils.Sequence):
    def __init__(self, emb_max=113.46356, mesh_max=201.41335, transform=None, limit=None, indices=[], ids=[], gender=None):
        self.emb_max = emb_max
        self.mesh_max = mesh_max
        self.gender = gender
        self.annotations = pd.read_csv(f'./output.csv')
        if limit != None:
            self.annotations = self.annotations.sample(limit).reset_index()
        if gender:
            self.annotations = self.annotations[self.annotations['gender'] == self.gender]
        if len(indices) != 0:
            if type(indices) is list:
                self.annotations = self.annotations.iloc[indices].reset_index()
        if len(ids) != 0:
            if type(indices) is list:
                self.annotations = self.annotations[self.annotations['id'].isin(ids)].reset_index()


    def get_df_(self):
        return self.annotations

    def __getitem__(self, index):
        embedding = self._get_embedding(index)
        mesh = self._get_mesh(index)
        return tf.cast((embedding / self.emb_max) , tf.float32), tf.cast((mesh / self.mesh_max) , tf.float32)

    def __len__(self):
        return len(self.annotations) 

    def _get_embedding(self, index):
        embedding_path = self.annotations.loc[index, 'emb_dir']
        embedding = np.load(embedding_path)
        embedding = embedding.flatten()
        return embedding

    def _get_mesh(self, index):
        mesh_path = self.annotations.loc[index, 'mesh_dir']
        pcd = o3d.io.read_triangle_mesh(mesh_path)
        pcd = np.array(pcd.vertices)
        pcd = pcd.flatten()
        return pcd


    def get_ids(self):
        print(len(self.annotations))
        return list(set(self.annotations['id']))
    

    @staticmethod
    def to_mesh_points(points):
        points = points.reshape(5023, 3)
        return points
    
    def get_faces(self):
        mesh_path = self.annotations.loc[1, 'mesh_dir']
        pcd = o3d.io.read_triangle_mesh(mesh_path)
        pcd = np.array(pcd.triangles)
        return pcd

class DataGenerator(Sequence):

    def __init__(self, dataset, batch_size=8):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.dataset) // self.batch_size) + (len(self.dataset) % self.batch_size > 0)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        inputs, outputs = zip(*[self.dataset[i] for i in range(start_idx, end_idx)])

        return np.array(inputs), np.array(outputs)


if __name__ == "__main__":

    dataset = VoxDataset("./annotations.csv")

    print(len(dataset[0][0]), len(dataset[0][1]))
    print(len(dataset[0]))