import tensorflow as tf
import numpy as np


class custom_metric():
    def __init__(self, id_dict, batch_size):
        self.id_dict = id_dict
        self.keys = []
        self.batch_size = batch_size
        for key in id_dict.keys():
            self.key = id_dict[key]
            self.keys.append(self.key)

    def __call__(self, y_test, y_pred):

        # y_pred = tf.reshape(y_pred, [self.batch_size, 5023, 3])
        # y_test = tf.reshape(y_test, [self.batch_size, 5023, 3])
        loss = self.directed_hausdorff(y_test, y_pred)


        return loss
    

    def directed_hausdorff(self, point_cloud_A, point_cloud_B):
        '''
        input:
            point_cloud_A: Tensor, B x N x 3
            point_cloud_B: Tensor, B x N x 3
        return:
            Tensor, B, directed hausdorff distance, A -> B
        '''
        npoint = point_cloud_A.shape[1]

        A = tf.expand_dims(point_cloud_A, axis=2) # (B, N, 1, 3)
        A = tf.tile(A, (1, 1, npoint, 1)) # (B, N, N, 3)

        B = tf.expand_dims(point_cloud_B, axis=1) # (B, 1, N, 3)
        B = tf.tile(B, (1, npoint, 1, 1)) # (B, N, N, 3)

        distances = tf.math.squared_difference(B, A) # (B, N, N, 3)
        # print(distances.shape)
        # print(distances)
        distances = tf.reduce_sum(distances, axis=-1) # (B, N, N, 1)
        distances = tf.sqrt(distances) # (B, N, N)

        shortest_dists, _ = tf.nn.top_k(-distances)
        shortest_dists = tf.squeeze(-shortest_dists) # (B, N)

        hausdorff_dists, _ = tf.nn.top_k(shortest_dists) # (B, 1)
        hausdorff_dists = tf.squeeze(hausdorff_dists)

        return hausdorff_dists ** 2
    

    def distances(self, y_test, y_pred):
        tot = []
        for j in range(self.batch_size):
            diff_tot = []
            diff0 = []
            diff1 = []
            x_tot = []
            y_tot = []
            z_tot = []

            points0 = tf.norm(y_test[j, :] - y_pred[j, :], ord='euclidean', axis=1) ** 2
            points1 = tf.norm(y_test[j, :] - y_pred[j, :], ord='euclidean', axis=1) ** 2
            # ----------------
            for i in self.keys:
                
                diff = (tf.norm(y_pred[j, i[0]] - y_pred[j, i[1]], ord='euclidean') - tf.norm(y_test[j, i[0]] - y_test[j, i[1]], ord='euclidean')) ** 2
                diff_tot.append(diff)

                # point0diff = tf.norm(y_test[j, i[0]] - y_pred[j, i[0]], ord='euclidean') ** 2
                # point1diff = tf.norm(y_test[j, i[1]] - y_pred[j, i[1]], ord='euclidean') ** 2

                # diff0.append(point0diff)
                # diff1.append(point1diff)

                diff0.append(points0[i[0]])
                diff1.append(points1[i[0]])
            # --------------
                # x = ((y_test[j, (i[0] * 3) + 0]) - (y_pred[j, (i[1] * 3) + 0]))**2
                # y = ((y_test[j, (i[0] * 3) + 1]) - (y_pred[j, (i[1] * 3) + 1]))**2
                # z = ((y_test[j, (i[0] * 3) + 2]) - (y_pred[j, (i[1] * 3) + 2]))**2

                # x_tot.append(x)
                # y_tot.append(y)
                # z_tot.append(z)

            # tot.append(tf.reduce_mean(x_tot) + tf.reduce_mean(y_tot) + tf.reduce_mean(z_tot))
            tot.append(tf.reduce_mean(diff_tot) 
                       + tf.reduce_mean(diff0)
                        + tf.reduce_mean(diff1))
            
        tf.reduce_mean(tot)

    # def __call__(self, y_test, y_pred):
    #     batch_size = 32
    #     num_points = 15069
    #     y_pred = tf.reshape(y_pred, [batch_size, num_points])
    #     y_test = tf.reshape(y_test, [batch_size, num_points])
        
    #     # Pre-calculate indices for faster indexing
    #     indices = [[i[0] * 3, i[1] * 3] for i in self.keys]
        
    #     print()
        

        
    #     return 10
    

if __name__ == "__main__":
    import open3d as o3d
    import numpy as np
    from custom_metrics import custom_metric
    import pickle

    id_dict = {
        'nose_bridge' : [3516, 3526],
        'r_eye_lid' : [3690, 2265],
        'l_eye_lid' : [3856, 809],
        'lip' : [3543, 3503],
        'r_lip_bend' : [2850, 3798],
        'l_lip_bend' : [1735, 3021],
        'r_lip_jaw' : [3798, 3406],
        'l_lip_jaw' : [3021, 3614],
        'lip_chin' : [3503, 3487],
        'orbital_lower' : [3710, 3866],
        'oribtal_upper' : [3154, 2135],
        'puffer' : [3436, 3667],
        'mouth_corner' : [2827, 1710],
        'jaw_end' : [3406, 3614],
        'ear_end' : [856, 288]
    }
    loss = custom_metric(id_dict=id_dict)

    pcd1 = o3d.io.read_triangle_mesh("Preprocessed/id00012/mesh.ply")
    pcd1 = np.array(pcd1.vertices)
    pcd1 = pcd1.flatten()

    pcd2 = o3d.io.read_triangle_mesh("Preprocessed/id00015/mesh.ply")
    pcd2 = np.array(pcd2.vertices)
    pcd2 = pcd2.flatten()

    print("Loss: ", loss.__call__(tf.reshape(tf.convert_to_tensor(pcd1), [1,15069]),tf.reshape(tf.convert_to_tensor(pcd2), [1,15069])))

    with open('custom_loss_dict.pkl', 'wb') as file:
        pickle.dump(id_dict, file)
