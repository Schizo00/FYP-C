import tensorflow as tf
import numpy as np
from custom_metrics import custom_metric
import open3d as o3d
class WeightedLoss(tf.keras.losses.Loss):
  def __init__(self, base_loss, attention_ids, weight, batch_size, custom_metric_dict=None, reduction=None):
    super(WeightedLoss, self).__init__(name='weighted_loss')
    self.base_loss = base_loss
    self.weight = weight
    self.weights = np.ones((batch_size, 5023, 3))
    self.attention_ids = attention_ids
    for i in range(batch_size):
      for j in attention_ids:
        self.weights[i][j] = (1 * 2) ** weight
    self.weights2 = np.zeros((batch_size, 5023, 3))
    for i in range(batch_size):
      for j in attention_ids:
        self.weights2[i][j] = (1 * 2) ** weight

    # self.weights = self.weights.flatten()
    self.weights = tf.cast(self.weights, dtype=tf.float32)
    self.batch_size = batch_size
    if custom_metric_dict:
      self.custom_metric_dict = custom_metric_dict
      self.custom_metric = custom_metric(custom_metric_dict, batch_size=self.batch_size)

  def call(self, y_true, y_pred):
    y_pred = tf.reshape(y_pred, [self.batch_size, 5023, 3]) * 201.41335 
    y_true = tf.reshape(y_true, [self.batch_size, 5023, 3]) * 201.41335 
    # Calculate standard loss
    loss = self.base_loss(y_true * self.weights, y_pred * self.weights)

    custom_loss = 0

    if self.custom_metric:
       custom_loss = self.custom_metric.__call__(y_true * self.weights2, y_pred * self.weights2)
       
    # Apply weights element-wise
    weighted_loss = loss
    # * self.weights

    # Return the mean weighted loss
    return tf.reduce_mean(weighted_loss) + (custom_loss) 
  # * 2 ** self.weight
    # return(custom_loss * 2) ** self.weight
  
  def get_weights(self):
     return self.weights
  
  def get_config(self):
    """Returns a dictionary containing the configuration of the WeightedLoss object."""
    return {
        'base_loss': self.base_loss, # Store the name of the base loss function
        'attention_ids': self.attention_ids,  # Convert attention_ids to a list
        'weight': self.weight,
        'batch_size': self.batch_size,
        'custom_metric_dict': self.custom_metric_dict
    }
  
  if __name__ == "__main__":
     
    from weighted_mse import WeightedLoss
    
    loss = WeightedLoss(tf.keras.losses.MeanSquaredError(), np.load("./weight_indices.npy"), 10)
    print("In")
    print(loss.get_config())

    pcd1 = o3d.io.read_triangle_mesh("Preprocessed/id00012/mesh.ply")
    pcd1 = np.array(pcd1.vertices)
    pcd1 = pcd1.flatten()

    pcd2 = o3d.io.read_triangle_mesh("Preprocessed/id00015/mesh.ply")
    pcd2 = np.array(pcd2.vertices)
    pcd2 = pcd2.flatten()

    print("Loss: ", loss.__call__(tf.reshape(tf.convert_to_tensor(pcd1), [1,15069]),tf.reshape(tf.convert_to_tensor(pcd2), [1,15069])))

