import numpy as np

class LogisticRegression(object):
    def __init__(self, dim, learning_rate=0.01, max_iter=1000, seed=None):
        np.random.seed(seed)
        self.lr = learning_rate
        self.max_iter = max_iter  # 定义学习率和训练轮数
        self.W = np.random.normal(1, 0.1, [dim+1, 1]) # w 的维度为输入维度+1
        # 可在此处补充类的属性
    
    def fit(self,X,Y):
        # 请在此处补充类的方法：训练函数，返回每轮loss的列表
        loss_arr = []
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        for i in range(self.max_iter):
            self.__train_step(X, Y)
            Y_pred, Y_pred_label = self.predict(X[:,:-1])
            loss_arr.append(self.loss(Y, Y_pred))
        return loss_arr
    
    def __calc_gradient(self, X, Y):
        N = X.shape[0]
        p = X.dot(self.W)
        p = 1 / (1 + np.exp(-p))
        grad = X.T.dot(p- Y)
        grad = grad / N
        return grad
    
        
    
    def __train_step(self, X, Y):
        d_w = self.__calc_gradient(X, Y)
        self.W = self.W - self.lr * d_w
        return self.W
        
    def __f(self, x, w):
        return 1 / (1 + np.exp(-x.dot(w)))
    
    def loss(self, Y, Y_pred):
        N = Y.shape[0]
        return -np.mean(Y * np.log(Y_pred) + (1 - Y) * np.log(1-Y_pred))
    
    def predict(self,X): 
        # 请在此处补充类的方法：测试函数，返回对应X的预测值和预测列表号    
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        Y_pred = self.__f(X, self.W)
        
        Y_pred_label = []
        for i in range(Y_pred.shape[0]):
            if Y_pred[i] > 0.5:
                Y_pred_label += [1]
            else:
                Y_pred_label += [0]
        Y_pred_label = np.array(Y_pred_label)
        
        
        return Y_pred, Y_pred_label
   
  
  
  
  
