import math
import numpy as np
from sklearn.cluster import KMeans
import warnings
from collections import Counter
import pandas as pd

warnings.filterwarnings("ignore")  # 处理警告


class GranularBall:
    """class of the granular ball"""

    def __init__(self, data):
        """
        :param data:  Labeled data set, the "-2" column is the class label, the last column is the index of each line
        and each of the preceding columns corresponds to a feature
        """
        self.data = data[:, :]
        self.data_no_label = data[:, :]
        self.num, self.dim = self.data_no_label.shape
        self.center = self.data_no_label.mean(0)
        self.r = self.__get_r()

    def __get_r(self):
        """
        :return: the label and purity of the granular ball.
        """
        arr = np.array(self.data_no_label) - self.center
        ar = np.square(arr)
        a = np.sqrt(np.sum(ar, 1))
        r = np.sum(a) / len(self.data_no_label)
        return r

    def split_2balls(self):
        """
        split the granular ball to 2 new balls by using 2_means.
        """
        # label_cluster = KMeans(X=self.data_no_label, n_clusters=2)[1]
        clu = KMeans(n_clusters=2).fit(self.data_no_label)

        label_cluster = clu.labels_
        if sum(label_cluster == 0) and sum(label_cluster == 1):
            ball1 = GranularBall(self.data[label_cluster == 0, :])
            ball2 = GranularBall(self.data[label_cluster == 1, :])
        else:
            ball1 = GranularBall(self.data[0:1, :])
            ball2 = GranularBall(self.data[1:, :])
        return ball1, ball2


class GBList:
    """class of the list of granular ball"""

    def __init__(self, data=None):
        self.data = data[:, :]
        self.granular_balls = [GranularBall(self.data)]  # gbs is initialized with all data

    def init_granular_balls(self, min_sample=100):
        """
        Split the balls, initialize the balls list.
        :param purity: If the purity of a ball is greater than this value, stop splitting.
        :param min_sample: If the number of samples of a ball is less than this value, stop splitting.最多有多少个number
        """
        ll = len(self.granular_balls)  # 粒球数量.-
        i = 0
        while True:
            if (self.granular_balls[i].num > min_sample) or (
                    len(self.granular_balls) <= 2 and self.granular_balls[i].num >= 2):
                split_balls = self.granular_balls[i].split_2balls()
                self.granular_balls[i] = split_balls[0]
                self.granular_balls.append(split_balls[1])
                ll += 1
            else:
                i += 1
            if i >= ll:
                break
        self.data = self.get_data()

    def get_data_size(self):
        return list(map(lambda x: len(x.data), self.granular_balls))

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_r(self):
        """
        :return: 返回半径r
        """
        return np.array(list(map(lambda x: x.r, self.granular_balls)))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)

    def del_ball(self, num_data=0):  # num_data = delbals,粒球中最少的样本数量
        T_ball = []
        for ball in self.granular_balls:
            if ball.num >= num_data:  # ball.num:粒球中的样本数
                T_ball.append(ball)
        self.granular_balls = T_ball.copy()
        self.data = self.get_data()


def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


def generate_ball_data(data, delbals):  # obs = 邻域
    # num, dim = data[:, :-1].shape
    num = data.shape[0]  # num:样本数，dim:列数
    dim = data.shape[1]
    index = np.array(range(num)).reshape(num, 1)  # column of index 样本索引
    data = np.hstack((data, index))  # Add the index column to the last column of the data
    gb = GBList(data)  # create the list of granular balls
    gb.init_granular_balls(min_sample=10)  # initialize the list
    gb.del_ball(num_data=delbals)
    centers = gb.get_center().tolist()
    rs = gb.get_r().tolist()
    b = []
    # print('gb_num is:', len(gb.granular_balls))
    redata = data[:, :607]  ####################################################### Feature数+1########################
    n, d = redata.shape  # 除去标记之后的行列
    for i in range(len(gb.granular_balls)):  # 在第i个粒度球中
        obs = []
        for j in range(n):  # j:样本数
            if eucliDist(redata[j], centers[i]) <= rs[i]:  # 在同一粒度球中的样本存进obs列表中
                obs.append(j)
        b.append(obs)
    return b


data = pd.read_csv('./data/MLFS/recreationdata.csv')#################################################
delbals = 10  # 粒球中至少有delbals个样本
neighborhood = generate_ball_data(data, delbals)  # neighborhood=obs

  # print

# 计算近似精度：下近似/上近似，和xi标记一致的
# 计算互信息 # U：len(neighborhood)
# 评估特征
# labeldata = pd.read_csv('./data/yeastlabel_1.csv')  # 标记数据 labeldata[0,:]：第一行标记
# labeldata = np.array(labeldata)
# m, l = labeldata.shape  # m:样本数593 l:标记数6


# 将多维列表展平为一维列表,统计元素个数
def flat_list(lst):
    result = []
    def inner(lst):
        for item in lst:
            if isinstance(item, list):
                inner(item)
            else:
                result.append(item)
    inner(lst)
    return result


def evfeature(data):
    # 计算和x标记一致的对象数量?
    # 计算邻域
    global nmi_fr
    S = []
    num, dim = data.shape  # 593 , 72
    zeroarray = np.zeros((5000), dtype=np.int)  # 593行的一列数组      ###########################################样本数############
    zeroseries = pd.Series(zeroarray.tolist())  # 对于单个特征，加一个全0的列  series:单列的dataframe
    t_red = pd.DataFrame()  # 定义一个red矩阵
    for j in range(0, dim):    # 对每个特征
        fdata = data.iloc[:, j]  # series类型的单个特征
        fdatazero = pd.DataFrame(list(zip(fdata, zeroseries)))  # 合并两个series
        neighborhood_f = generate_ball_data(fdatazero, delbals)  # 在每个特征上生成的粒度球邻域 len(neighborhood) = 有多少个邻域类
        t_red = t_red.append(fdata)  # 将f逐个添加到red子集中     # [72 rows x 593 columns] 需要转置
        f_union_red = pd.DataFrame(t_red.values.T, index=t_red.columns, columns=t_red.index)  # [593 rows x 72 columns]
        neighborhood_f_red = generate_ball_data(f_union_red, delbals)  # f和red的并集上生成粒度球邻域
        red = f_union_red
        red = red.drop(red.columns[[-1]], axis=1)
        neighborhood_red = generate_ball_data(red, delbals)  # 在添加f之前的red中生成的邻域
        a = flat_list(neighborhood_f)  # 计算u
        u = len(a)
        for i in neighborhood_f:
            if i in neighborhood_f_red and neighborhood_red:
                neighborhood_f.sort(key=lambda x: x[0] != i)
                neighborhood_f_red.sort(key=lambda x: x[0] != i)
                neighborhood_red.sort(key=lambda x: x[0] != i)
            nmi_fr = -(
                math.log((len(neighborhood_f[0])) * (len(neighborhood_red[0])) / (len(neighborhood_f_red[0])))) / u
        print(nmi_fr)


data = pd.read_csv('./data/MLFS/recreationdata.csv') #################################################
evresult = evfeature(data)
# print(evresult)
