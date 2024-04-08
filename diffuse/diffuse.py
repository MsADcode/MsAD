from logging import raiseExceptions
import torch
import math
import numpy as np
from functorch import vmap, jacrev, jacfwd
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time
import copy
from diffuse.gaussian_diffusion import GaussianDiffusion, UniformSampler, get_named_beta_schedule, mean_flat, \
    LossType, ModelMeanType, ModelVarType
from diffuse.nn import DiffAtt
from diffuse.utils import full_DAG
import random

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



class Diffuse():
    def __init__(self, dim,val_ratio,epsilon,n_nodes, n_votes, beta_start, beta_end,  orderSize, prunSize, thresh,
                 epochs, batch_size, n_steps,learning_rate: float = 0.001):

        self.val_ratio=val_ratio
        self.epsilon = epsilon
        self.thresh=thresh
        self.orderSize = orderSize
        self.prunSize = prunSize

        self.n_nodes = n_nodes
        assert self.n_nodes > 1, "Not enough nodes, make sure the dataset contain at least 2 variables (columns)."
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Diffusion parameters
        self.n_steps = n_steps
        betas = get_named_beta_schedule(schedule_name="linear", num_diffusion_timesteps=self.n_steps, scale=1,
                                        beta_start=beta_start, beta_end=beta_end)
        self.gaussian_diffusion = GaussianDiffusion(betas=betas,
                                                    loss_type=LossType.MSE,
                                                    model_mean_type=ModelMeanType.EPSILON,  # START_X,EPSILON
                                                    model_var_type=ModelVarType.FIXED_LARGE,
                                                    rescale_timesteps=True,
                                                    )


        self.schedule_sampler = UniformSampler(self.gaussian_diffusion)

        ## Diffusion training
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = DiffAtt(n_nodes,dim).to(self.device)
        self.model.float()
        self.opt = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.val_diffusion_loss = []
        self.best_loss = float("inf")
        self.early_stopping_wait = 300

        ## Topological Ordering
        self.n_votes = n_votes
        self.masking = True
        self.residue = False
        self.sorting = (not self.masking) and (not self.residue)


    def fit1(self, X):
        X = (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True)
        X = torch.FloatTensor(X).to(self.device)
        self.train_score(X)

    def fit2(self, prunX):
        prunX = (prunX - prunX.mean(0, keepdims=True)) / prunX.std(0, keepdims=True)
        self.corrcoef = np.corrcoef(prunX.transpose(1, 0))
        prunX = torch.FloatTensor(prunX).to(self.device)
        self.active_nodes = list(range(self.n_nodes))

        order = self.topological_ordering(prunX)
        dag_order=np.array(full_DAG(order))

        out_dag = self.pruning_by_coef_2nd(dag_order, prunX.detach().cpu().numpy())

        return out_dag

    def fit3(self, prunX):
        prunX = (prunX - prunX.mean(0, keepdims=True)) / prunX.std(0, keepdims=True)
        prunX = torch.FloatTensor(prunX).to(self.device)
        self.active_nodes = list(range(self.n_nodes))

        order = self.topological_ordering(prunX)

        return order


    def format_seconds(self,seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


    def pruning_by_coef_2nd(self, graph_batch, X) -> np.ndarray:
        """
        for a given graph, pruning the edge according to edge weights;
        quadratic regression for each causal regression for edge weights and then
        thresholding
        对于给定的图，根据边的权值对边进行修剪，对每条边进行二次回归，对边的权值进行因果回归，然后进行阈值分割
        """

        start_time = time.time()
        thresh = self.thresh
        d = graph_batch.shape[0]
        reg = LinearRegression()

        poly = PolynomialFeatures()
        W = []
        W_continuous = []

        pbar = tqdm(range(d), desc="Pruning_by_coef_2nd")
        for i in pbar:
            col = graph_batch[i] > 0.1

            if np.sum(col) <= 0.1:
                W.append(np.zeros(d))
                W_continuous.append(np.zeros(d))
                continue

            X_train = X[:, col]
            X_train_expand = poly.fit_transform(X_train)[:, 1:]
            X_train_expand_names = poly.get_feature_names()[1:]

            y = X[:, i]
            reg.fit(X_train_expand, y)
            reg_coeff = reg.coef_
            reg_coeff = torch.from_numpy(reg_coeff)
            # reg_coeff = torch.softmax(reg_coeff, dim=-1)
            cj = 0
            new_reg_coeff = np.zeros(d, )
            new_reg_coeff_continuous = np.zeros(d, )
            for ci in range(d):
                if col[ci]:
                    xxi = 'x{}'.format(cj)
                    for iii, xxx in enumerate(X_train_expand_names):
                        if xxi in xxx:

                            if reg_coeff[iii] > thresh:
                                new_reg_coeff[ci] = 1
                                new_reg_coeff_continuous[ci] = reg_coeff[iii]

                            break
                    cj += 1
            W.append(new_reg_coeff)
            W_continuous.append(new_reg_coeff_continuous)

        end_time = time.time()
        total_elapsed_time = end_time - start_time
        formatted_time = self.format_seconds(total_elapsed_time)
        print("prun_time: ", formatted_time)

        return np.array(W)



    def train_score(self, X, fixed=None):
        start_time = time.time()
        if fixed is not None:
            self.epochs = fixed
        best_model_state_epoch = 300
        self.model.train()
        n_samples = X.shape[0]
        self.batch_size = min(n_samples, self.batch_size)
        val_ratio = self.val_ratio
        val_size = int(n_samples * val_ratio)
        train_size = n_samples - val_size
        X = X.to(self.device)
        X_train, X_val = X[:train_size], X[train_size:]
        data_loader_val = torch.utils.data.DataLoader(X_val, min(val_size, self.batch_size))
        data_loader = torch.utils.data.DataLoader(X_train, min(train_size, self.batch_size), drop_last=True)
        pbar = tqdm(range(self.epochs), desc="Training Epoch")
        for epoch in pbar:
            loss_per_step = []
            for steps, x_start in enumerate(data_loader):
                # apply noising and masking
                x_start = x_start.float().to(self.device)
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)

                noise = torch.randn_like(x_start).to(self.device)
                x_t = self.gaussian_diffusion.q_sample(x_start, t, noise=noise)
                # x_t  是对  x_start  施加偏移和噪声后的结果

                # get loss function
                model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t))
                diffusion_losses = (noise - model_output) ** 2
                diffusion_loss = (
                            diffusion_losses.mean(dim=list(range(1, len(diffusion_losses.shape)))) * weights).mean()
                loss_per_step.append(diffusion_loss.item())
                self.opt.zero_grad()
                diffusion_loss.backward()
                self.opt.step()
            if fixed is None:
                if epoch % 10 == 0 and epoch > best_model_state_epoch:
                    with torch.no_grad():
                        loss_per_step_val = []
                        for steps, x_start in enumerate(data_loader_val):
                            t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                            noise = torch.randn_like(x_start).to(self.device)
                            x_t = self.gaussian_diffusion.q_sample(x_start, t, noise=noise)
                            model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t))
                            diffusion_losses = (noise - model_output) ** 2
                            diffusion_loss = (diffusion_losses.mean(
                                dim=list(range(1, len(diffusion_losses.shape)))) * weights).mean()
                            loss_per_step_val.append(diffusion_loss.item())
                        epoch_val_loss = np.mean(loss_per_step_val)

                        if self.best_loss > epoch_val_loss:
                            self.best_loss = epoch_val_loss
                            best_model_state = deepcopy(self.model.state_dict())
                            best_model_state_epoch = epoch
                    pbar.set_postfix({'Epoch Loss': epoch_val_loss})

                if epoch - best_model_state_epoch > self.early_stopping_wait:  # Early stopping
                    break

        if fixed is None:
            print(f"Early stoping at epoch {epoch}")
            print(f"Best model at epoch {best_model_state_epoch} with loss {self.best_loss}")
            self.model.load_state_dict(best_model_state)



        end_time = time.time()
        total_elapsed_time = end_time - start_time
        formatted_time = self.format_seconds(total_elapsed_time)
        print("train_time: ",formatted_time)



    def topological_ordering(self, X, step=None):
        start_time = time.time()
        X = X[:self.orderSize]

        self.model.eval()
        order = []

        active_nodes = self.active_nodes
        # 活动节点active_nodes

        steps_list = [step] if step is not None else range(0, self.n_steps + 1, self.n_steps // self.n_votes)

        if self.sorting:
            steps_list = [self.n_steps // 2]
        pbar = tqdm(range(len(self.active_nodes) - 1), desc="Nodes ordered ")
        # 使用tqdm库中的range函数来创建迭代器，在每次迭代时会更新进度条，显示描述为"Nodes ordered "的进度条
        leaf = None

        # 遍历n-1个结点:
        for jac_step in pbar:

            leaves = []
            #每一次迭代，step不同（扩散程度系数）：

            for i, steps in enumerate(steps_list):

                model_fn_functorch = self.get_model_function_with_residue(steps, active_nodes, order)

                leaf_ = self.compute_jacobian_and_get_leaf(X, active_nodes, model_fn_functorch)

                if self.sorting:
                    order = leaf_.tolist()
                    order.reverse()
                    return order
                leaves.append(leaf_)

            leaf = Counter(leaves).most_common(1)[0][0]

            leaf_global = active_nodes[leaf]  # leaf_global表示该叶节点在active_nodes中的索引
            # 将其添加到order列表中，并从active_nodes中移除:
            order.append(leaf_global)
            active_nodes.pop(leaf)

        order.append(active_nodes[0])  # 最后一个节点直接放进order列表的末尾
        order.reverse()  # 将order列表反转并返回

        end_time = time.time()
        total_elapsed_time = end_time - start_time
        formatted_time = self.format_seconds(total_elapsed_time)
        print("topological_time: ", formatted_time)

        return order

    # 生成一个带有残差项的模型函数，用于计算结点分数
    def get_model_function_with_residue(self, step, active_nodes, order):
        t_functorch = (torch.ones(1) * step).long().to(
            self.device)  # test if other ts or random ts are better, self.n_steps
        get_score_active = lambda x: self.model(x, self.gaussian_diffusion._scale_timesteps(t_functorch))[:,
                                     active_nodes]
        get_score_previous_leaves = lambda x: self.model(x, self.gaussian_diffusion._scale_timesteps(t_functorch))[:,
                                              order]

        def model_fn_functorch(X):
            score_active = get_score_active(X).squeeze()  # 当前活动节点的得分

            if self.residue and len(order) > 0:
                score_previous_leaves = get_score_previous_leaves(X).squeeze()
                jacobian_ = jacfwd(get_score_previous_leaves)(X).squeeze()
                if len(order) == 1:
                    jacobian_, score_previous_leaves = jacobian_.unsqueeze(0), score_previous_leaves.unsqueeze(0)
                score_active += torch.einsum("i,ij -> j", score_previous_leaves / jacobian_[:, order].diag(),
                                             jacobian_[:, active_nodes])  #

            return score_active

        return model_fn_functorch


    def get_masked(self, x, active_nodes):
        dropout_mask = torch.zeros_like(x).to(self.device)
        dropout_mask[:, active_nodes] = 1

        return (x * dropout_mask).float()


    def compute_jacobian_and_get_leaf(self, X, active_nodes, model_fn_functorch):

        with torch.no_grad():

            epsilon = self.epsilon

            x_batch_dropped = self.get_masked(X, active_nodes) if self.masking else X  # 获取掩码下batch数据

            outputs = vmap(model_fn_functorch)(x_batch_dropped.unsqueeze(1))  # 计算原始输入的输出

            i2 = 0
            jacobian = torch.zeros(len(active_nodes), len(active_nodes)).float().to(self.device)

            for i1 in range(self.n_nodes):

                if float(x_batch_dropped[0, i1]) != 0:
                    perturbed_inputs = x_batch_dropped.clone()
                    perturbed_inputs[:, i1] += epsilon  # 对第i个node维度进行扰动

                    perturbed_outputs = vmap(model_fn_functorch)(perturbed_inputs.unsqueeze(1))  # 得到扰动后的输出

                    jac = (perturbed_outputs - outputs)[:, i2] / epsilon  # 计算偏导数值
                    variance = torch.var(jac).item()

                    jacobian[i2, i2] = variance
                    i2 += 1

            leaf = self.get_leaf(jacobian)

        return leaf

    def get_leaf(self, jacobian_active):

        jacobian_active = jacobian_active.cpu().numpy()
        jacobian_var_diag = jacobian_active.diagonal()  # 获取 jacobian_var 方差矩阵的对角线元素,表示
        # 在第 i 个维度上所有样本的方差
        var_sorted_nodes = np.argsort(jacobian_var_diag)
        if self.sorting:
            return var_sorted_nodes
        leaf_current = var_sorted_nodes[0]
        return leaf_current