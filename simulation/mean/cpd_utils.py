import numpy as np
import scipy.stats as stats
import random

import itertools

import bisect

from sklearn import linear_model



###### EDA ######


class cpd_eda_linear():
    def __init__(self):
        pass
    
    def likelihood_path(self, Y, X, sm, em, smooth_s = 100, smooth_e = 0, step = 1, lam = 0.1):
        """
        smooth parameter > 0
        """
        Y = Y.reshape((-1,))
        Y = Y[int(sm):int(em) + 1]
        X = X[int(sm):int(em) + 1]
        nt, p = X.shape

        if nt < (smooth_s + smooth_e):
            return None, None, None

        n = int((nt - smooth_s - smooth_e) // step) + 1
        beta_hat = np.zeros((n, p))
        loglike = np.zeros(n)
        sample_size = np.array([smooth_s + i * step for i in range(n)])

        solver = linear_model.Lasso(fit_intercept = False, alpha = lam, random_state = 0, warm_start = True)
        for i in range(n):
            loc = sample_size[i]
            X_cur, Y_cur = X[:loc], Y[:loc]
            solver.fit(X_cur, Y_cur)
            Y_cur_pred = solver.predict(X_cur)
            
            loglike[i] = -np.sum((Y_cur - Y_cur_pred)**2)
            beta_hat[i] = solver.coef_.squeeze().copy()

        return beta_hat, loglike, sample_size

    def glr_path(self, Y, X, sm, em, smooth = 100, step = 1, lam = 0.1):
        """
        Y: T x -1 array
        X: T x p array

        """
        Y = Y.reshape((-1,))
        Y = Y[int(sm):int(em) + 1]
        X = X[int(sm):int(em) + 1]
        nt = len(Y)

        beta_left, loglike_left, sample_size_left = self.likelihood_path(Y, X, 0, nt - 1, smooth, smooth, step, lam)
        resid = nt - sample_size_left[-1]
        beta_right, loglike_right, sample_size_right = self.likelihood_path(Y[::-1], X[::-1], 0, nt-1, resid, smooth, step, lam)

        return (beta_left, loglike_left, sample_size_left), (beta_right[::-1], loglike_right[::-1], sample_size_right[::-1])

    def likelihood_interval(self, Y, X, s, e, lam = 0.1):
        Y = Y.reshape((-1,))
        nt, p = X.shape
        solver = linear_model.Lasso(fit_intercept = False, alpha = lam, random_state = 0, warm_start = True)
        solver.fit(X, Y)
        Y_pred = solver.predict(X)
        loglike = -np.sum((Y - Y_pred)**2)
        
        beta_hat = solver.coef_.squeeze()
        return beta_hat, loglike

    def draw_glr_path(self, X_train, Y_train, smooth, step):
        T = len(Y_train)
        lam = 0.1

        l_left, l_right = self.glr_path(Y_train, X_train, 0, T - 1, smooth = smooth, step = step, lam = lam)

        beta_joint, loglike_joint = self.likelihood_interval(Y_train, X_train, 0, T - 1, lam = lam)

        beta_left, loglike_left, sample_size_left = l_left
        beta_right, loglike_right, sample_size_right = l_right

        loglike_split = loglike_left + loglike_right

        return sample_size_left, loglike_split - loglike_joint
    






#######################################
#### functions for general purpose
#######################################

def cp_distance(cp1, cp2):
    """
    Hausdoff distance between two sets of change points
    """
    cp1.sort()
    cp2.sort()
    cp2_ = np.concatenate([[-np.infty], cp2, [np.infty]])
    mx = -np.infty
    for x in cp1:
        y = bisect.bisect_left(cp2_, x)
        mx = max(mx, min(abs(cp2_[y - 1] - x), abs(cp2_[y] - x)))
    cp1_ = np.concatenate([[-np.infty], cp1, [np.infty]])
    for x in cp2:
        y = bisect.bisect_left(cp1_, x)
        mx = max(mx, min(abs(cp1_[y - 1] - x), abs(cp1_[y] - x)))
    return mx



########## dcdp related functions ##########


#### general class ####

class dp_cv_grid():
    def __init__(self, grid_n, lam_list, gamma_list, smooth = 10, buffer = 10):
        self.grid_n = grid_n
        self.lam_list = lam_list
        self.gamma_list = gamma_list
        self.smooth = smooth
        self.buffer = buffer
        
    def estimate_func(self, Y, lam):
        pass

    def loss_func(self, Y, beta):
        pass
 
    def goodness_of_fit(self, Y_train, Y_test, cp_loc, lam, gamma):
        '''
        currently the penalty is not included in the value of goodness-of-fit
        '''        
        n = Y_train.shape[0]
        cp_loc = np.concatenate([[0], cp_loc, [n]])
        cp_loc = cp_loc.astype(int)
        res = 0
        for i in range(len(cp_loc) - 1):
            if cp_loc[i + 1] > cp_loc[i] + 1:
                Y_sub = Y_train[cp_loc[i]:cp_loc[i + 1]]
                estimate = self.estimate_func(Y_sub, lam)
                
                Y_sub = Y_test[cp_loc[i]:cp_loc[i + 1]]
                res += self.loss_func(Y_sub, estimate)
            else:
                res += 0
        return res
        
    def fit(self, Y_train, Y_test):
        ## reshape Y to (n,p) array if necessary
        if len(Y_train.shape) == 1:
            Y_train = Y_train.reshape((-1, 1))
        if len(Y_test.shape) == 1:
            Y_test = Y_test.reshape((-1, 1))
        
        lam_gamma_list = list(itertools.product(self.lam_list, self.gamma_list))
        fit_res_best = np.infty
        cp_best = None
        param_best = None

        n = Y_train.shape[0]
        grid_n = min(self.grid_n, n - 1)
        # step = max(n // self.grid_n, 1)
        # grid = np.arange(0, n, step)
        step = n / (grid_n + 1)
        grid = np.floor(np.arange(1, grid_n + 1) * step).astype(int)

        for i, x in enumerate(lam_gamma_list):
            lam, gamma = x
            cp_loc, obj = self.dp_grid(Y_train, grid, lam, gamma)
            fit_res = self.goodness_of_fit(Y_train, Y_test, cp_loc, lam, gamma)
            if fit_res < fit_res_best:
                cp_best = cp_loc
                param_best = x
                fit_res_best = fit_res
        return cp_best, param_best


    def dp_grid(self, Y, grid, lam, gamma):
        """
        Y: n x p rray, p can be -1
        cp_loc: for each ix in cp_loc, it means theta[ix - 1] != theta[ix]
        """
        n = Y.shape[0]
        point_map = np.ones(n + 1) * -1
        B = np.ones(n + 1) * np.infty
        B[0] = gamma

        r_cand = np.unique(np.concatenate([[0], grid, [n]]))
        grid_ix = {x:i for i, x in enumerate(r_cand)}
        l_cand = np.unique(np.concatenate([[0], grid]))
        m = len(grid)

        for r in r_cand:
            for l in l_cand:
                if l >= r or r - l < self.buffer:
                    break
                Y_sub = Y[l:r]
                estimate = self.estimate_func(Y_sub, lam)
        
                b = B[l] + gamma + self.loss_func(Y_sub, estimate)
                if b < B[r]:
                    B[r] = b
                    point_map[r] = l

        k = n
        cp_loc, cp_val = set([]), []
        while k > 0:
            h = int(point_map[k])
            if h > 0:
                cp_loc.add(h)
            k = h
        cp_loc = np.array(sorted(cp_loc))
        return cp_loc, B[-1]

    def dp_grid_test(self, Y, grid, lam, gamma):
        """
        Y: n x p array, p can be -1
        cp_loc: for each ix in cp_loc, it means theta[ix - 1] != theta[ix]
        """
        n = Y.shape[0]
        point_map = np.ones(n + 1) * -1
        B = np.ones(n + 1) * np.infty
        B[0] = gamma

        r_cand = np.concatenate([[0], grid, [n]])
        grid_ix = {x:i for i, x in enumerate(r_cand)}
        l_cand = np.concatenate([[0], grid])
        m = len(grid)
        dp = [[np.infty] * (m + 2) for r in range(m + 1)]

        for r in r_cand:
            for l in l_cand:
                if l >= r:
                    break
                Y_sub = Y[l:r]
                estimate = self.estimate_func(Y_sub, lam)
                dp[grid_ix[l]][grid_ix[r]] = self.loss_func(Y_sub, estimate)

        for r in r_cand:
            for l in l_cand:
                if l >= r:
                    break
                b = B[l] + gamma + dp[grid_ix[l]][grid_ix[r]]
                if b < B[r]:
                    B[r] = b
                    point_map[r] = l

        k = n
        cp_loc, cp_val = set([]), []
        while k > 0:
            h = int(point_map[k])
            if h > 0:
                cp_loc.add(h)
            k = h
        cp_loc = np.array(sorted(cp_loc))
        return cp_loc, B, dp   
    
    
    
class dcdp_cv_grid(dp_cv_grid):
    '''
    The input data Y_train, Y_test, Y should be of shape (n, p)
    '''
    def __init__(self, grid_n, lam_list, gamma_list, smooth = 10, 
                 buffer = 10, step_refine = 1, buffer_refine = 10, lam_refine = 0.1):
        self.grid_n = grid_n
        self.lam_list = lam_list
        self.gamma_list = gamma_list
        
        self.smooth = smooth
        self.buffer = buffer
        
        self.step_refine = step_refine
        self.buffer_refine = buffer_refine
        self.lam_refine = lam_refine

    def joint_estimate_func(self, Y1, Y2, lam):
        pass
        
    def estimate_func(self, Y, lam):
        pass
    
    def loss_func(self, Y, beta):
        pass
 
    def goodness_of_fit(self, Y_train, Y_test, cp_loc, lam, gamma):
        n = Y_train.shape[0]
        cp_loc = np.concatenate([[0], cp_loc, [n]])
        cp_loc = cp_loc.astype(int)
        res = 0
        for i in range(len(cp_loc) - 1):
            if cp_loc[i + 1] > cp_loc[i] + 1:
                Y_sub = Y_train[cp_loc[i]:cp_loc[i + 1]]
                estimate = self.estimate_func(Y_sub, lam)
                
                Y_sub = Y_test[cp_loc[i]:cp_loc[i + 1]]
                res += self.loss_func(Y_sub, estimate)
            else:
                Y_sub = Y_test[cp_loc[i]:cp_loc[i + 1]]
                res += np.sum(Y_sub**2)
        return res
        
    def fit(self, Y_train, Y_test):
        lam_gamma_list = list(itertools.product(self.lam_list, self.gamma_list))
        fit_res_best = np.infty
        cp_best = None
        param_best = None

        n = Y_train.shape[0]
        grid_n = min(self.grid_n, n - 1)
        # step = max(n // self.grid_n, 1)
        # grid = np.arange(0, n, step)
        step = n / (grid_n + 1)
        grid = np.floor(np.arange(1, grid_n + 1) * step).astype(int)

        for i, x in enumerate(lam_gamma_list):
            lam, gamma = x
            cp_loc, obj = self.dp_grid(Y_train, grid, lam, gamma)
            cp_loc_refined = self.local_refine(Y_train, cp_loc)
            fit_res = self.goodness_of_fit(Y_train, Y_test, cp_loc_refined, lam, gamma)
            if fit_res < fit_res_best:
                cp_best = cp_loc_refined
                cp_best_cand = cp_loc
                param_best = x
                fit_res_best = fit_res
        return cp_best, param_best, cp_best_cand

    def local_refine(self, Y, cp_cand):
        n = Y.shape[0]
        K = len(cp_cand)
        cp_cand = np.concatenate([[0], cp_cand[:], [n]])
        cp_loc = []
        for i in range(1, K + 1):
            if cp_cand[i - 1] + 2 * self.buffer_refine > cp_cand[i]:
                continue
            st, ed = int(2/3 * cp_cand[i - 1] + 1/3 * cp_cand[i]), int(2/3 * cp_cand[i + 1] + 1/3 * cp_cand[i])
            cp = self.screen_cp(Y, st, ed)
            if cp is not None:
                cp_loc.append(cp)
        return cp_loc

    def screen_cp(self, Y, st, ed):
        '''
        currently not using group lasso in refinement
        '''
        buffer_refine = self.buffer_refine
        step_refine = self.step_refine
        
        if st + 2 * buffer_refine + step_refine >= ed:
            return None
        n_step = (ed - st - 2 * buffer_refine - 1) // step_refine + 1

        loss_min = np.infty
        loc_min = None
        for i in range(1, n_step):
            loc = st + buffer_refine + int(i * step_refine)
            theta_1, theta_2 = self.joint_estimate_func(Y[st:loc],
                                                        Y[loc:ed], self.lam_refine)
            loss_split = self.loss_func(Y[st:loc], theta_1)
            loss_split += self.loss_func(Y[loc:ed], theta_2)
            if loss_split < loss_min:
                loss_min = loss_split
                loc_min = st + buffer_refine + int(i * step_refine)
        return loc_min
    
    def local_refine_test(self, Y, cp_cand):
        n = Y.shape[0]
        K = len(cp_cand)
        cp_cand = np.concatenate([[0], cp_cand[:], [n]])
        cp_loc = []
        for i in range(1, K + 1):
            if cp_cand[i - 1] + 2 * self.buffer_refine > cp_cand[i]:
                continue
            st, ed = int(2/3 * cp_cand[i - 1] + 1/3 * cp_cand[i]), int(2/3 * cp_cand[i + 1] + 1/3 * cp_cand[i])
            cp = self.screen_cp(Y, st, ed)
            if cp is not None:
                cp_loc.append(cp)
        return cp_loc

    def screen_cp_test(self, Y, st, ed):
        buffer_refine = self.buffer_refine
        step_refine = self.step_refine
        
        if st + 2 * buffer_refine + step_refine >= ed:
            return None
        n_step = (ed - st - 2 * buffer_refine - 1) // step_refine + 1

        loss_min = np.infty
        loc_min = None
        loss_list = []
        loc_list = []
        
        for i in range(1, n_step):
            loc = st + buffer_refine + int(i * step_refine)
            theta_1, theta_2 = self.joint_estimate_func(Y[st:loc],
                                                        Y[loc:ed], self.lam_refine)
            loss_split = self.loss_func(Y[st:loc], theta_1)
            loss_split += self.loss_func(Y[loc:ed], theta_2)
            
            loss_list.append(loss_split)
            loc_list.append(loc)
            
            if loss_split < loss_min:
                loss_min = loss_split
                loc_min = st + buffer_refine + int(i * step_refine)
        return loc_min, loss_list, loc_list



class dp_cv_grid_mean(dp_cv_grid):
    def __init__(self, grid_n, lam_list, gamma_list, smooth = 100, buffer = 200):
        self.grid_n = grid_n
        self.lam_list = lam_list
        self.gamma_list = gamma_list
        self.smooth = smooth
        self.buffer = buffer
    
    def estimate_func(self, Y, lam):
        mu = np.mean(Y, axis = 0)
        return np.sign(mu) * np.maximum(np.abs(mu) - lam, 0)

    def loss_func(self, Y, beta):
        '''
        currently the penalty is not included in the value of loss
        '''
        return np.sum((Y - beta)**2)


class dcdp_cv_grid_mean(dcdp_cv_grid):
    def __init__(self, grid_n, lam_list, gamma_list, smooth = 10, 
                 buffer = 10, step_refine = 1, buffer_refine = 10, lam_refine = 0.1):
        self.grid_n = grid_n
        self.lam_list = lam_list
        self.gamma_list = gamma_list
        
        self.smooth = smooth
        self.buffer = buffer
        
        self.step_refine = step_refine
        self.buffer_refine = buffer_refine
        self.lam_refine = lam_refine

    def joint_estimate_func(self, Y1, Y2, lam):
        """
        may also try group lasso
        """
        mu1 = np.mean(Y1, axis = 0)
        beta_hat1 = np.sign(mu1) * np.maximum(np.abs(mu1) - lam, 0)
        
        mu2 = np.mean(Y2, axis = 0)
        beta_hat2 = np.sign(mu2) * np.maximum(np.abs(mu2) - lam, 0)
        return beta_hat1, beta_hat2
        
    def estimate_func(self, Y, lam):
        mu = np.mean(Y, axis = 0)
        return np.sign(mu) * np.maximum(np.abs(mu) - lam, 0)

    def loss_func(self, Y, beta):
        '''
        currently the penalty is not included in the value of loss
        '''
        return np.sum((Y - beta)**2)
    
    def fit_with_cp(self, Y, cp, lam = 0.1):
        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))
        K = len(cp) - 1
        p = Y.shape[1]
        beta_path = np.zeros((K, p))

        for i in range(K):
            Y_cur = Y[cp[i]:cp[i + 1]]
            beta_cur = self.estimate_func(Y_cur, lam).reshape((-1,))
            beta_path[i] = beta_cur.copy()    

        return beta_path



#######################
#####  covariance #####
#######################



class dp_cv_grid_covariance(dp_cv_grid):
    def __init__(self, grid_n, lam_list, gamma_list, smooth = 10, buffer = 10):
        self.grid_n = grid_n
        self.lam_list = lam_list
        self.gamma_list = gamma_list
        self.smooth = smooth
        self.buffer = buffer
    
    def estimate_func(self, Y, lam):
        return Y.T @ Y / Y.shape[0]

    def loss_func(self, Y, sig):
        '''
        currently the penalty is not included in the value of loss
        '''
        if np.linalg.det(sig) < 1e-7:
            sig_reg = sig + 0.1 * np.eye(Y.shape[1])
        else:
            sig_reg = sig.copy()
        omega = np.linalg.inv(sig_reg)
        return np.trace(omega @ (Y.T @ Y)) + Y.shape[0] * np.log(np.linalg.det(sig_reg))

class dcdp_cv_grid_covariance(dcdp_cv_grid):
    def __init__(self, grid_n, lam_list, gamma_list, smooth = 10, 
                 buffer = 10, step_refine = 1, buffer_refine = 10, lam_refine = 0.1):
        self.grid_n = grid_n
        self.lam_list = lam_list
        self.gamma_list = gamma_list
        
        self.smooth = smooth
        self.buffer = buffer
        
        self.step_refine = step_refine
        self.buffer_refine = buffer_refine
        self.lam_refine = lam_refine

    def joint_estimate_func(self, Y1, Y2, lam):
        """
        may also try group lasso
        """
        beta_hat1 = self.estimate_func(Y1, lam)
        
        beta_hat2 = self.estimate_func(Y2, lam)
        return beta_hat1, beta_hat2
        
    def estimate_func(self, Y, lam):
        return Y.T @ Y / Y.shape[0]

    def loss_func(self, Y, sig):
        '''
        currently the penalty is not included in the value of loss
        '''
        if np.linalg.det(sig) < 1e-7:
            sig_reg = sig + 0.1 * np.eye(Y.shape[1])
        else:
            sig_reg = sig.copy()
        omega = np.linalg.inv(sig_reg)
        return np.trace(omega @ (Y.T @ Y)) + Y.shape[0] * np.log(np.linalg.det(sig_reg))
    
    def fit_with_cp(self, Y, cp, lam = 0.1):
        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))
        K = len(cp) - 1
        p = Y.shape[1]
        beta_path = np.zeros((K, p, p))

        for i in range(K):
            Y_cur = Y[cp[i]:cp[i + 1]]
            beta_cur = self.estimate_func(Y_cur, lam)
            beta_path[i] = beta_cur.copy()    

        return beta_path




class dp_cv_grid_covariate():
    """
    DP for data with response Y and covariate X

    """
    def __init__(self, grid_n, lam_list, gamma_list, smooth = 10, buffer = 10):
        self.grid_n = grid_n
        self.lam_list = lam_list
        self.gamma_list = gamma_list
        self.smooth = smooth
        self.buffer = buffer
        
    def estimate_func(self, y, X, lam):
        pass

    def loss_func(self, Y, X, beta):
        pass
 
    def goodness_of_fit(self, data_train, data_test, cp_loc, lam, gamma):
        '''
        currently the penalty is not included in the value of goodness-of-fit
        '''
        Y_train, X_train = data_train
        Y_test, X_test = data_test
        
        n = len(Y_train)
        p = X_train.shape[1]
        cp_loc = np.concatenate([[0], cp_loc, [n]])
        cp_loc = cp_loc.astype(int)
        res = 0
        for i in range(len(cp_loc) - 1):
            if cp_loc[i + 1] > cp_loc[i] + np.log(p) + 1:
                Y_sub = Y_train[cp_loc[i]:cp_loc[i + 1]]
                X_sub = X_train[cp_loc[i]:cp_loc[i + 1]]
                estimate = self.estimate_func(Y_sub, X_sub, lam)
                
                Y_sub = Y_test[cp_loc[i]:cp_loc[i + 1]]
                X_sub = X_test[cp_loc[i]:cp_loc[i + 1]]
                res += self.loss_func(Y_sub, X_sub, estimate)
            else:
                res += 0
        return res
        
    def fit(self, data_train, data_test):
#         Y_train, X_train = data_train
#         Y_test, X_test = data_test

        lam_gamma_list = list(itertools.product(self.lam_list, self.gamma_list))
        fit_res_best = np.infty
        cp_best = None
        param_best = None

        n = len(data_train[0])
        grid_n = min(self.grid_n, n - 1)
        # step = max(n // self.grid_n, 1)
        # grid = np.arange(0, n, step)
        step = n / (grid_n + 1)
        grid = np.floor(np.arange(1, grid_n + 1) * step).astype(int)

        for i, x in enumerate(lam_gamma_list):
            lam, gamma = x
            cp_loc, obj = self.dp_grid(data_train, grid, lam, gamma)
            fit_res = self.goodness_of_fit(data_train, data_test, cp_loc, lam, gamma)
            if fit_res < fit_res_best:
                cp_best = cp_loc
                param_best = x
                fit_res_best = fit_res
        return cp_best, param_best


    def dp_grid(self, data, grid, lam, gamma):
        """
        data: (Y, X)
        Y: n x -1 array
        cp_loc: for each ix in cp_loc, it means theta[ix - 1] != theta[ix]
        """
        Y, X = data
        
        n = len(Y)
        point_map = np.ones(n + 1) * -1
        B = np.ones(n + 1) * np.infty
        B[0] = gamma

        r_cand = np.unique(np.concatenate([[0], grid, [n]]))
        grid_ix = {x:i for i, x in enumerate(r_cand)}
        l_cand = np.unique(np.concatenate([[0], grid]))
        m = len(grid)

        for r in r_cand:
            for l in l_cand:
                if l >= r or r - l < self.buffer:
                    break
                Y_sub = Y[l:r]
                X_sub = X[l:r]
                estimate = self.estimate_func(Y_sub, X_sub, lam)
                b = B[l] + gamma + self.loss_func(Y_sub, X_sub, estimate)
                if b < B[r]:
                    B[r] = b
                    point_map[r] = l

        k = n
        cp_loc, cp_val = set([]), []
        while k > 0:
            h = int(point_map[k])
            if h > 0:
                cp_loc.add(h)
            k = h
        cp_loc = np.array(sorted(cp_loc))
        return cp_loc, B[-1]

    def dp_grid_test(self, data, grid, lam, gamma):
        """
        data: (Y, X)
        Y: n x -1 array
        cp_loc: for each ix in cp_loc, it means theta[ix - 1] != theta[ix]
        """
        Y, X = data
        
        n = len(Y)
        point_map = np.ones(n + 1) * -1
        B = np.ones(n + 1) * np.infty
        B[0] = gamma

        r_cand = np.concatenate([[0], grid, [n]])
        grid_ix = {x:i for i, x in enumerate(r_cand)}
        l_cand = np.concatenate([[0], grid])
        m = len(grid)
        dp = [[np.infty] * (m + 2) for r in range(m + 1)]

        for r in r_cand:
            for l in l_cand:
                if l >= r:
                    break
                Y_sub = Y[l:r]
                X_sub = X[l:r]
                estimate = self.estimate_func(Y_sub, X_sub, lam)
                dp[grid_ix[l]][grid_ix[r]] = self.loss_func(Y_sub, X_sub, estimate)

        for r in r_cand:
            for l in l_cand:
                if l >= r:
                    break
                b = B[l] + gamma + dp[grid_ix[l]][grid_ix[r]]
                if b < B[r]:
                    B[r] = b
                    point_map[r] = l

        k = n
        cp_loc, cp_val = set([]), []
        while k > 0:
            h = int(point_map[k])
            if h > 0:
                cp_loc.add(h)
            k = h
        cp_loc = np.array(sorted(cp_loc))
        return cp_loc, B, dp   
    
    
    
    
class dcdp_cv_grid_covariate(dp_cv_grid_covariate):
    def __init__(self, grid_n, lam_list, gamma_list, smooth = 10, 
                 buffer = 10, step_refine = 1, buffer_refine = 10, lam_refine = 0.1):
        self.grid_n = grid_n
        self.lam_list = lam_list
        self.gamma_list = gamma_list
        
        self.smooth = smooth
        self.buffer = buffer
        
        self.step_refine = step_refine
        self.buffer_refine = buffer_refine
        self.lam_refine = lam_refine

    def joint_estimate_func(self, Y1, X1, Y2, X2, lam):
        pass
        
    def estimate_func(self, y, X, lam):
        pass
    
    def loss_func(self, Y, X, beta):
        pass
 
    def goodness_of_fit(self, data_train, data_test, cp_loc, lam, gamma):
        Y_train, X_train = data_train
        Y_test, X_test = data_test
        
        n = len(Y_train)
        p = X_train.shape[1]
        cp_loc = np.concatenate([[0], cp_loc, [n]])
        cp_loc = cp_loc.astype(int)
        res = 0
        for i in range(len(cp_loc) - 1):
            if cp_loc[i + 1] > cp_loc[i] + np.log(p) + 1:
                Y_sub = Y_train[cp_loc[i]:cp_loc[i + 1]]
                X_sub = X_train[cp_loc[i]:cp_loc[i + 1]]
                estimate = self.estimate_func(Y_sub, X_sub, lam)
                
                Y_sub = Y_test[cp_loc[i]:cp_loc[i + 1]]
                X_sub = X_test[cp_loc[i]:cp_loc[i + 1]]
                res += self.loss_func(Y_sub, X_sub, estimate)
            else:
                Y_sub = Y_test[cp_loc[i]:cp_loc[i + 1]]
                res += np.sum(Y_sub**2)
        return res
        
    def fit(self, data_train, data_test):
        lam_gamma_list = list(itertools.product(self.lam_list, self.gamma_list))
        fit_res_best = np.infty
        cp_best = None
        param_best = None

        n = len(data_train[0])
        grid_n = min(self.grid_n, n - 1)
        # step = max(n // self.grid_n, 1)
        # grid = np.arange(0, n, step)
        step = n / (grid_n + 1)
        grid = np.floor(np.arange(1, grid_n + 1) * step).astype(int)

        for i, x in enumerate(lam_gamma_list):
            lam, gamma = x
            cp_loc, obj = self.dp_grid(data_train, grid, lam, gamma)
            cp_loc_refined = self.local_refine(data_train, cp_loc)
            fit_res = self.goodness_of_fit(data_train, data_test, cp_loc_refined, lam, gamma)
            if fit_res < fit_res_best:
                cp_best = cp_loc_refined
                cp_best_cand = cp_loc
                param_best = x
                fit_res_best = fit_res
        return cp_best, param_best, cp_best_cand

    def local_refine(self, data, cp_cand):
        n = len(data[0])
        K = len(cp_cand)
        cp_cand = np.concatenate([[0], cp_cand[:], [n]])
        cp_loc = []
        for i in range(1, K + 1):
            if cp_cand[i - 1] + 2 * self.buffer_refine > cp_cand[i]:
                continue
            st, ed = int(2/3 * cp_cand[i - 1] + 1/3 * cp_cand[i]), int(2/3 * cp_cand[i + 1] + 1/3 * cp_cand[i])
            cp = self.screen_cp(data, st, ed)
            if cp is not None:
                cp_loc.append(cp)
        return cp_loc

    def screen_cp(self, data, st, ed):
        '''
        currently not using group lasso in refinement
        '''
        Y, X = data
        buffer_refine = self.buffer_refine
        step_refine = self.step_refine
        
        if st + 2 * buffer_refine + step_refine >= ed:
            return None
        n_step = (ed - st - 2 * buffer_refine - 1) // step_refine + 1

        loss_min = np.infty
        loc_min = None
        for i in range(1, n_step):
            loc = st + buffer_refine + int(i * step_refine)
            theta_1, theta_2 = self.joint_estimate_func(Y[st:loc],
                                                        X[st:loc],
                                                        Y[loc:ed],
                                                        X[loc:ed],self.lam_refine)
            loss_split = self.loss_func(Y[st:loc], 
                                        X[st:loc], theta_1)
            loss_split += self.loss_func(Y[loc:ed],
                                         X[loc:ed], theta_2)
            if loss_split < loss_min:
                loss_min = loss_split
                loc_min = st + buffer_refine + int(i * step_refine)
        return loc_min
    
    def local_refine_test(self, data, cp_cand):
        n = len(data[0])
        K = len(cp_cand)
        cp_cand = np.concatenate([[0], cp_cand[:], [n]])
        cp_loc = []
        for i in range(1, K + 1):
            if cp_cand[i - 1] + 2 * self.buffer_refine > cp_cand[i]:
                continue
            st, ed = int(2/3 * cp_cand[i - 1] + 1/3 * cp_cand[i]), int(2/3 * cp_cand[i + 1] + 1/3 * cp_cand[i])
            cp = self.screen_cp(data, st, ed)
            if cp is not None:
                cp_loc.append(cp)
        return cp_loc

    def screen_cp_test(self, data, st, ed):
        Y, X = data
        buffer_refine = self.buffer_refine
        step_refine = self.step_refine
        
        if st + 2 * buffer_refine + step_refine >= ed:
            return None
        n_step = (ed - st - 2 * buffer_refine - 1) // step_refine + 1

        loss_min = np.infty
        loc_min = None
        loss_list = []
        loc_list = []
        
        for i in range(1, n_step):
            loc = st + buffer_refine + int(i * step_refine)
            theta_1, theta_2 = self.joint_estimate_func(Y[st:loc],
                                                        X[st:loc],
                                                        Y[loc:ed],
                                                        X[loc:ed],self.lam_refine)
            loss_split = self.loss_func(Y[st:loc], 
                                        X[st:loc], theta_1)
            loss_split += self.loss_func(Y[loc:ed],
                                         X[loc:ed], theta_2)
            
            loss_list.append(loss_split)
            loc_list.append(loc)
            
            if loss_split < loss_min:
                loss_min = loss_split
                loc_min = st + buffer_refine + int(i * step_refine)
        return loc_min, loss_list, loc_list


class dp_cv_grid_linear(dp_cv_grid_covariate):
    def __init__(self, grid_n, lam_list, gamma_list, smooth = 100, buffer = 200):
        self.grid_n = grid_n
        self.lam_list = lam_list
        self.gamma_list = gamma_list
        self.smooth = smooth
        self.buffer = buffer
        
    def estimate_func(self, y, X, lam):
#         fit = lr_newton_solver_linear(X, y)

#         sm_model = sm.OLS(y, (X)).fit(disp=0)
#         beta_hat = sm_model.params

        lasso = linear_model.Lasso(alpha=lam, fit_intercept = False)
        lasso.fit(X, y)
        beta_hat = lasso.coef_
        return beta_hat.reshape((-1,1))

    def loss_func(self, Y, X, beta):
        '''
        currently the penalty is not included in the value of loss
        '''
        return np.sum((Y - X @ beta)**2)
    
    
class dcdp_cv_grid_linear(dcdp_cv_grid_covariate):
    def __init__(self, grid_n, lam_list, gamma_list, smooth = 100, 
                 buffer = 200, step_refine = 1, buffer_refine = 30, lam_refine = 0.1):
        self.grid_n = grid_n
        self.lam_list = lam_list
        self.gamma_list = gamma_list
        
        self.smooth = smooth
        self.buffer = buffer
        
        self.step_refine = step_refine
        self.buffer_refine = buffer_refine
        self.lam_refine = lam_refine

    def joint_estimate_func(self, Y1, X1, Y2, X2, lam):
        lasso = linear_model.Lasso(alpha=lam, fit_intercept = False)
        lasso.fit(X1, Y1)
        beta_hat1 = lasso.coef_
        
        lasso = linear_model.Lasso(alpha=lam, fit_intercept = False)
        lasso.fit(X2, Y2)
        beta_hat2 = lasso.coef_
        return beta_hat1.reshape((-1,1)), beta_hat2.reshape((-1,1))
        
    def estimate_func(self, y, X, lam):
#         fit = lr_newton_solver_linear(X, y)
        lasso = linear_model.Lasso(alpha=lam, fit_intercept = False)
        lasso.fit(X, y)
        beta_hat = lasso.coef_
        return beta_hat.reshape((-1,1))

    def loss_func(self, Y, X, beta):
        '''
        currently the penalty is not included in the value of loss
        '''
        return np.sum((Y - X @ beta)**2)

