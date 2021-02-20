import pandas as pd
import numpy as np
# import random
# import pickle
# from arm_class import ArmGaussian
from DLinTS import DLinTS
# from environment_class import Environment
# from simulator_class import Simulator
from data.loader import load_data


class Simulator(object):
    def __init__(self, init_params:dict):
        self.arms = {}
        self.init_params = init_params

    def get_arm(self, domain):
        if domain not in self.arms:
            self.arms[domain] = DLinTS(dim=self.init_params["dimension"],
                                       delta=self.init_params["delta"],
                                       alpha=self.init_params["alpha"],
                                       lambda_=self.init_params["lambda_"],
                                       gamma=self.init_params["gamma"], 
                                       sm=False,
                                       sigma_noise=self.init_params["sigma_noise"],
                                       verbose=self.init_params["verbose"])
        return self.arms[domain]

    def select_arm(self, features:np.ndarray):
        dim = features.shape[0]
        _max = -np.Inf
        _max_domain = ""
        _max_arm = None
        for (domain, arm) in self.arms.items():
            if arm.t < dim:
                ucb_s = np.Inf
            else:
                ucb_s = arm.hat_theta.dot(features)
            if ucb_s>_max:
                _max_domain = domain
                _max_arm = arm
        print("max_ucb:", _max, "selected domain:", _max_domain)
        return _max_arm, _max_domain


def train(features, targets, domains, simulator):
    for domain, feature, target in zip(domains, features, targets):
        arm = simulator.get_arm(domain)
        arm.update_state(feature.toarray(), target)
        simulator.arms[domain] = arm
    return simulator


def test(features, domains, simulator):
    win_rates = np.zeros(features.shape[0])
    for i, (domain, feature) in enumerate(zip(domains, features)):
        arm = simulator.get_arm(domain)
        win_rates[i] = arm.calculate_win_rate(feature.toarray())
    return win_rates


def main(train_args:dict, test_args:dict):
    # General parameters
    # q = 10 # Diplaying the quantile (in %)
    # steps = data.shape[0]  # number of steps for the experiment
    # k = data.domains.unique().shape[0]/2  # number of arms / 2
    # l = df.converted.value_counts().max()  # max(#click, #non_click)
    features, targets, domains, encoder = load_data(train_args)
    params = {"delta": 0.01,  # Probability of being outside the confidence interval
              "lambda_": 1,  # Regularisation parameter
              "alpha": 1, 
              "dimension": features.shape[1],  # feature dimension
              "sigma_noise": np.sqrt(0.15), # Square root of the variance of the noise
              "gamma": 1 - (2.0/(features.shape[1] * features.shape[0]))**(2/3), 
              "verbose": False}
    simulator = Simulator(params)
    train(features, targets.values, domains.values, simulator)

    test_args["encoder"] = encoder
    features, targets, domains, encoder = load_data(test_args)
    results = test(features, domains.values, simulator)
    print(results)


if __name__=="__main__":
    target = "converted"
    arm_key = "domain"
    select_cols = ["converted", "hour", "is_app", "creative_ad_type", "domain", "device", "slot_size", "categories", "advertiser_company_id", "imp_banner_pos", "imp_tagid"]
    category_cols = ["hour", "creative_ad_type", "is_app", "domain", "device", "slot_size", "categories", "advertiser_company_id", "imp_banner_pos", "imp_tagid"]
    drop_cols = ["hour", "imp_tagid", "is_app"]
    train_args = {"path":"./data/dummy_train.csv",
                  "target":target, 
                  "sort_key":"imp_time",
                  "arm_key":arm_key,
                  "select_cols":select_cols, 
                  "category_cols":drop_cols,
                  "drop_cols":drop_cols,
                  "is_app":"is_app",
                  "encoder":None}
    test_args = {"path":"./data/dummy_test.csv",
                  "target":target, 
                  "sort_key":"imp_time",
                  "arm_key":arm_key,
                  "select_cols":select_cols, 
                  "category_cols":drop_cols,
                  "drop_cols":drop_cols,
                  "is_app":"is_app"}
    main(train_args, test_args)