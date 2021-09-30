from tqdm import tqdm

class SGPBase:
    def __init__(self):
        pass

    def teach_bootstrap(self, sgp_dataset, config):
        '''
        Teach without deep models. Use classical FPFH/SIFT features.
        '''
        for i, data in tqdm(enumerate(sgp_dataset)):
            src_data, dst_data, src_info, dst_info, pair_info = data

            label, pair_info = self.perception_bootstrap(
                src_data, dst_data, src_info, dst_info, config)
            sgp_dataset.write_pseudo_label(i, label, pair_info)

    def teach(self, sgp_dataset, model, config):
        '''
        Teach with deep models. Use learned FCGF/CAPS features.
        '''
        for i, data in tqdm(enumerate(sgp_dataset)):
            src_data, dst_data, src_info, dst_info, pair_info = data

            # if self.is_valid(src_info, dst_info, pair_info):
            label, pair_info = self.perception(src_data, dst_data, src_info,
                                               dst_info, model, config)
            sgp_dataset.write_pseudo_label(i, label, pair_info)

    def learn(self, sgp_dataset, config):
        # Adapt and dispatch training script to external implementations
        self.train_adaptor(sgp_dataset, config)

    # override
    def train_adaptor(self, sgp_dataset, config):
        pass

    # override
    def perception_bootstrap(self, src_data, dst_data, src_info, dst_info):
        pass

    # override
    def perception(self, src_data, dst_data, src_info, dst_info, model):
        pass

