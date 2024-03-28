import argparse

class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='LLM-Prop')
        
        self.parser.add_argument('--epochs',
                            help='Number of epochs',
                            type=int,
                            default=200)
        self.parser.add_argument('--bs',
                            help='Batch size',
                            type=int,
                            default=16)
        self.parser.add_argument('--lr',
                            help='Learning rate',
                            type=float,
                            default=0.002
                            ) # 0.001
        self.parser.add_argument('--max_len',
                            help='Max input sequence length',
                            type=int,
                            default=256
                            )
        self.parser.add_argument('--dr',
                            help='Drop rate',
                            type=float,
                            default=0.5)
        self.parser.add_argument('--warmup_steps',
                            help='Warmpup steps',
                            type=int,
                            default=30000)
        
        self.parser.add_argument('--preprocessing_strategy',
                            help='Data preprocessing technique: "none", "bond_lengths_replaced_with_num", "bond_angles_replaced_with_ang", "no_stopwords", or "no_stopwords_and_lengths_and_angles_replaced"',
                            type=str,
                            default="none")
        
        self.parser.add_argument('--tokenizer',
                            help='Tokenizer name: "t5_tokenizer" or "modified"',
                            type=str,
                            default="t5_tokenizer")
        
        self.parser.add_argument('--pooling', 
                            help='Pooling method. "cls" or "mean"',
                            type=str,
                            default="cls")
        
        self.parser.add_argument('--normalizer', 
                            help='Labels scaling technique. "z_norm", "mm_norm", or "ls_norm"',
                            type=str,
                            default="z_norm") 
        
        self.parser.add_argument('--scheduler', 
                            help='Learning rate scheduling technique. "linear", "onecycle", "step", or "lambda" (no scheduling))',
                            type=str,
                            default="onecycle")
        
        self.parser.add_argument('--property_name', 
                            help='The name of the property to predict. "band_gap", "volume", or "is_gap_direct"',
                            type=str,
                            # default="band_gap"
                            default="exfoliation_en"
                            # default="log10(G_VRH)"
                            # default="log10(K_VRH)"
                            # default="last phdos peak"
                            # default="e_form"  
                            # default="n" 
                            )
        self.parser.add_argument('--optimizer', 
                            help='Optimizer type. "adamw" or "sgd"',
                            type=str,
                            default="adamw")
        self.parser.add_argument('--task_name', 
                            help='the name of the task: "regression" if propert_name is band_gap or volume or "classification" if property_name is is_gap_direct',
                            type=str,
                            default="regression")
        self.parser.add_argument('--train_data_path',
                            help="the path to the training data",
                            type=str,
                            # default="data/samples/textedge_prop_mp22_train.csv"
                            default="data/allmatbench_jdft2d/matbench_jdft2d_train.csv"
                            # default="data/matbench_log_gvrh/matbench_log_gvrh_train.csv"
                            # default="data/allmatbench_phonons/matbench_phonons_train.csv"
                            # default="data/allmatbench_perovskites/matbench_perovskites_train.csv"
                            # default="data/matbench_dielectric/matbench_dielectric_train.csv"
                            # default="data/allmatbench_log_kvrh/matbench_log_kvrh_train.csv"
                            )
        
        self.parser.add_argument('--valid_data_path',
                            help="the path to the valid data",
                            type=str,
                            # default="data/samples/textedge_prop_mp22_valid.csv"
                            default="data/allmatbench_jdft2d/matbench_jdft2d_valid.csv"
                            # default="data/matbench_log_gvrh/matbench_log_gvrh_valid.csv"
                            # default="data/allmatbench_phonons/matbench_phonons_valid.csv"
                            # default="data/allmatbench_perovskites/matbench_perovskites_valid.csv"
                            # default="data/matbench_dielectric/matbench_dielectric_valid.csv"
                            # default="data/allmatbench_log_kvrh/matbench_log_kvrh_valid.csv"
                            )
        self.parser.add_argument('--test_data_path',
                            help="the path to the test data",
                            type=str,
                            # default="data/samples/textedge_prop_mp22_test.csv"
                            default="data/allmatbench_jdft2d/matbench_jdft2d_test.csv"
                            # default="data/matbench_log_gvrh/matbench_log_gvrh_test.csv"
                            # default="data/allmatbench_phonons/matbench_phonons_test.csv"
                            # default="data/allmatbench_perovskites/matbench_perovskites_test.csv"
                            # default="data/matbench_dielectric/matbench_dielectric_test.csv"
                            # default="data/allmatbench_log_kvrh/matbench_log_kvrh_test.csv"
                            )
        
        self.parser.add_argument('--all_data_path',
                            help="the path to the test data",
                            type=str,
                            # default="data/samples/textedge_prop_mp22.csv"
                            default="data/allmatbench_jdft2d/matbench_jdft2d.csv"
                            # default="data/matbench_log_gvrh/matbench_log_gvrh.csv"
                            # default="data/allmatbench_phonons/matbench_phonons.csv"
                            # default="data/allmatbench_perovskites/matbench_perovskites.csv"
                            # default="data/matbench_dielectric/matbench_dielectric.csv"
                            # default="data/allmatbench_log_kvrh/matbench_log_kvrh.csv"
                            )
        
        self.parser.add_argument('--checkpoint',
                            help="the path to the the best checkpoint for evaluation",
                            type=str,
                            default="") 
    
    def get_config(self):
        args, unknown = self.parser.parse_known_args()
        return vars(args)
    
    # def get_config(self):
    #         args = self.parser.parse_args()
    #         return vars(args)