from .test_options import TestOptions

class GeneratorOptions(TestOptions):
    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)
        parser.add_argument('--use_dataset', type=bool, default=False, help='target to generate')
        parser.add_argument('--result_dir', type=str, default='./generator/lcs')
        return parser
