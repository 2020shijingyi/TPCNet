from eval import *
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str,default=r'E:\Open_source\Anonymous_TPCNet\option\TPCNet_LOLV2_S.yml',
                        help='The yaml file for the train network')
    parser.add_argument('--weight_path', type=str,default=r'E:\Open_source\Anonymous_TPCNet\Weight\LOLV2_S.pth',
                        help='The weight_path for the train network')
    parser.add_argument('--save_path', type=str,default=r'repect\\testlolv2s',
                        help='The output save path')
    parser.add_argument('--target_path', type=str,default=r'F:\LOLv2\Synthetic\Test\Normal',
                        help='The target path ')
    parser.add_argument('--alpha_color', type=float,default=None,help='The parameter for the color transform')

    parser.add_argument('--test_type', type=str, default='R', help='test type can choose R;woR;Unpaired')

    parser.add_argument('--use_GT_mean', action='store_true', help='start GT_mean ')

    parser.add_argument('--Unpaired_metrics', type=str, nargs='+',
                        help='niqe,musiq,pi')
    parser.add_argument('--Unpaired', action='store_true',
                        help='the command is used to Unpaired dataset reconstruction')
    parser.add_argument('--Unpaired_path', type=str,default='Unpaired_path',
                        help='Unpaired path')

    parser.add_argument('--TotalAvg', action='store_true',
                        help='calculating the avg metrics in all unpaired datasets (DICM, NPE...)')
    args = parser.parse_args()

    args_dict = vars(args)  # Namespace -> dict
    test_type = args_dict.pop('test_type', None)

    if test_type =='R':
        eval_and_reconstruction(yaml_path=args.yaml_path,weight_path = args.weight_path,target_path = args.target_path,alpha_color =
        args.alpha_color,use_GT_mean = args.use_GT_mean,save_paths = args.save_path,Unpaired=args.Unpaired)
    elif test_type =='woR':
        eval_wiout_reconstruction(save_paths = args.save_path,target_path=args.target_path,use_GT_mean=args.use_GT_mean)
    elif test_type=='Unpaired':
        eval_unpaired(Unpaired_path=args.Unpaired_path,TotalAvg=args.TotalAvg,metrics=args.Unpaired_metrics)
