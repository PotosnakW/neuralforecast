def get_data_parameters(args):
    exog = {}
        
    if args.dataset == 'simglucose':
        data_dir = '../datasets/simglucose_exog_9_day_test.csv'
        static_dir = '../datasets/simglucose_static.csv'
        val_size = 2592
        test_size = 2592
        freq = '5min'
        exog['stat_exog_list'] = None
        exog['hist_exog_list'] = None
        exog['futr_exog_list'] = None
        
    if args.dataset == 'ohiot1dm':
        data_dir = '../datasets/ohiot1dm_exog_9_day_test.csv'
        static_dir = '../datasets/ohiot1dm_static.csv'
        val_size = 2691
        test_size = 2691
        freq = '5min'
        exog['stat_exog_list'] = None
        exog['hist_exog_list'] = None
        exog['futr_exog_list'] = None

    return data_dir, static_dir, val_size, test_size, freq, exog
