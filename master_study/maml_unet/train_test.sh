#!/usr/bin/env zsh
# python -u synthesis_train_normal.py >> train.log &&
python -u synthesis_test.py --result predict_result_withMaml/test_test/ --gpu  --dir_checkpoint run_1 >> test_test_withMaml.log &&
python -u synthesis_test.py --test_data val --result predict_result_withMaml/test_val/ --gpu  --dir_checkpoint run_2 >> test_val_withMaml.log &&
python -u synthesis_test.py --load run_history/non_maml_1/best_epoch.pth --result predict_result_withoutMaml/test_test/ --gpu  --dir_checkpoint run_3 >> test_test_withoutMaml.log &&
python -u synthesis_test.py --load run_history/non_maml_1/best_epoch.pth --test_data val --result predict_result_withoutMaml/test_val/ --gpu  --dir_checkpoint run_4 >> test_val_withoutMaml.log