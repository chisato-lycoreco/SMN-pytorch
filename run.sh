echo "*-----------------------LINK START-----------------------*"
#

#SMN-last
python run_train.py --model_type smn --fusion_type last --do_train   --overwrite_output_dir --per_gpu_train_batch_size 80 --per_gpu_eval_batch_size 80 --max_seq_length 50 --learning_rate 0.001 --num_train_epochs 1 --data_dir /home/gong/zty/MTRSDataSet/Ubuntu_SMN/ --output_dir ./output/
wait
# 11/10/2020 11:24:46 - INFO - train_utils -   Training Epoch - 1
# 1249it [04:57,  4.28it/s]11/10/2020 11:29:44 - INFO - train_utils -   the current train_steps is 1250
# 11/10/2020 11:29:44 - INFO - train_utils -   the current logging_loss is 0.3995216190814972
# 2499it [09:49,  4.29it/s]11/10/2020 11:34:36 - INFO - train_utils -   the current train_steps is 2500
# 11/10/2020 11:34:36 - INFO - train_utils -   the current logging_loss is 0.4114433228969574
# 3749it [14:41,  4.33it/s]11/10/2020 11:39:28 - INFO - train_utils -   the current train_steps is 3750
# 11/10/2020 11:39:28 - INFO - train_utils -   the current logging_loss is 0.23701295256614685
# 4999it [19:33,  4.22it/s]11/10/2020 11:44:21 - INFO - train_utils -   the current train_steps is 5000
# 11/10/2020 11:44:21 - INFO - train_utils -   the current logging_loss is 0.31664037704467773
# 6249it [24:26,  4.28it/s]11/10/2020 11:49:13 - INFO - train_utils -   the current train_steps is 6250
# 11/10/2020 11:49:13 - INFO - train_utils -   the current logging_loss is 0.253940612077713
# 11/10/2020 11:49:13 - INFO - train_utils -   --------------------------------- Model Evaulation ---------------------------------
# 3125it [05:06, 10.20it/s]
# 11/10/2020 11:54:20 - INFO - train_utils -     R10@1 = 0.73156
# 11/10/2020 11:54:20 - INFO - train_utils -     R10@2 = 0.85294
# 11/10/2020 11:54:20 - INFO - train_utils -     R10@5 = 0.96228

#SMN-static
python run_train.py --model_type smn --fusion_type static --do_train  --overwrite_output_dir --per_gpu_train_batch_size 80 --per_gpu_eval_batch_size 80 --max_seq_length 50 --learning_rate 0.001 --num_train_epochs 1 --data_dir /home/gong/zty/MTRSDataSet/Ubuntu_SMN/ --output_dir ./output/
wait
# 11/10/2020 13:38:05 - INFO - train_utils -   Training Epoch - 1
# 1249it [04:54,  4.15it/s]11/10/2020 13:43:00 - INFO - train_utils -   the current train_steps is 1250
# 11/10/2020 13:43:00 - INFO - train_utils -   the current logging_loss is 0.3853358328342438
# 2499it [09:46,  4.27it/s]11/10/2020 13:47:52 - INFO - train_utils -   the current train_steps is 2500
# 11/10/2020 13:47:52 - INFO - train_utils -   the current logging_loss is 0.3313216269016266
# 3749it [14:38,  4.28it/s]11/10/2020 13:52:44 - INFO - train_utils -   the current train_steps is 3750
# 11/10/2020 13:52:44 - INFO - train_utils -   the current logging_loss is 0.38397884368896484
# 4999it [19:31,  4.25it/s]11/10/2020 13:57:37 - INFO - train_utils -   the current train_steps is 5000
# 11/10/2020 13:57:37 - INFO - train_utils -   the current logging_loss is 0.4128851294517517
# 6249it [24:24,  4.25it/s]11/10/2020 14:02:30 - INFO - train_utils -   the current train_steps is 6250
# 11/10/2020 14:02:30 - INFO - train_utils -   the current logging_loss is 0.2757640480995178
# 11/10/2020 14:02:30 - INFO - train_utils -   --------------------------------- Model Evaulation ---------------------------------
# 3125it [05:04, 10.27it/s]
# 11/10/2020 14:07:34 - INFO - train_utils -     R10@1 = 0.73018
# 11/10/2020 14:07:34 - INFO - train_utils -     R10@2 = 0.85008
# 11/10/2020 14:07:34 - INFO - train_utils -     R10@5 = 0.96188

#SMN-dynamic
python run_train.py --model_type smn --fusion_type dynamic --do_train  --overwrite_output_dir --per_gpu_train_batch_size 80 --per_gpu_eval_batch_size 80 --max_seq_length 50 --learning_rate 0.001 --num_train_epochs 1 --data_dir /home/gong/zty/MTRSDataSet/Ubuntu_SMN/ --output_dir ./output/

#11/10/2020 00:19:36 - INFO - train_utils -   Training Epoch - 1
#11/10/2020 00:24:29 - INFO - train_utils -   the current train_steps is 1250
#11/10/2020 00:24:29 - INFO - train_utils -   the current logging_loss is 0.31432414054870605
#11/10/2020 00:29:22 - INFO - train_utils -   the current train_steps is 2500
#11/10/2020 00:29:22 - INFO - train_utils -   the current logging_loss is 0.4119081497192383
#11/10/2020 00:34:14 - INFO - train_utils -   the current train_steps is 3750
#11/10/2020 00:34:14 - INFO - train_utils -   the current logging_loss is 0.34874385595321655
#11/10/2020 00:39:08 - INFO - train_utils -   the current train_steps is 5000
#11/10/2020 00:39:08 - INFO - train_utils -   the current logging_loss is 0.2641168534755707
#11/10/2020 00:44:01 - INFO - train_utils -   the current train_steps is 6250
#11/10/2020 00:44:01 - INFO - train_utils -   the current logging_loss is 0.3406740725040436
#11/10/2020 00:44:01 - INFO - train_utils -   --------------------------------- Model Evaulation #---------------------------------
#11/10/2020 00:55:43 - INFO - train_utils -     R10@1 = 0.72968
#11/10/2020 00:55:43 - INFO - train_utils -     R10@2 = 0.85318
#11/10/2020 00:55:43 - INFO - train_utils -     R10@5 = 0.9623

echo "DONE!"