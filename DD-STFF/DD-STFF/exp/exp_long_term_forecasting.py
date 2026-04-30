import random

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from sklearn.metrics import r2_score
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

import psutil


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.update(loss.item(), batch_x.size(0))
        total_loss = total_loss.avg
        self.model.train()
        return total_loss


    def train(self, setting):
        # train_data (8449)  train_loader(265)
        train_data, train_loader = self._get_data(flag='train')
        # vali_data (2785)  train_loader(88)
        vali_data, vali_loader = self._get_data(flag='val')
        # vali_data (2785)  train_loader(88)
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)  # 265
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # # Define a function to save results to a txt file
        def save_results_to_file(epoch, train_steps, train_loss, vali_loss, test_loss, time_cost, file_path):
            with open(file_path, 'a', encoding='utf-8', errors='replace') as f:
                f.write(
                    f"Epoch: {epoch + 1}, Steps: {train_steps}, Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}, Time Cost: {time_cost:.4f}s\n")

        fix_seed = 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        torch.set_num_threads(6)

        #  # Record start time and memory usage
        start_time = time.time()
        process = psutil.Process()  # Get current process information
        start_memory = process.memory_info().rss  # Get initial memory usage

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()

            # The purpose of the for loop is to iterate over the batch data generated by the train_loader.
            # batch_x(32,96,7), batch_y(32,144,7), batch_x_mark(32,96,4), batch_y_mark(32,144,4)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # (32,144,7)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        print('OUTPUTS SHAPE',outputs.shape)
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)  # batch_y(32,144,7),pred_len 是预测的时间步数（即 144）

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        # （32,96,7）
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]  #32 96 7
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  #32 96 7
                    loss = criterion(outputs, batch_y)

                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_float))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)  # 通过早停保存下来的最佳模型，在测试集上进行评估。这确保了测试是在最优模型参数下进行的。

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # Save results to txt file
            time_cost = time.time() - epoch_time
            save_results_to_file(epoch, train_steps, train_loss, vali_loss, test_loss, time_cost,
                                 './result/results.txt')

# 代码中使用了 EarlyStopping 类，这个机制会在验证集的损失（vali_loss）不再改善时停止训练，并保存当前模型的最佳参数。
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Training ends, record end time and memory usage
        end_time = time.time()
        end_memory = process.memory_info().rss   # Get final memory usage

        # Calculate total time and memory usage
        total_time = end_time - start_time
        total_memory = end_memory - start_memory  # Calculate memory change

        print(f"Total Training Time: {total_time:.2f} seconds")
        print(f"Total Memory Usage: {total_memory / 1024:.2f} MB")  # Convert to MB

        # Write results to file
        with open('./result/results.txt', 'a', encoding='utf-8') as f:
            f.write(f"Total Training Time: {total_time:.2f} seconds\n")
            f.write(f"Total Training Memory Usage: {total_memory / 1024:.2f} MB\n")

        # Load the best model
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)
        return self.model

    def test(self, setting, test=0):
        # fix_seed = 2021
        # random.seed(fix_seed)
        # torch.manual_seed(fix_seed)
        # np.random.seed(fix_seed)
        # torch.set_num_threads(6)

        # Record start time and memory usage
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB

        test_data, test_loader = self._get_data(flag='test')

        #   'checkpoint.pth' 是通过 EarlyStopping 或其他方法在训练过程中保存的模型参数文件。
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        mse = AverageMeter()
        mae = AverageMeter()

        predictions = []  # Store predictions
        ground_truth = []  # Store ground truth

        total_inference_time = 0   # Initialize total inference time
        total_memory_used = 0   # Initialize total inference time

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                # Record the start time of the current batch inference
                batch_start_time = time.time()

                # Record current memory usage
                process = psutil.Process(os.getpid())
                batch_start_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                predictions.append(outputs.cpu().numpy())  # Save predictions
                ground_truth.append(batch_y.cpu().numpy())  # Save ground truth

                # Record the end time of the current batch inference
                batch_end_time = time.time()
                # Record current memory usage
                batch_end_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
                # Accumulate current batch inference time
                total_inference_time += (batch_end_time - batch_start_time)
                # Calculate current batch memory usage
                total_memory_used += (batch_end_memory - batch_start_memory)
                # Save ground truth

                mse.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))

                # After processing all batches, print total inference time and memory usage, and save to file
        print(f"Total Inference Time: {total_inference_time:.2f}S")
        print(f"Total  Inference Memory Usage: {total_memory_used:.2f} MB")

            # Save results to text file
        result_file_path = './result/results.txt'
        with open(result_file_path, 'a', encoding='utf-8', errors='replace') as file:
                file.write(f'Total Inference Time: {total_inference_time:.2f}S\n')
                file.write(f'Total Memory Usage: {total_memory_used:.2f} MB\n')

        mse = mse.avg
        mae = mae.avg

        # Record end time and memory usage
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB

        predictions = np.concatenate(predictions, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)

        # 计算总的R^2
        tss = np.sum((ground_truth - np.mean(ground_truth)) ** 2)  # 总变异
        rss = np.sum((ground_truth - predictions) ** 2)  # 残差平方和
        r2 = 1 - rss / tss
        print("整体 R^2:", r2)

        # # 计算R方
        # r2 = r2_score(ground_truth, predictions)
        # print("R² (决定系数):", r2)

        predicted_values = predictions[1, :, 0]
        true_values = ground_truth[1, :, 0]

        # import matplotlib.pyplot as plt
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False

        # lines = plt.plot(true_values, label='True_values')
        # lines = plt.plot(predicted_values, label='Predicted_values')
        #
        # # Set font size
        # plt.rcParams['font.size'] = 18
        #
        # legend = plt.legend(fontsize=16)
        # for text in legend.get_texts():
        #     text.set_fontsize(16)
        #
        # plt.xlabel('Time', fontsize=18)
        # plt.ylabel('Value', fontsize=18)
        # plt.title('DD-STFF', fontsize=18)
        #
        # # Get current axis object
        # ax = plt.gca()
        # # Set axis tick font size
        # ax.tick_params(axis='both', labelsize=18)
        #
        # # # Adjust left margin to move the chart to the right
        # plt.subplots_adjust(left=0.2)  # Try different values
        # # # Adjust bottom margin to move the chart up
        # plt.subplots_adjust(bottom=0.2)  # Try different values
        #
        # # # Set subplot margins to 0 to make the image fill the entire plotting area
        # # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #
        # # save the plot to file
        # plt.savefig('./imgsDD-STFF', bbox_inches='tight')  # 指定路径和文件名
        #
        # plt.show()

        # Save results to file
        with open('./result/results.txt', 'a', encoding='utf-8', errors='replace') as file:
            file.write(f'MSE: {mse}\n')
            file.write(f'MAE: {mae}\n')

            file.write(f'R²: {r2}\n')
            file.write(f'Total test time: {end_time - start_time:.2f} seconds\n')
            file.write(f'test Memory used: {end_memory - start_memory:.2f} MB\n')
        print('mse:{}, mae:{}, r2:{}'.format(mse, mae, r2))



        # the total number of parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)


        with open('./result/results.txt', 'a') as f:
            f.write(f'Test the total number of parameters.: {total_params}\n')

        return
