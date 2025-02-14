day01 - 10/02/2025 15:36

decided to use the chess1 and 2 files for training and touch 3 only for testing
that gives us around 4M patterns to play with

Model 1 :

Let us not use any features in basemodel1
for model1 let us just use normal nn 
give it 64 inputs , each square as input
for each piece,we give it it's material advantage(that cant differentiate between bishop and knite)

16:21 - done encoding for basemode1

21:17 - Done training and the best testing loss is at 1.5480 i.e 537033 which is very bad

day02 - 14/02/2025 14:19

basemodel 2 : a simple conv layer


encoding :
    12 planes for each piece type (K, k, P, p, B, b, N, n, R, r, Q, q)
    1 plane for empty spaces
    1 plane for active color (1 for white, -1 for black)
    4 planes for castling rights (K, Q, k, q)
    1 plane for en passant target square (1 at the target square, 0 elsewhere)
    1 plane for halfmove clock
    1 plane for fullmove number

no extra features

model arc : 

2 conv layers -> 3 linear hidden layers -> output


a sample run : 
"
Epoch [1/40], Training Loss: 1.0565
Epoch 1/40 - Testing:   0%|          | 0/32 [00:00<?, ?it/s]/home/suneel/anaconda3/lib/python3.12/site-packages/torch/nn/modules/loss.py:610: UserWarning: Using a target size (torch.Size([32])) that is different to the input size (torch.Size([32, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
Epoch 1/40 - Testing:  94%|█████████▍| 30/32 [00:00<00:00, 43.98it/s]/home/suneel/anaconda3/lib/python3.12/site-packages/torch/nn/modules/loss.py:610: UserWarning: Using a target size (torch.Size([8])) that is different to the input size (torch.Size([8, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
                                                                     
Epoch [1/40], Test Loss: 1.1991

Last 5 Predictions and Actuals:
Predicted: [-0.11214086], Actual: 0.130116805434227
Predicted: [0.02706032], Actual: -0.06185539439320564
Predicted: [-0.0984931], Actual: -0.6785448789596558
Predicted: [-0.1060049], Actual: 1.3940753936767578
Predicted: [-0.12783623], Actual: 0.4240211546421051
Test loss improved, saving model...
Model saved to /home/suneel/Desktop/chess-engine/models/basemodel2/best_model_epoch_1.pth
                                                                        
Epoch [2/40], Training Loss: 1.0314
                                                                     
Epoch [2/40], Test Loss: 1.1847

Last 5 Predictions and Actuals:
Predicted: [0.07361269], Actual: 0.130116805434227
Predicted: [-0.03392443], Actual: -0.06185539439320564
Predicted: [-0.03039206], Actual: -0.6785448789596558
Predicted: [-0.03083194], Actual: 1.3940753936767578
Predicted: [-0.03122261], Actual: 0.4240211546421051
Test loss improved, saving model...
Model saved to /home/suneel/Desktop/chess-engine/models/basemodel2/best_model_epoch_2.pth
                                                                        
Epoch [3/40], Training Loss: 1.0267
                                                                     
Epoch [3/40], Test Loss: 1.1783

Last 5 Predictions and Actuals:
Predicted: [0.05684524], Actual: 0.130116805434227
Predicted: [-0.02487345], Actual: -0.06185539439320564
Predicted: [-0.01136977], Actual: -0.6785448789596558
Predicted: [-0.01168436], Actual: 1.3940753936767578
Predicted: [-0.01749507], Actual: 0.4240211546421051
Test loss improved, saving model...
Model saved to /home/suneel/Desktop/chess-engine/models/basemodel2/best_model_epoch_3.pth
                                                                        
Epoch [4/40], Training Loss: 1.0262
                                                                     
Epoch [4/40], Test Loss: 1.1737

Last 5 Predictions and Actuals:
Predicted: [0.05384374], Actual: 0.130116805434227
Predicted: [-0.02157692], Actual: -0.06185539439320564
Predicted: [0.0502
"

and this is clearly seen that the model is predicting zeros for almost all the values

does initializing weights change it ?? - yes it did 

also added weight loss funtion log(1 + alpha*|target|) 

here is a sample run : "
Epoch [1/40] Training: 100%|██████████| 313/313 [00:26<00:00, 11.90it/s, train_loss=1.5805] 
Epoch [1/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 46.05it/s, test_loss=0.5467]

Saved new best model with test loss: 1.0038

Epoch [1/40], Train Loss: 8.7029, Test Loss: 1.0038

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = -0.1821, Actual = -0.0839
Sample 2: Predicted = 0.9728, Actual = -0.0279
Sample 3: Predicted = -0.4842, Actual = 0.1760
Sample 4: Predicted = 0.1676, Actual = -0.8060
Sample 5: Predicted = -0.3613, Actual = -0.3150
--------------------------------------------------
Epoch [2/40] Training: 100%|██████████| 313/313 [00:27<00:00, 11.22it/s, train_loss=0.6351]
Epoch [2/40] Testing: 100%|██████████| 32/32 [00:01<00:00, 30.89it/s, test_loss=0.5063]

Epoch [2/40], Train Loss: 1.1019, Test Loss: 1.0091

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = -0.1048, Actual = -0.0839
Sample 2: Predicted = 0.9912, Actual = -0.0279
Sample 3: Predicted = -0.5257, Actual = 0.1760
Sample 4: Predicted = 0.1443, Actual = -0.8060
Sample 5: Predicted = -0.0832, Actual = -0.3150
--------------------------------------------------
Epoch [3/40] Training: 100%|██████████| 313/313 [00:27<00:00, 11.44it/s, train_loss=0.3105]
Epoch [3/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 47.39it/s, test_loss=0.5975]

Epoch [3/40], Train Loss: 0.9043, Test Loss: 1.0254

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = -0.0475, Actual = -0.0839
Sample 2: Predicted = 0.9081, Actual = -0.0279
Sample 3: Predicted = -0.4285, Actual = 0.1760
Sample 4: Predicted = 0.1349, Actual = -0.8060
Sample 5: Predicted = 0.1977, Actual = -0.3150
--------------------------------------------------
Epoch [4/40] Training: 100%|██████████| 313/313 [00:26<00:00, 11.77it/s, train_loss=0.4801]
Epoch [4/40] Testing: 100%|██████████| 32/32 [00:01<00:00, 31.07it/s, test_loss=0.4911]

Saved new best model with test loss: 0.9617

Epoch [4/40], Train Loss: 0.8550, Test Loss: 0.9617

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = -0.0218, Actual = -0.0839
Sample 2: Predicted = 0.0623, Actual = -0.0279
Sample 3: Predicted = -0.2310, Actual = 0.1760
Sample 4: Predicted = 0.3201, Actual = -0.8060
Sample 5: Predicted = 0.2778, Actual = -0.3150
--------------------------------------------------
Epoch [5/40] Training: 100%|██████████| 313/313 [00:28<00:00, 11.16it/s, train_loss=0.4243]
Epoch [5/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 45.50it/s, test_loss=0.4211]

Saved new best model with test loss: 0.9339

Epoch [5/40], Train Loss: 0.7808, Test Loss: 0.9339

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = 0.0571, Actual = -0.0839
Sample 2: Predicted = 0.2508, Actual = -0.0279
Sample 3: Predicted = -0.3035, Actual = 0.1760
Sample 4: Predicted = 0.2704, Actual = -0.8060
Sample 5: Predicted = 0.1445, Actual = -0.3150
--------------------------------------------------
Epoch [6/40] Training: 100%|██████████| 313/313 [00:24<00:00, 12.64it/s, train_loss=0.3389]
Epoch [6/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 43.94it/s, test_loss=0.5332]

Epoch [6/40], Train Loss: 0.7396, Test Loss: 0.9891

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = 0.0886, Actual = -0.0839
Sample 2: Predicted = 0.1927, Actual = -0.0279
Sample 3: Predicted = -0.1517, Actual = 0.1760
Sample 4: Predicted = 0.5344, Actual = -0.8060
Sample 5: Predicted = 0.2196, Actual = -0.3150
--------------------------------------------------
Epoch [7/40] Training: 100%|██████████| 313/313 [00:26<00:00, 11.88it/s, train_loss=0.2547]
Epoch [7/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 45.21it/s, test_loss=0.5392]

Epoch [7/40], Train Loss: 0.7079, Test Loss: 0.9725

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = 0.0531, Actual = -0.0839
Sample 2: Predicted = 0.1793, Actual = -0.0279
Sample 3: Predicted = -0.3322, Actual = 0.1760
Sample 4: Predicted = 0.4286, Actual = -0.8060
Sample 5: Predicted = 0.2869, Actual = -0.3150
--------------------------------------------------
Epoch [8/40] Training: 100%|██████████| 313/313 [00:25<00:00, 12.43it/s, train_loss=0.3092]
Epoch [8/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 45.23it/s, test_loss=0.6185]

Epoch [8/40], Train Loss: 0.6977, Test Loss: 0.9936

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = 0.1515, Actual = -0.0839
Sample 2: Predicted = 0.6055, Actual = -0.0279
Sample 3: Predicted = -0.2183, Actual = 0.1760
Sample 4: Predicted = 0.4750, Actual = -0.8060
Sample 5: Predicted = 0.2704, Actual = -0.3150
--------------------------------------------------
Epoch [9/40] Training: 100%|██████████| 313/313 [00:26<00:00, 11.81it/s, train_loss=0.1798]
Epoch [9/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 42.36it/s, test_loss=0.6712]

Epoch [9/40], Train Loss: 0.6786, Test Loss: 1.0281

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = 0.0101, Actual = -0.0839
Sample 2: Predicted = 0.5495, Actual = -0.0279
Sample 3: Predicted = -0.0748, Actual = 0.1760
Sample 4: Predicted = 0.5800, Actual = -0.8060
Sample 5: Predicted = 0.3727, Actual = -0.3150
--------------------------------------------------
Epoch [10/40] Training: 100%|██████████| 313/313 [00:27<00:00, 11.29it/s, train_loss=0.1760]
Epoch [10/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 41.80it/s, test_loss=0.6836]

Epoch [10/40], Train Loss: 0.6503, Test Loss: 1.0144

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = -0.0501, Actual = -0.0839
Sample 2: Predicted = 0.6634, Actual = -0.0279
Sample 3: Predicted = -0.0063, Actual = 0.1760
Sample 4: Predicted = 0.6357, Actual = -0.8060
Sample 5: Predicted = 0.3473, Actual = -0.3150
--------------------------------------------------
Epoch [11/40] Training: 100%|██████████| 313/313 [00:27<00:00, 11.35it/s, train_loss=0.1604]
Epoch [11/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 42.86it/s, test_loss=0.6016]

Epoch [11/40], Train Loss: 0.6216, Test Loss: 0.9756

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = -0.0725, Actual = -0.0839
Sample 2: Predicted = 0.4669, Actual = -0.0279
Sample 3: Predicted = 0.0342, Actual = 0.1760
Sample 4: Predicted = 0.6050, Actual = -0.8060
Sample 5: Predicted = 0.2610, Actual = -0.3150
--------------------------------------------------
Epoch [12/40] Training: 100%|██████████| 313/313 [00:28<00:00, 11.10it/s, train_loss=0.1803]
Epoch [12/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 43.24it/s, test_loss=0.6053]

Epoch [12/40], Train Loss: 0.6193, Test Loss: 0.9532

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = -0.1461, Actual = -0.0839
Sample 2: Predicted = 0.0331, Actual = -0.0279
Sample 3: Predicted = 0.1017, Actual = 0.1760
Sample 4: Predicted = 0.6589, Actual = -0.8060
Sample 5: Predicted = 0.1511, Actual = -0.3150
--------------------------------------------------
Epoch [13/40] Training: 100%|██████████| 313/313 [00:25<00:00, 12.12it/s, train_loss=0.2857]
Epoch [13/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 44.22it/s, test_loss=0.5598]

Saved new best model with test loss: 0.9088

Epoch [13/40], Train Loss: 0.6034, Test Loss: 0.9088

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = -0.2244, Actual = -0.0839
Sample 2: Predicted = 0.3490, Actual = -0.0279
Sample 3: Predicted = 0.2271, Actual = 0.1760
Sample 4: Predicted = 0.5400, Actual = -0.8060
Sample 5: Predicted = -0.1063, Actual = -0.3150
--------------------------------------------------
Epoch [14/40] Training: 100%|██████████| 313/313 [00:26<00:00, 11.62it/s, train_loss=0.2635]
Epoch [14/40] Testing: 100%|██████████| 32/32 [00:00<00:00, 34.24it/s, test_loss=0.6274]

Saved new best model with test loss: 0.8989

Epoch [14/40], Train Loss: 0.5907, Test Loss: 0.8989

Last 5 test samples for this epoch:
Predictions vs Actuals:
Sample 1: Predicted = -0.3468, Actual = -0.0839
Sample 2: Predicted = 0.1813, Actual = -0.0279
Sample 3: Predicted = 0.2337, Actual = 0.1760
Sample 4: Predicted = 0.4882, Actual = -0.8060
Sample 5: Predicted = -0.2195, Actual = -0.3150
"
ok so trying for alpha = 0,1 for today and also cleaned the data and removed outliner(eval_std > 3)

17 : 06 : 
running them on kaggle alpha = 0 : basemodel2alpha0
                       alpha = 1 : basemodel2alpha1
