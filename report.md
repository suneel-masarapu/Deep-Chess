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