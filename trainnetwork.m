%%
%--------------------------------------------------------------------------------------------------------
% This code was submitted in the requirment to submi the manuscript of C&G
% jounal
% 20th January 2020
%--------------------------------------------------------------------------------------------------------
%%
% A novel method to improve vertical accuracy of CARTOSAT DEM using machine learning models
% Kasi Venkatesh1, Yeditha Pavan Kumar1, Rathinasamy Maheswaran1, Ankit Agarwal2, Pinninti Ramdas1, Landa Sankar Rao1, and Sangamreddi. Chandramouli.
% 1Department of Civil Engineering, MVGR College of Engineering, Vizianagaram, 535005, India
% 2Department of Hydrology, Indian Institute of Technology Roorkee, 247667, India
% 
% *Corresponding Author email: ankitfhy@iitr.ac.in 


% code for feed forward and narx neural network
% training and validation 
% Department of Civil enfineering, MVGR college of Engg
% Author: Pavan kumar yeditha, Rathinasamy Maheswaran
% made: october 2019
% updated: december 2019

%------------------------------------------------------------------------------------------------------
function [y,z]=trainnetwork(ip,ta,jp,n,itr)
% ip= training data set
% ta=training data set
% jp=testing data set
% itr= no of times the model is to run
for i=1:itr
 net1 = feedforwardnet(n);
%  n= no of neurons used for training the network 
 net1.trainFcn = 'trainlm'
%  training function= trainnlm(Levenberg Marquardt)
net1.trainParam.max_fail = 100;
%  max fails =100(default)
 net1.trainParam.min_grad=1e-7;
 net1.trainParam.show=25;
  net1.trainParam.lr=0.9;
 net1.trainParam.epochs=1000;
%  no of epchos =100(default)
     net1.trainParam.goal=0;
  net = train(net1,ip',ta');
 y(:,i)=net(ip')';
 z(:,i)=net(jp');
end

%---------------------------------------------------------------------------------------------------------------
%%