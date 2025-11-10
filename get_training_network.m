function get_training_network

M=1;                     
L=10;                    %number of neurons in each layer

l=0.001;                 %learning rate


filepath='C:\Users\DELL EMC\Desktop\Machine_learning\our PDE\code\hyperbolic\different initials\for analysis-linear-test2\';

load([filepath 'training_data' num2str(1) '.mat'],'p_input','p_target','xL','xR','c1','c','N_m','N_t','T','N_tm')


dim_input=size(p_input{1},1); 
dim_output=1;

W=cell(M+1,1);
b=cell(M+1,1);

% W{1}=randn(L,dim_input)*sqrt(1/dim_input);
% b{1}=zeros(L,1);
% W{M+1}=randn(dim_output,L)*sqrt(1/L);
% b{M+1}=zeros(1,1);

W{1}=randn(L,dim_input)*0.01;
b{1}=zeros(L,1);
W{M+1}=randn(dim_output,L)*0.01;
b{M+1}=zeros(1,1);
for k=2:M
    W{k}=randn(L)*0.01;
    b{k}=randn(L,1)*0.01;
end

for i=1:N_tm  
    load([filepath 'training_data' num2str(i) '.mat'],'p_input','p_target')
    z1=p_input;
    z2=p_target;
     
    [W,b,M]=train_by_Resnet(M,L,l,z1,z2,W,b,c);
end
  save('train_parameters','W','b','M');
