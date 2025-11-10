function [W,b,M]=train_by_Resnet(M,L,l,z1,z2,W,b,c)

%M hidden layer with L neuron

S=(1);           %scaling value for the inputs and outputs

J=size(z2,1);     %number of training points
tolerance_step=5*10^5;   %iterations to train the network

s=1;                     %"scaling value" for the weights, if initial weights are too big then system explodes


a=cell(M,1);
e_inner=zeros(1,J);
en=zeros(1,tolerance_step);
iteration_step=1;                    %initializing iteration number
for i=1:J
    a{1}=transfer(W{1}*z1{i}(:,1)+b{1});
    for k=2:M           %first run through the network, to see how everything looks initially
        a{k}=transfer(W{k}*a{k-1}+b{k});
    end
    a_out=z1{i}(c,1)+(W{M+1}*a{M}+b{M+1});        
    e_inner(i)=z2(i)-a_out;
end

en(1)=(1/J)*(norm(e_inner).^2);            %how we measure the error of our 2xJ matrix
n=cell(M,1);
s=cell(M+1,1);
Ms=cell(M,1);
while iteration_step<tolerance_step        %training our neural network
    for i=1:J
        %feeding forward
        n{1}=W{1}*z1{i}(:,1)+b{1};
        a{1}=transfer(n{1});
        for k=2:M
            n{k}=W{k}*a{k-1}+b{k};
            a{k}=transfer(n{k});
        end
        a_out=z1{i}(c,1)+(W{M+1}*a{M}+b{M+1});
        
        e_inner(i)=z2(i)-a_out;
        
        %backpropagating sensitivities
        s{M+1}=-2*e_inner(i);
        for k=1:M
            Ms{M}=zeros(L);
        end
        for j=1:L
            for k=M:-1:1
                Ms{k}(j,j)=transferd(n{k}(j));
            end
        end
  
        for k=M:-1:1
            s{k}=Ms{k}*W{k+1}'*s{k+1};
        end
        %updating weights and biases  
        for k=(M+1):-1:2
            W{k}=W{k}-(l*s{k}*a{k-1}');
            b{k}=b{k}-l*s{k};
        end
        W{1}=W{1}-l*s{1}*z1{i}(:,1)';
        b{1}=b{1}-l*s{1};
        %feedforward again
        a{1}=transfer(W{1}*z1{i}(:,1)+b{1});
        for k=2:M
            a{k}=transfer(W{k}*a{k-1}+b{k});
        end
        a_out=z1{i}(c,1)+(W{M+1}*a{M}+b{M+1});
        e_inner(i)=S*(z2(i)-a_out);
    end
    iteration_step=iteration_step+1;                                   %one iteration is one run through the entire training set
    en(iteration_step)=(1/J)*(norm(e_inner).^2);           %keep track of error after each iteration
end


% save('train_parameters_multi.mat','W','b','M');

nn=linspace(1,tolerance_step,tolerance_step);             %plotting results


figure
loglog(nn,en)
grid on
title(['error after ',num2str(iteration_step),' iterations'])
xlabel('iteration number')
ylabel('error')


