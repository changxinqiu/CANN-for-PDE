error_l2=[8.0425e-03 4.2878e-03 2.0172e-03 1.0628e-03 5.3261e-04];

error_h1=[2.5872e-03 1.7697e-03 8.1254e-04 4.5226e-04 2.2907e-04];

N_n=[5 10 20 40 80];
n=size(N_n,2);
order_l2=zeros(size(N_n,2)-1,1);
order_h1=zeros(size(N_n,2)-1,1);

for i=1:n-1
    order_l2(i)=(log(error_l2(i+1))-log(error_l2(i)))/(log(N_n(i))-log(N_n(i+1)));
    order_h1(i)=(log(error_h1(i+1))-log(error_h1(i)))/(log(N_n(i))-log(N_n(i+1)));
end
order_l2
order_h1