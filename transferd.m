function m=transferd(x)
% if x<0
%     m=0;
% else
%     m=1;
% end
m=1-(tanh(x)).^2;